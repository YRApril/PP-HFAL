import time

import torch
import torch.nn.functional as F

from client import (
    build_mamba_delta_t,
    init_state,
    train_epoch_dmf,
    train_epoch_mamba,
    train_epoch_rnn,
    train_epoch_unequal_lstm,
    validate,
)
from data_set_module import dataset
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Server(object):

    def __init__(self, conf):
        self.conf = conf
        self.global_model = self.initialize_global_model()
        self.cloud_history = self.initialize_cloud_history()

    def initialize_global_model(self):
        return models.get_model(self.conf).to(device)

    def initialize_cloud_history(self):
        cloud_history_source = self.conf.get("cloud_history_source", "ground_truth_proxy")
        return {
            "source": cloud_history_source,
            "proxy_data": self.conf.get("pd_data"),
            "per_client": {},
            "meta": {},
        }

    def is_server_pretrain_enabled(self):
        return bool(self.conf.get("server_pretrain_enable", False))

    def initialize_server_history_source(self):
        full_truth = torch.from_numpy(self.conf["pd_data"].values).to(device)
        num_locations, num_slices = full_truth.shape
        history_ratio = self.conf.get("server_pretrain_ratio", 0.5)
        split_mode = self.conf.get("server_pretrain_split_mode", "time_first_half")
        history_slice_count = self.compute_server_history_slice_count(num_slices, history_ratio, split_mode)

        return {
            "source_name": self.conf.get("server_history_source", "train_data_half_proxy"),
            "split_mode": split_mode,
            "history_ratio": history_ratio,
            "full_truth": full_truth,
            "num_locations": num_locations,
            "num_slices": num_slices,
            "history_slice_count": history_slice_count,
        }

    def compute_server_history_slice_count(self, num_slices, history_ratio, split_mode):
        if split_mode == "time_first_half":
            history_slice_count = int(num_slices * history_ratio)
        else:
            raise ValueError("Unsupported server_pretrain_split_mode")

        history_slice_count = min(history_slice_count, num_slices)
        history_slice_count = max(history_slice_count, 0)
        return history_slice_count

    def build_server_pretrain_mask(self, history_source):
        num_locations = history_source["num_locations"]
        num_slices = history_source["num_slices"]
        history_slice_count = history_source["history_slice_count"]
        full_truth = history_source["full_truth"]

        history_mask = torch.zeros((num_locations, num_slices), device=full_truth.device, dtype=full_truth.dtype)
        if history_slice_count > 0:
            history_mask[:, :history_slice_count] = 1
        return history_mask

    def build_server_pretrain_data(self, history_source):
        ground_truth = history_source["full_truth"]
        history_mask = self.build_server_pretrain_mask(history_source)
        sparse_X = ground_truth * history_mask

        if self.conf["data_form"] != "unequal_sliced":
            data_list = [sparse_X, history_mask, ground_truth]
        else:
            data_list = [sparse_X, history_mask, ground_truth, self.conf["timestamp"]]

        return {
            "ground_truth": ground_truth,
            "history_mask": history_mask,
            "sparse_X": sparse_X,
            "data_list": data_list,
        }

    def build_server_pretrain_dataset(self, server_pretrain_data):
        return dataset(self.conf, server_pretrain_data["data_list"], self.global_model)

    def create_server_pretrain_optimizer(self):
        optimizer_name = self.conf.get("server_pretrain_optimizer") or self.conf["optimizer"]
        learning_rate = self.conf.get("server_pretrain_lr")
        if learning_rate is None:
            learning_rate = self.conf["lr"]

        if optimizer_name == "Rprop":
            return torch.optim.Rprop(self.global_model.parameters(), lr=learning_rate)
        raise ValueError("Unsupported server_pretrain_optimizer")

    def create_server_pretrain_criterion(self):
        return torch.nn.MSELoss()

    def train_one_server_pretrain_epoch(self, server_dataset, optimizer, criterion, ground_truth):
        if self.conf["model_name"] == "DMF":
            train_loss, completion, _single_step_gradient = train_epoch_dmf(
                server_dataset,
                self.global_model,
                optimizer,
                criterion,
                capture_single_step_gradient=False,
            )
        elif self.conf["model_name"] == "RNN_DMF":
            train_loss, completion, _single_step_gradient = train_epoch_rnn(
                self.conf,
                server_dataset,
                self.global_model,
                optimizer,
                criterion,
                capture_single_step_gradient=False,
            )
        elif self.conf["model_name"] == "TIME_DMF":
            train_loss, completion, _single_step_gradient = train_epoch_unequal_lstm(
                self.conf,
                server_dataset,
                self.global_model,
                optimizer,
                criterion,
                capture_single_step_gradient=False,
            )
        elif self.conf["model_name"] == "Mamba_DMF":
            train_loss, completion, _single_step_gradient = train_epoch_mamba(
                server_dataset,
                self.global_model,
                optimizer,
                criterion,
                capture_single_step_gradient=False,
            )
        else:
            raise ValueError("Unknown model_name")

        validate_loss = validate(completion, ground_truth)
        return train_loss, validate_loss, completion

    def pretrain_global_model(self):
        history_source = self.initialize_server_history_source()
        history_slice_count = history_source["history_slice_count"]

        if not self.is_server_pretrain_enabled():
            return {
                "enabled": False,
                "executed": False,
                "history_slice_count": history_slice_count,
                "train_loss": None,
                "validate_loss": None,
                "time_cost": 0.0,
            }

        epochs = int(self.conf.get("server_pretrain_epochs", 0))
        if epochs <= 0 or history_slice_count == 0:
            return {
                "enabled": True,
                "executed": False,
                "history_slice_count": history_slice_count,
                "train_loss": None,
                "validate_loss": None,
                "time_cost": 0.0,
            }

        server_pretrain_data = self.build_server_pretrain_data(history_source)
        optimizer = self.create_server_pretrain_optimizer()
        criterion = self.create_server_pretrain_criterion()

        last_train_loss = None
        last_validate_loss = None
        pretrain_start_time = time.time()
        self.global_model.train()
        for _ in range(epochs):
            server_dataset = self.build_server_pretrain_dataset(server_pretrain_data)
            train_loss, validate_loss, _ = self.train_one_server_pretrain_epoch(
                server_dataset, optimizer, criterion, server_pretrain_data["ground_truth"]
            )
            last_train_loss = train_loss
            last_validate_loss = validate_loss
        pretrain_time_cost = time.time() - pretrain_start_time

        return {
            "enabled": True,
            "executed": True,
            "history_slice_count": history_slice_count,
            "train_loss": last_train_loss,
            "validate_loss": last_validate_loss,
            "time_cost": pretrain_time_cost,
        }

    def get_global_model(self):
        return self.global_model

    def get_global_state_dict(self):
        return self.global_model.state_dict()

    def get_aggregation_scale(self, num_participants):
        if num_participants == 0:
            raise ValueError("num_participants must be non-zero for model aggregation")
        return self.conf["lambda"] / num_participants

    def compute_layer_update(self, layer_name, weight_accumulator, num_participants):
        scale = self.get_aggregation_scale(num_participants)
        return weight_accumulator[layer_name] * scale

    def align_update_dtype(self, layer_data, update_per_layer):
        if layer_data.type() != update_per_layer.type():
            return update_per_layer.to(layer_data.dtype)
        return update_per_layer

    def apply_layer_update(self, layer_data, update_per_layer):
        aligned_update = self.align_update_dtype(layer_data, update_per_layer)
        layer_data.add_(aligned_update)

    def model_aggregate(self, weight_accumulator, num_participants):
        for name, data in self.get_global_state_dict().items():
            update_per_layer = self.compute_layer_update(name, weight_accumulator, num_participants)
            self.apply_layer_update(data, update_per_layer)

    def ensure_client_history(self, client_id):
        per_client_history = self.cloud_history["per_client"]
        if client_id not in per_client_history:
            per_client_history[client_id] = {
                "selected_indices": [],
                "score_snapshots": [],
                "notes": {},
            }
        return per_client_history[client_id]

    def build_global_score_context(self, client_id, missing_mask=None, client_context=None):
        return {
            "client_id": client_id,
            "missing_mask": missing_mask,
            "client_context": client_context,
            "client_history": self.ensure_client_history(client_id),
            "proxy_history_data": self.cloud_history["proxy_data"],
            "global_model": self.global_model,
        }

    def create_score_map_template(self, missing_mask):
        return torch.full(missing_mask.shape, float("-inf"), device=missing_mask.device, dtype=torch.float32)

    def compute_global_score_map_reserved(self, client_id, missing_mask, client_context=None):
        _ = self.build_global_score_context(client_id, missing_mask, client_context)
        raise NotImplementedError("Future cloud score-map computation is reserved but not wired into main flow yet.")

    def normalize_global_missing_mask(self, missing_mask):
        return missing_mask.to(device=device, dtype=torch.bool)

    def _candidate_limit(self, limit_key, default_limit):
        limit = self.conf.get(limit_key)
        if limit is None:
            limit = default_limit
        if limit is None:
            return 0
        return max(int(limit), 0)

    def _build_candidate_rank_map(self, score_map, candidate_limit):
        rank_map = torch.zeros_like(score_map, dtype=torch.int64)
        valid_mask = torch.isfinite(score_map)
        valid_count = valid_mask.sum().item()
        if valid_count == 0 or candidate_limit <= 0:
            return rank_map

        topk = min(candidate_limit, valid_count)
        _, flat_idx = torch.topk(score_map.view(-1), topk)
        rank_map.view(-1)[flat_idx] = torch.arange(1, topk + 1, device=score_map.device, dtype=torch.int64)
        return rank_map

    def extract_context_local_candidate_mask(self, context, missing_mask):
        client_context = context.get("client_context") or {}
        local_candidate_mask = client_context.get("local_candidate_mask")
        if local_candidate_mask is None:
            return torch.zeros_like(missing_mask, dtype=torch.bool)
        local_candidate_mask = local_candidate_mask.to(device=missing_mask.device, dtype=torch.bool)
        if local_candidate_mask.shape != missing_mask.shape:
            raise ValueError("local candidate mask shape must match missing_mask shape")
        return local_candidate_mask & missing_mask

    def build_global_score_data_list(self, missing_mask, client_context=None):
        if client_context is not None and client_context.get("data_list") is not None:
            return client_context["data_list"]

        ground_truth = torch.from_numpy(self.conf["pd_data"].values).to(device=device)
        observed_mask = (~missing_mask).to(device=device, dtype=ground_truth.dtype)
        sparse_X = ground_truth * observed_mask

        if self.conf["data_form"] != "unequal_sliced":
            return [sparse_X, observed_mask, ground_truth]
        return [sparse_X, observed_mask, ground_truth, self.conf["timestamp"]]

    def build_global_score_dataset(self, data_list):
        return dataset(self.conf, data_list, self.global_model)

    def predict_completion_with_global_model(self, global_score_dataset):
        was_training = self.global_model.training
        try:
            self.global_model.eval()
            with torch.no_grad():
                if self.conf["model_name"] == "DMF":
                    comp = self.global_model().T
                elif self.conf["model_name"] == "RNN_DMF":
                    outputs = []
                    h = init_state(self.conf["z_size"])
                    for (x_steps, z_steps, m_steps, _) in global_score_dataset:
                        y_steps, h = self.global_model(z_steps, h)
                        outputs.append(y_steps)
                    comp = torch.cat(outputs, dim=1)
                elif self.conf["model_name"] == "TIME_DMF":
                    outputs = []
                    h = init_state(self.conf["z_size"])
                    c = init_state(self.conf["c_size"])
                    last_t = -1
                    for (x_steps, z_steps, m_steps, t_steps) in global_score_dataset:
                        y_steps, h, c, last_t = self.global_model(z_steps, h, c, t_steps, last_t, self.conf["delta"])
                        outputs.append(y_steps)
                    comp = torch.cat(outputs, dim=1)
                elif self.conf["model_name"] == "Mamba_DMF":
                    delta_t = build_mamba_delta_t(global_score_dataset.timestamp, next(self.global_model.parameters()).device)
                    comp, _h, _c, _last_t = self.global_model(delta_t)
                else:
                    raise ValueError("Unknown model_name")

            return comp * (1 - global_score_dataset.M) + global_score_dataset.sparse_X
        finally:
            self.global_model.train(was_training)

    def get_cloud_proxy_history_tensor(self, target_device, target_dtype=torch.float32):
        proxy_data = self.cloud_history.get("proxy_data")
        if proxy_data is None:
            proxy_data = self.conf["pd_data"]
        if torch.is_tensor(proxy_data):
            return proxy_data.to(device=target_device, dtype=target_dtype)
        return torch.from_numpy(proxy_data.values).to(device=target_device, dtype=target_dtype)

    def normalize_cloud_metric_map(self, metric_map, valid_mask):
        normalized_map = torch.zeros_like(metric_map, dtype=torch.float32)
        valid_values = metric_map.to(torch.float32)[valid_mask]
        if valid_values.numel() == 0:
            return normalized_map
        min_value = valid_values.min()
        max_value = valid_values.max()
        span = max_value - min_value
        if span.abs().item() <= 1e-12:
            normalized_values = torch.zeros_like(valid_values)
        else:
            normalized_values = (valid_values - min_value) / span
        normalized_map[valid_mask] = normalized_values
        return normalized_map

    def normalize_cloud_metric_values(self, values):
        if values.numel() == 0:
            return values.to(torch.float32)
        values = values.to(torch.float32)
        min_value = values.min()
        max_value = values.max()
        span = max_value - min_value
        if span.abs().item() <= 1e-12:
            return torch.zeros_like(values)
        return (values - min_value) / span

    def build_cloud_candidate_indices(self, missing_mask):
        rows, cols = torch.where(missing_mask)
        flat_indices = rows * missing_mask.size(1) + cols
        return rows, cols, flat_indices

    def compute_cloud_utility_raw_map(self, completion, missing_mask, strategy):
        if strategy in {"history_proxy_placeholder", "cloud_joint_cost"}:
            proxy_history = self.get_cloud_proxy_history_tensor(completion.device, torch.float32)
            raw_utility_map = (completion.to(torch.float32) - proxy_history).abs()
        elif strategy == "global_prediction_abs":
            raw_utility_map = completion.to(torch.float32).abs()
        else:
            raise ValueError("Unknown global_strategy")
        return raw_utility_map.masked_fill(~missing_mask, 0.0)

    def compute_neighborhood_mean_map(self, source_map, row_radius=0, col_radius=0, exclude_center=True):
        if row_radius <= 0 and col_radius <= 0:
            return source_map.to(torch.float32)
        kernel_height = 2 * row_radius + 1
        kernel_width = 2 * col_radius + 1
        source_4d = source_map.to(torch.float32).unsqueeze(0).unsqueeze(0)
        valid_4d = torch.ones_like(source_4d)
        kernel = torch.ones((1, 1, kernel_height, kernel_width), device=source_map.device, dtype=torch.float32)
        padding = (row_radius, col_radius)
        neighbor_sum = F.conv2d(source_4d, kernel, padding=padding).squeeze(0).squeeze(0)
        neighbor_count = F.conv2d(valid_4d, kernel, padding=padding).squeeze(0).squeeze(0)
        if exclude_center:
            neighbor_sum = neighbor_sum - source_map.to(torch.float32)
            neighbor_count = neighbor_count - 1.0
        neighbor_count = neighbor_count.clamp_min(1.0)
        return neighbor_sum / neighbor_count

    def compute_observed_density_map(self, missing_mask):
        radius = int(self.conf.get("cloud_density_radius", 1))
        observed_map = (~missing_mask).to(dtype=torch.float32)
        if radius <= 0:
            return observed_map
        kernel_size = 2 * radius + 1
        observed_4d = observed_map.unsqueeze(0).unsqueeze(0)
        valid_4d = torch.ones_like(observed_4d)
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=missing_mask.device, dtype=torch.float32)
        observed_sum = F.conv2d(observed_4d, kernel, padding=radius).squeeze(0).squeeze(0)
        window_count = F.conv2d(valid_4d, kernel, padding=radius).squeeze(0).squeeze(0).clamp_min(1.0)
        return observed_sum / window_count

    def compute_cloud_anomaly_risk_map(self, completion, missing_mask, metric_mask=None):
        valid_mask = missing_mask if metric_mask is None else metric_mask
        proxy_history = self.get_cloud_proxy_history_tensor(completion.device, torch.float32)
        row_std = proxy_history.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        hist_raw = (completion.to(torch.float32) - proxy_history).abs() / row_std
        hist_norm = self.normalize_cloud_metric_map(hist_raw, valid_mask)

        spatial_radius = int(self.conf.get("cloud_spatial_neighbor_radius", 1))
        spatial_mean = self.compute_neighborhood_mean_map(completion, row_radius=spatial_radius, col_radius=0)
        spatial_raw = (completion.to(torch.float32) - spatial_mean).abs()
        spatial_norm = self.normalize_cloud_metric_map(spatial_raw, valid_mask)

        temporal_radius = int(self.conf.get("cloud_temporal_neighbor_radius", 1))
        temporal_mean = self.compute_neighborhood_mean_map(completion, row_radius=0, col_radius=temporal_radius)
        temporal_raw = (completion.to(torch.float32) - temporal_mean).abs()
        temporal_norm = self.normalize_cloud_metric_map(temporal_raw, valid_mask)

        history_weight = float(self.conf.get("cloud_anomaly_history_weight", 1.0))
        spatial_weight = float(self.conf.get("cloud_anomaly_spatial_weight", 1.0))
        temporal_weight = float(self.conf.get("cloud_anomaly_temporal_weight", 1.0))
        weighted_anomaly = (
            history_weight * hist_norm +
            spatial_weight * spatial_norm +
            temporal_weight * temporal_norm
        )
        anomaly_map = self.normalize_cloud_metric_map(weighted_anomaly, valid_mask)

        return anomaly_map, {
            "history_zscore": hist_norm,
            "spatial_deviation": spatial_norm,
            "temporal_deviation": temporal_norm,
        }

    def build_manual_cloud_candidate_features(self, rows, cols, completion, missing_mask, utility_map, anomaly_map):
        num_rows, num_cols = completion.shape
        row_denominator = max(num_rows - 1, 1)
        col_denominator = max(num_cols - 1, 1)
        loc_feature = rows.to(torch.float32) / row_denominator
        time_feature = cols.to(torch.float32) / col_denominator
        prediction_norm_map = self.normalize_cloud_metric_map(completion.to(torch.float32), missing_mask)
        prediction_feature = prediction_norm_map[rows, cols]
        utility_feature = utility_map[rows, cols]
        error_feature = anomaly_map[rows, cols]
        density_map = self.compute_observed_density_map(missing_mask)
        density_feature = density_map[rows, cols]
        return torch.stack(
            [loc_feature, time_feature, prediction_feature, utility_feature, error_feature, density_feature],
            dim=1,
        ).to(torch.float32)

    def project_dmf_hidden_before_output(self, hidden_input):
        hidden = hidden_input.to(torch.float32)
        for module in list(self.global_model.dmf.children())[:-1]:
            hidden = module(hidden)
        return hidden.to(torch.float32)

    def get_global_output_row_embedding(self, rows):
        output_layer = list(self.global_model.dmf.children())[-1]
        if not hasattr(output_layer, "weight"):
            raise ValueError("global model dmf output layer must expose weight for model_embedding feature mode")
        return output_layer.weight[rows].to(torch.float32)

    def compute_global_time_hidden_embeddings(self, global_score_dataset):
        was_training = self.global_model.training
        try:
            self.global_model.eval()
            with torch.no_grad():
                if self.conf["model_name"] == "DMF":
                    return self.project_dmf_hidden_before_output(self.global_model.z.T)

                if self.conf["model_name"] == "Mamba_DMF":
                    delta_t = build_mamba_delta_t(global_score_dataset.timestamp, next(self.global_model.parameters()).device)
                    mamba_output = self.global_model.mamba(self.global_model.z.T.unsqueeze(0), delta_t).squeeze(0)
                    return self.project_dmf_hidden_before_output(mamba_output)

                if self.conf["model_name"] == "RNN_DMF":
                    h = init_state(self.conf["z_size"])
                    hidden_outputs = []
                    for (x_steps, z_steps, m_steps, _) in global_score_dataset:
                        for z in z_steps.T:
                            h = self.global_model.delta * h + (1 - self.global_model.delta) * self.global_model.rnn_ih(z)
                            hidden_outputs.append(self.project_dmf_hidden_before_output(h))
                    return torch.cat(hidden_outputs, dim=0)

                if self.conf["model_name"] == "TIME_DMF":
                    h = init_state(self.conf["z_size"])
                    c_long = init_state(self.conf["c_size"])
                    last_t = -1
                    hidden_outputs = []
                    for (x_steps, z_steps, m_steps, t_steps) in global_score_dataset:
                        for (z, t) in zip(z_steps.mT, t_steps.T):
                            delta_t = (t - last_t).reshape(1, 1).to(device=z.device, dtype=torch.float32).detach()
                            last_t = t.detach()
                            t1 = self.global_model.T1(delta_t)
                            t2 = self.global_model.T2(delta_t)
                            x = self.conf["delta"] * h + (1 - self.conf["delta"]) * self.global_model.rnn_ih(z)
                            c_long = c_long * t1 + x * (1 - t1)
                            c_short = c_long * t2 + x * (1 - t2)
                            h = c_short
                            hidden_outputs.append(self.project_dmf_hidden_before_output(c_short))
                    return torch.cat(hidden_outputs, dim=0)

                raise ValueError("Unknown model_name")
        finally:
            self.global_model.train(was_training)

    def build_model_embedding_cloud_candidate_features(
            self, rows, cols, completion, missing_mask, utility_map, anomaly_map, global_score_dataset):
        time_embeddings = self.compute_global_time_hidden_embeddings(global_score_dataset)
        if int(cols.max().item()) >= time_embeddings.size(0):
            raise ValueError("model embedding time dimension is smaller than candidate column index")
        time_feature = time_embeddings[cols].to(torch.float32)
        row_feature = self.get_global_output_row_embedding(rows)

        num_rows, num_cols = completion.shape
        row_denominator = max(num_rows - 1, 1)
        col_denominator = max(num_cols - 1, 1)
        loc_feature = (rows.to(torch.float32) / row_denominator).unsqueeze(1)
        time_position_feature = (cols.to(torch.float32) / col_denominator).unsqueeze(1)
        prediction_norm_map = self.normalize_cloud_metric_map(completion.to(torch.float32), missing_mask)
        prediction_feature = prediction_norm_map[rows, cols].unsqueeze(1)
        utility_feature = utility_map[rows, cols].unsqueeze(1)
        anomaly_feature = anomaly_map[rows, cols].unsqueeze(1)
        return torch.cat(
            [time_feature, row_feature, loc_feature, time_position_feature, prediction_feature, utility_feature, anomaly_feature],
            dim=1,
        ).to(torch.float32)

    def build_cloud_candidate_features(
            self, rows, cols, completion, missing_mask, utility_map, anomaly_map, global_score_dataset):
        feature_mode = self.conf.get("cloud_feature_mode", "manual")
        if feature_mode == "manual":
            return self.build_manual_cloud_candidate_features(rows, cols, completion, missing_mask, utility_map, anomaly_map)
        if feature_mode == "model_embedding":
            return self.build_model_embedding_cloud_candidate_features(
                rows, cols, completion, missing_mask, utility_map, anomaly_map, global_score_dataset
            )
        raise ValueError("cloud_feature_mode must be 'manual' or 'model_embedding'")

    def limit_cloud_joint_candidates(self, rows, cols, flat_indices, utility_raw_map):
        candidate_size = self.conf.get("cloud_joint_candidate_size")
        if candidate_size is None or int(candidate_size) <= 0:
            return rows, cols, flat_indices
        candidate_size = min(int(candidate_size), rows.numel())
        if candidate_size == rows.numel():
            return rows, cols, flat_indices
        candidate_values = utility_raw_map[rows, cols].to(torch.float32)
        _, selected_positions = torch.topk(candidate_values, candidate_size)
        return rows[selected_positions], cols[selected_positions], flat_indices[selected_positions]

    def compute_cloud_similarity_chunk(self, query_features, all_features):
        similarity = self.conf.get("cloud_similarity", "rbf")
        if similarity == "rbf":
            tau = float(self.conf.get("cloud_rbf_tau", 1.0))
            squared_distance = torch.cdist(query_features, all_features, p=2).pow(2)
            return torch.exp(-squared_distance / (tau ** 2))
        if similarity == "clipped_cosine":
            query_norm = F.normalize(query_features, dim=1, eps=1e-12)
            all_norm = F.normalize(all_features, dim=1, eps=1e-12)
            return torch.matmul(query_norm, all_norm.T).clamp_min(0.0)
        raise ValueError("Unknown cloud_similarity")

    def compute_cloud_representativeness_values(self, features):
        if features.numel() == 0:
            return torch.empty(0, device=features.device, dtype=torch.float32)
        chunk_size = int(self.conf.get("cloud_similarity_chunk_size", 512))
        chunk_size = max(chunk_size, 1)
        representativeness_chunks = []
        for start in range(0, features.size(0), chunk_size):
            end = min(start + chunk_size, features.size(0))
            similarity_chunk = self.compute_cloud_similarity_chunk(features[start:end], features)
            representativeness_chunks.append(similarity_chunk.mean(dim=1))
        representativeness = torch.cat(representativeness_chunks, dim=0)
        return self.normalize_cloud_metric_values(representativeness)

    def scatter_candidate_values_to_map(self, rows, cols, values, shape, fill_value=0.0):
        output_map = torch.full(shape, fill_value, device=values.device, dtype=torch.float32)
        if values.numel() > 0:
            output_map[rows, cols] = values.to(torch.float32)
        return output_map

    def build_cloud_joint_cost_score_package(self, completion, missing_mask, context, global_score_dataset):
        rows, cols, flat_indices = self.build_cloud_candidate_indices(missing_mask)
        score_map = self.create_score_map_template(missing_mask)
        candidate_limit = self._candidate_limit("global_candidate_size", self.conf.get("final_query_budget", 0))
        if rows.numel() == 0:
            candidate_rank_map = self._build_candidate_rank_map(score_map, candidate_limit)
            return {
                "score_map": score_map,
                "candidate_rank_map": candidate_rank_map,
                "candidate_mask": candidate_rank_map > 0,
                "joint": {},
                "meta": {
                    "client_id": context["client_id"],
                    "strategy": "cloud_joint_cost",
                    "candidate_limit": candidate_limit,
                    "missing_count": 0,
                    "candidate_count": 0,
                    "joint_candidate_count": 0,
                    "metric_candidate_count": 0,
                    "cloud_joint_candidate_size": self.conf.get("cloud_joint_candidate_size"),
                    "cloud_joint_candidate_source": self.conf.get("cloud_joint_candidate_source", "rank_union"),
                    "history_selected_rounds": len(context["client_history"]["selected_indices"]),
                    "score_rule": "empty_missing_domain",
                },
            }

        utility_raw_map = self.compute_cloud_utility_raw_map(completion, missing_mask, "cloud_joint_cost")
        candidate_source = self.conf.get("cloud_joint_candidate_source", "rank_union")
        global_rank_source_map = utility_raw_map.masked_fill(~missing_mask, float("-inf"))
        global_candidate_rank_map = self._build_candidate_rank_map(global_rank_source_map, candidate_limit)
        use_rank_union_domain = (
            self.conf.get("fusion_policy") == "cloud_joint_greedy" and candidate_source == "rank_union"
        )
        if use_rank_union_domain:
            local_candidate_mask = self.extract_context_local_candidate_mask(context, missing_mask)
            final_candidate_mask = missing_mask & (local_candidate_mask | (global_candidate_rank_map > 0))
            cloud_rows, cloud_cols, _ = self.limit_cloud_joint_candidates(
                rows, cols, flat_indices, utility_raw_map
            )
            metric_candidate_mask = torch.zeros_like(missing_mask, dtype=torch.bool)
            metric_candidate_mask[cloud_rows, cloud_cols] = True
            metric_candidate_mask = missing_mask & (metric_candidate_mask | final_candidate_mask)
        else:
            rows, cols, flat_indices = self.limit_cloud_joint_candidates(
                rows, cols, flat_indices, utility_raw_map
            )
            final_candidate_mask = torch.zeros_like(missing_mask, dtype=torch.bool)
            final_candidate_mask[rows, cols] = True
            metric_candidate_mask = final_candidate_mask
        metric_rows, metric_cols, metric_flat_indices = self.build_cloud_candidate_indices(metric_candidate_mask)
        final_rows, final_cols, final_flat_indices = self.build_cloud_candidate_indices(final_candidate_mask)
        utility_map = self.normalize_cloud_metric_map(utility_raw_map, metric_candidate_mask)
        anomaly_map, anomaly_parts = self.compute_cloud_anomaly_risk_map(
            completion, missing_mask, metric_mask=metric_candidate_mask
        )
        feature_mode = self.conf.get("cloud_feature_mode", "manual")
        candidate_features = self.build_cloud_candidate_features(
            metric_rows, metric_cols, completion, metric_candidate_mask, utility_map, anomaly_map, global_score_dataset
        )
        representativeness_values = self.compute_cloud_representativeness_values(candidate_features)
        representativeness_map = self.scatter_candidate_values_to_map(
            metric_rows, metric_cols, representativeness_values, completion.shape, fill_value=0.0
        )

        utility_weight = float(self.conf.get("cloud_joint_utility_weight", 1.0))
        represent_weight = float(self.conf.get("cloud_joint_represent_weight", 0.5))
        anomaly_weight = float(self.conf.get("cloud_joint_anomaly_weight", 0.3))
        base_cost_map = (
            -utility_weight * utility_map -
            represent_weight * representativeness_map +
            anomaly_weight * anomaly_map
        )
        score_map = (-base_cost_map).masked_fill(~final_candidate_mask, float("-inf"))
        if use_rank_union_domain:
            candidate_rank_map = global_candidate_rank_map
        else:
            candidate_rank_map = self._build_candidate_rank_map(score_map, candidate_limit)
        feature_map = torch.zeros((*completion.shape, candidate_features.size(1)), device=completion.device, dtype=torch.float32)
        feature_map[metric_rows, metric_cols] = candidate_features

        return {
            "score_map": score_map,
            "candidate_rank_map": candidate_rank_map,
            "candidate_mask": candidate_rank_map > 0,
            "joint": {
                "rows": final_rows,
                "cols": final_cols,
                "flat_indices": final_flat_indices,
                "feature_map": feature_map,
                "utility_score_map": utility_map,
                "representativeness_score_map": representativeness_map,
                "anomaly_score_map": anomaly_map,
                "base_cost_map": base_cost_map.masked_fill(~final_candidate_mask, float("inf")),
                "candidate_mask": final_candidate_mask,
                "metric_candidate_mask": metric_candidate_mask,
                "anomaly_parts": anomaly_parts,
            },
            "meta": {
                "client_id": context["client_id"],
                "strategy": "cloud_joint_cost",
                "candidate_limit": candidate_limit,
                "missing_count": int(missing_mask.sum().item()),
                "candidate_count": int((candidate_rank_map > 0).sum().item()),
                "joint_candidate_count": int(final_rows.numel()),
                "metric_candidate_count": int(metric_rows.numel()),
                "cloud_joint_candidate_size": self.conf.get("cloud_joint_candidate_size"),
                "cloud_joint_candidate_source": self.conf.get("cloud_joint_candidate_source", "rank_union"),
                "history_selected_rounds": len(context["client_history"]["selected_indices"]),
                "score_rule": "joint_cost_static_terms_without_dynamic_redundancy",
                "feature_mode": feature_mode,
                "similarity": self.conf.get("cloud_similarity", "rbf"),
                "utility_weight": utility_weight,
                "represent_weight": represent_weight,
                "anomaly_weight": anomaly_weight,
                "redundancy_weight": float(self.conf.get("cloud_joint_redundancy_weight", 0.5)),
            },
        }

    def build_global_score_map_from_prediction(self, completion, missing_mask, strategy):
        if completion.shape != missing_mask.shape:
            raise ValueError("global completion shape must match missing_mask shape")

        if strategy == "history_proxy_placeholder":
            proxy_history = self.get_cloud_proxy_history_tensor(completion.device, torch.float32)
            raw_score_map = (completion.to(torch.float32) - proxy_history).abs()
        elif strategy == "global_prediction_abs":
            raw_score_map = completion.to(torch.float32).abs()
        else:
            raise ValueError("Unknown global_strategy")

        return raw_score_map.masked_fill(~missing_mask, float("-inf"))

    def compute_global_score_map(self, client_id, missing_mask, client_context=None):
        missing_mask = self.normalize_global_missing_mask(missing_mask)
        context = self.build_global_score_context(client_id, missing_mask, client_context)
        data_list = self.build_global_score_data_list(missing_mask, client_context)
        global_score_dataset = self.build_global_score_dataset(data_list)
        completion = self.predict_completion_with_global_model(global_score_dataset)
        strategy = self.conf.get("global_strategy", "history_proxy_placeholder")
        if strategy == "cloud_joint_cost":
            return self.build_cloud_joint_cost_score_package(
                completion, missing_mask, context, global_score_dataset
            )
        score_map = self.build_global_score_map_from_prediction(completion, missing_mask, strategy)
        candidate_limit = self._candidate_limit("global_candidate_size", self.conf.get("final_query_budget", 0))
        candidate_rank_map = self._build_candidate_rank_map(score_map, candidate_limit)

        return {
            "score_map": score_map,
            "candidate_rank_map": candidate_rank_map,
            "candidate_mask": candidate_rank_map > 0,
            "meta": {
                "client_id": client_id,
                "strategy": strategy,
                "candidate_limit": candidate_limit,
                "missing_count": int(missing_mask.sum().item()),
                "candidate_count": int((candidate_rank_map > 0).sum().item()),
                "history_selected_rounds": len(context["client_history"]["selected_indices"]),
            },
        }

    def update_cloud_history(self, client_id, selected_indices=None, score_snapshot=None):
        client_history = self.ensure_client_history(client_id)
        if selected_indices is not None:
            client_history["selected_indices"].append(selected_indices)
        if score_snapshot is not None:
            client_history["score_snapshots"].append(score_snapshot)
        return client_history
