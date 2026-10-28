import os
import time

import numpy as np

from data_set_module import dataset
import torch
import models
from statistics import mean
from console_utils import NAN, emit_console, format_console_record

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_epoch_progress(epoch_idx, total_epochs):
    current_epoch = int(epoch_idx) + 1
    if total_epochs is None:
        return current_epoch
    return f"{current_epoch}/{int(total_epochs)}"


def prepare_filesystem_path(path_value):
    normalized_path = os.path.normpath(os.path.abspath(path_value))
    if os.name != "nt":
        return normalized_path
    if normalized_path.startswith("\\\\?\\"):
        return normalized_path
    if normalized_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + normalized_path.lstrip("\\")
    if len(normalized_path) >= 240:
        return "\\\\?\\" + normalized_path
    return normalized_path


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir:
        os.makedirs(prepare_filesystem_path(parent_dir), exist_ok=True)


def append_experiment_log(out_file_path, content):
    ensure_parent_dir(out_file_path)
    with open(prepare_filesystem_path(out_file_path), "a", encoding="utf-8") as f:
        f.write(content)


def append_experiment_console(out_file_path, algorithm, event, *, blank_after=0, **fields):
    append_experiment_log(out_file_path, format_console_record(algorithm, event, blank_after=blank_after, **fields))


def emit_and_append_experiment(out_file_path, algorithm, event, *, blank_after=0, **fields):
    emit_console(algorithm, event, blank_after=blank_after, **fields)
    append_experiment_console(out_file_path, algorithm, event, blank_after=blank_after, **fields)


class Client(object):

    def __init__(self, conf, client_id):
        self.conf = conf
        self.local_model = models.get_model(self.conf).to(device)
        self.client_id = client_id
        self.ground_truth = torch.from_numpy(conf["pd_data"].values).to(device)
        miss_ratio = 1 - self.conf["sparse_ratio"]
        fixed_mask_path = self.conf.get("fixed_observed_mask_path")
        if fixed_mask_path:
            self.M = self.load_fixed_observed_mask(fixed_mask_path, client_id)
        else:
            self.M = self.get_mask(self.conf["num_locations"], self.conf["num_slices"], miss_ratio)
        self.sparse_X = self.M * self.ground_truth
        self.initial_missing_mask = self.M == 0

    def get_data(self):
        if self.conf["data_form"] != "unequal_sliced":
            return [self.sparse_X, self.M, self.ground_truth]
        else:
            return [self.sparse_X, self.M, self.ground_truth, self.conf["timestamp"]]

    def get_mask(self, n, m, miss_ratio):
        x = torch.ones((n, m)).to(device)
        x[0: int(miss_ratio * n), :] = 0
        for i in range(m):
            x[:, i] = x[torch.randperm(n), i]
        return x

    def resolve_fixed_mask_path(self, mask_path):
        if os.path.isabs(mask_path):
            return mask_path
        module_relative_path = os.path.join(os.path.dirname(__file__), mask_path)
        if os.path.exists(module_relative_path):
            return module_relative_path
        return os.path.abspath(mask_path)

    def load_fixed_observed_mask(self, mask_path, client_id):
        resolved_path = self.resolve_fixed_mask_path(mask_path)
        mask_array = np.load(resolved_path)
        if mask_array.ndim == 3:
            if client_id >= mask_array.shape[0]:
                raise ValueError("fixed_observed_mask_path does not contain mask for client {}".format(client_id))
            mask_array = mask_array[client_id]
        if mask_array.shape != (self.conf["num_locations"], self.conf["num_slices"]):
            raise ValueError(
                "fixed mask shape {} does not match data shape ({}, {})".format(
                    mask_array.shape, self.conf["num_locations"], self.conf["num_slices"]
                )
            )
        return torch.from_numpy(mask_array.astype("float32")).to(device)

    def sync_with_server(self, server_model):
        for name, param in server_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

    def build_dataset(self):
        self.data_set = dataset(self.conf, self.get_data(), self.local_model)
        return self.data_set

    def create_optimizer(self):
        if self.conf["optimizer"] == "Rprop":
            return torch.optim.Rprop(self.local_model.parameters(), lr=self.conf['lr'])
        raise ValueError("Unsupported optimizer")

    def train_one_local_epoch(self, optimizer, criterion, capture_single_step_gradient=False):
        if self.conf["model_name"] == 'DMF':
            train_loss, completion, single_step_gradient = train_epoch_dmf(
                self.data_set,
                self.local_model,
                optimizer,
                criterion,
                capture_single_step_gradient=capture_single_step_gradient,
            )
        elif self.conf["model_name"] == 'RNN_DMF':
            train_loss, completion, single_step_gradient = train_epoch_rnn(
                self.conf,
                self.data_set,
                self.local_model,
                optimizer,
                criterion,
                capture_single_step_gradient=capture_single_step_gradient,
            )
        elif self.conf["model_name"] == 'TIME_DMF':
            train_loss, completion, single_step_gradient = train_epoch_unequal_lstm(
                self.conf,
                self.data_set,
                self.local_model,
                optimizer,
                criterion,
                capture_single_step_gradient=capture_single_step_gradient,
            )
        elif self.conf["model_name"] == 'Mamba_DMF':
            train_loss, completion, single_step_gradient = train_epoch_mamba(
                self.data_set,
                self.local_model,
                optimizer,
                criterion,
                capture_single_step_gradient=capture_single_step_gradient,
            )
        else:
            raise ValueError("Unknown model_name")

        self.build_dataset()
        validate_loss = validate(completion, self.ground_truth)
        return train_loss, validate_loss, completion, single_step_gradient

    def should_run_active_learning(self):
        active_learning_timing = self.conf.get("al_timing") or self.conf.get("al_when", "per_local_epoch")
        return (self.conf.get("al_enable", False) and
                not self.conf.get("cloud_al_enable", False) and
                active_learning_timing == "per_local_epoch")

    def get_query_budget(self):
        budget = self.conf.get("al_budget", 0)
        if budget is None:
            return 0
        return int(budget)

    def get_missing_mask(self):
        return self.M == 0

    def has_active_dropout(self):
        dropout_types = (
            torch.nn.Dropout,
            torch.nn.Dropout1d,
            torch.nn.Dropout2d,
            torch.nn.Dropout3d,
            torch.nn.AlphaDropout,
            torch.nn.FeatureAlphaDropout,
        )
        for module in self.local_model.modules():
            if isinstance(module, dropout_types) and getattr(module, "p", 0) > 0:
                return True
        return False

    def _candidate_limit(self, limit_key, default_limit):
        limit = self.conf.get(limit_key)
        if limit is None:
            return default_limit
        return int(limit)

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

    def compute_local_score_map(self, completion=None):
        with torch.no_grad():
            miss_mask = self.get_missing_mask()
            score_map = torch.full(self.M.shape, float('-inf'), device=self.M.device, dtype=torch.float32)
            if miss_mask.sum().item() == 0:
                return {
                    "score_map": score_map,
                    "candidate_rank_map": torch.zeros(self.M.shape, device=self.M.device, dtype=torch.int64),
                    "candidate_mask": miss_mask,
                    "meta": {
                        "strategy": self.conf.get("al_strategy", "mc_dropout_var"),
                        "candidate_limit": 0,
                    }
                }

            strategy = self.conf.get("al_strategy", "mc_dropout_var")
            if strategy == "mc_dropout_var":
                if not self.has_active_dropout():
                    raise ValueError(
                        "al_strategy='mc_dropout_var' requires at least one active Dropout layer "
                        "with dropout_rate > 0"
                    )
                mc_passes = self.conf.get("al_mc_passes", 8)
                preds = []
                for _ in range(mc_passes):
                    comp_t = self.predict_completion_only(stochastic=True)
                    preds.append(comp_t.unsqueeze(0).to(torch.float32))
                preds = torch.cat(preds, dim=0)
                score_map = preds.var(dim=0, unbiased=False)
            elif strategy == "residual":
                if completion is None:
                    completion = self.predict_completion_only()
                score_map = (completion - self.ground_truth).abs().to(torch.float32)
            elif strategy == "random":
                score_map = torch.rand(self.M.shape, device=self.M.device, dtype=torch.float32)
            else:
                raise ValueError("Unknown al_strategy")

            score_map = score_map.masked_fill(~miss_mask, float('-inf'))
            candidate_limit = self._candidate_limit("local_candidate_size", self.get_query_budget())
            candidate_rank_map = self._build_candidate_rank_map(score_map, candidate_limit)

            return {
                "score_map": score_map,
                "candidate_rank_map": candidate_rank_map,
                "candidate_mask": miss_mask,
                "meta": {
                    "strategy": strategy,
                    "candidate_limit": candidate_limit,
                }
            }

    def compute_local_score_map_for_fusion(self, completion=None):
        strategy = self.conf.get("local_strategy") or self.conf.get("al_strategy", "mc_dropout_var")
        had_old_strategy = "al_strategy" in self.conf
        old_strategy = self.conf.get("al_strategy")
        should_override_strategy = strategy != old_strategy

        if should_override_strategy:
            self.conf["al_strategy"] = strategy

        try:
            local_scores = self.compute_local_score_map(completion)
        finally:
            if should_override_strategy and had_old_strategy:
                self.conf["al_strategy"] = old_strategy
            elif should_override_strategy:
                self.conf.pop("al_strategy", None)

        local_scores["candidate_mask"] = local_scores["candidate_rank_map"] > 0
        local_scores["meta"]["strategy"] = strategy
        local_scores["meta"]["candidate_domain"] = "local_top_candidates"
        return local_scores

    def select_topk_indices(self, score_map, k):
        valid_count = torch.isfinite(score_map).sum().item()
        if valid_count == 0 or k <= 0:
            empty = torch.empty(0, device=score_map.device, dtype=torch.long)
            return empty, empty

        topk = min(k, valid_count)
        _, flat_idx = torch.topk(score_map.view(-1), topk)
        rows = torch.div(flat_idx, score_map.size(1), rounding_mode="floor")
        cols = flat_idx % score_map.size(1)
        return rows, cols

    def _coerce_selected_indices(self, selected_indices):
        if isinstance(selected_indices, tuple):
            rows, cols = selected_indices
            rows = torch.as_tensor(rows, device=device, dtype=torch.long).reshape(-1)
            cols = torch.as_tensor(cols, device=device, dtype=torch.long).reshape(-1)
            return rows, cols

        if torch.is_tensor(selected_indices):
            if selected_indices.numel() == 0:
                empty = torch.empty(0, device=device, dtype=torch.long)
                return empty, empty
            if selected_indices.dim() != 2 or selected_indices.size(1) != 2:
                raise ValueError("selected_indices tensor must have shape [k, 2]")
            rows = selected_indices[:, 0].to(device=device, dtype=torch.long)
            cols = selected_indices[:, 1].to(device=device, dtype=torch.long)
            return rows, cols

        if not selected_indices:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return empty, empty

        rows = torch.tensor([idx[0] for idx in selected_indices], device=device, dtype=torch.long)
        cols = torch.tensor([idx[1] for idx in selected_indices], device=device, dtype=torch.long)
        return rows, cols

    def apply_selected_queries(self, selected_indices):
        rows, cols = self._coerce_selected_indices(selected_indices)
        if rows.numel() != cols.numel():
            raise ValueError("selected query row and column counts must match")
        if rows.numel() == 0:
            return 0

        row_count, col_count = self.M.shape
        rows_in_range = (rows >= 0) & (rows < row_count)
        cols_in_range = (cols >= 0) & (cols < col_count)
        if not torch.all(rows_in_range & cols_in_range).item():
            raise IndexError("selected query index is outside the client matrix")

        flat_indices = rows * col_count + cols
        unique_flat_indices = torch.unique(flat_indices)
        unique_rows = torch.div(unique_flat_indices, col_count, rounding_mode="floor")
        unique_cols = unique_flat_indices % col_count
        still_missing = self.M[unique_rows, unique_cols] == 0
        write_rows = unique_rows[still_missing]
        write_cols = unique_cols[still_missing]
        if write_rows.numel() == 0:
            return 0

        self.M[write_rows, write_cols] = 1
        self.sparse_X[write_rows, write_cols] = self.ground_truth[write_rows, write_cols].to(self.sparse_X.dtype)
        return int(write_rows.numel())

    def build_raw_model_diff(self, server_model):
        raw_diff = dict()
        server_state = server_model.state_dict()
        for name, data in self.local_model.state_dict().items():
            server_data = server_state[name].to(data.device)
            raw_diff[name] = data.detach() - server_data.detach()
        return raw_diff

    def compute_diff_l2_norm(self, diff):
        squared_norm_sum = 0.0
        for update in diff.values():
            squared_norm_sum += torch.sum(update.detach().float().pow(2)).item()
        return squared_norm_sum ** 0.5

    def clip_model_diff_by_l2(self, diff, clip_norm):
        raw_l2_norm = self.compute_diff_l2_norm(diff)
        clip_scale = min(1.0, float(clip_norm) / (raw_l2_norm + 1e-12))
        clipped_diff = dict()
        for name, update in diff.items():
            clipped_diff[name] = update * clip_scale
        return clipped_diff, raw_l2_norm, clip_scale

    def add_laplace_noise_to_diff(self, diff, noise_scale):
        noisy_diff = dict()
        for name, update in diff.items():
            noise_array = np.random.laplace(0, noise_scale, tuple(update.shape))
            noise = torch.as_tensor(noise_array, dtype=update.dtype, device=update.device)
            noisy_diff[name] = update + noise
        return noisy_diff

    def compute_model_diff(self, server_model):
        raw_diff = self.build_raw_model_diff(server_model)
        if self.conf["privacy"] == "no":
            return raw_diff
        if self.conf["privacy"] == "clip":
            clip_norm = float(self.conf["dp_clip_norm"])
            clipped_diff, _raw_l2_norm, _clip_scale = self.clip_model_diff_by_l2(raw_diff, clip_norm)
            return clipped_diff
        if self.conf["privacy"] == "laplace":
            clip_norm = float(self.conf["dp_clip_norm"])
            epsilon = float(self.conf["epsilon"])
            sensitivity_multiplier = float(self.conf["dp_sensitivity_multiplier"])
            clipped_diff, _raw_l2_norm, _clip_scale = self.clip_model_diff_by_l2(raw_diff, clip_norm)
            sensitivity = clip_norm * sensitivity_multiplier
            noise_scale = sensitivity / epsilon
            return self.add_laplace_noise_to_diff(clipped_diff, noise_scale)
        raise ValueError("Unsupported privacy mode")

    def train_local_epochs(self, server_model, e, run_active_learning=False, capture_single_step_gradient=False):
        start_time = time.time()

        self.sync_with_server(server_model)
        self.build_dataset()
        optimizer = self.create_optimizer()
        criterion = torch.nn.MSELoss()

        self.local_model.train()
        last_train_loss = 0
        last_validate_loss = 0
        last_initial_missing_validate_loss = 0
        last_missing_mae = 0
        last_observed_rmse = 0
        last_completion = None
        captured_single_step_gradient = None

        for i in range(self.conf["local_epochs"]):
            should_capture_epoch_gradient = capture_single_step_gradient and captured_single_step_gradient is None
            train_loss, validate_loss, completion, single_step_gradient = self.train_one_local_epoch(
                optimizer,
                criterion,
                capture_single_step_gradient=should_capture_epoch_gradient,
            )
            if single_step_gradient is not None and captured_single_step_gradient is None:
                captured_single_step_gradient = single_step_gradient
            last_train_loss = train_loss
            last_validate_loss = validate_loss
            last_initial_missing_validate_loss = validate_on_mask(
                completion, self.ground_truth, self.initial_missing_mask
            )
            last_missing_mae = mae_on_mask(
                completion, self.ground_truth, self.initial_missing_mask
            )
            last_observed_rmse = validate_on_mask(
                completion, self.ground_truth, ~self.initial_missing_mask
            )
            last_completion = completion

            if run_active_learning and self.should_run_active_learning():
                queried = self.active_query(completion)
                if queried > 0:
                    self.build_dataset()

            if (i + 1) % 5 == 0:
                emit_and_append_experiment(
                    self.conf["out_file_path"],
                    "Ours",
                    "client_progress",
                    client_id=self.client_id + 1,
                    clients=self.conf["num_clients"],
                    local_epoch=f"{i + 1}/{self.conf['local_epochs']}",
                    train_loss=train_loss,
                    validate_loss=validate_loss,
                    missing_rmse=last_initial_missing_validate_loss,
                    missing_mae=last_missing_mae,
                    observed_rmse=last_observed_rmse,
                    time_cost=NAN,
                )

        diff = self.compute_model_diff(server_model)
        end_time = time.time()
        time_cost = end_time - start_time

        emit_and_append_experiment(
            self.conf["out_file_path"],
            "Ours",
            "client_time",
            blank_after=1,
            client_id=self.client_id + 1,
            round=format_epoch_progress(e, self.conf.get("global_epochs")),
            time_cost=time_cost,
            train_loss=NAN,
            validate_loss=NAN,
        )

        return {
            "diff": diff,
            "train_loss": last_train_loss,
            "validate_loss": last_validate_loss,
            "initial_missing_validate_loss": last_initial_missing_validate_loss,
            "missing_rmse": last_initial_missing_validate_loss,
            "missing_mae": last_missing_mae,
            "observed_rmse": last_observed_rmse,
            "time_cost": time_cost,
            "completion": last_completion,
            "single_step_gradient": captured_single_step_gradient,
        }

    def local_train(self, server_model, e):
        start_time = time.time()

        self.sync_with_server(server_model)
        self.build_dataset()
        optimizer = self.create_optimizer()
        criterion = torch.nn.MSELoss()

        self.local_model.train()
        last_train_loss = 0
        last_validate_loss = 0
        for i in range(self.conf["local_epochs"]):
            train_loss, validate_loss, completion, _single_step_gradient = self.train_one_local_epoch(
                optimizer,
                criterion,
                capture_single_step_gradient=False,
            )
            missing_rmse = validate_on_mask(completion, self.ground_truth, self.initial_missing_mask)
            missing_mae = mae_on_mask(completion, self.ground_truth, self.initial_missing_mask)
            observed_rmse = validate_on_mask(completion, self.ground_truth, ~self.initial_missing_mask)

            if self.should_run_active_learning():
                queried = self.active_query(completion)
                if queried > 0:
                    self.build_dataset()

            if (i + 1) % 5 == 0:
                emit_and_append_experiment(
                    self.conf["out_file_path"],
                    "Ours",
                    "client_progress",
                    client_id=self.client_id + 1,
                    clients=self.conf["num_clients"],
                    local_epoch=f"{i + 1}/{self.conf['local_epochs']}",
                    train_loss=train_loss,
                    validate_loss=validate_loss,
                    missing_rmse=missing_rmse,
                    missing_mae=missing_mae,
                    observed_rmse=observed_rmse,
                    time_cost=NAN,
                )

            if (i + 1) == self.conf["local_epochs"]:
                last_train_loss = train_loss
                last_validate_loss = validate_loss

        diff = self.compute_model_diff(server_model)

        end_time = time.time()
        emit_and_append_experiment(
            self.conf["out_file_path"],
            "Ours",
            "client_time",
            blank_after=1,
            client_id=self.client_id + 1,
            round=format_epoch_progress(e, self.conf.get("global_epochs")),
            time_cost=end_time - start_time,
            train_loss=NAN,
            validate_loss=NAN,
        )
        return diff, last_train_loss, last_validate_loss, end_time - start_time


    def predict_completion_only(self, stochastic: bool = False):
        was_training = self.local_model.training
        try:
            if stochastic:
                self.local_model.train()
            else:
                self.local_model.eval()

            with torch.no_grad():
                if self.conf["model_name"] == 'DMF':
                    comp = self.local_model().T

                elif self.conf["model_name"] == 'RNN_DMF':
                    outputs = []
                    h = init_state(self.conf["z_size"])
                    for (x_steps, z_steps, m_steps, _) in self.data_set:
                        y_steps, h = self.local_model(z_steps, h)
                        outputs.append(y_steps)
                    comp = torch.cat(outputs, dim=1)

                elif self.conf["model_name"] == 'TIME_DMF':
                    outputs = []
                    h = init_state(self.conf["z_size"])
                    c = init_state(self.conf["c_size"])
                    last_t = -1
                    for (x_steps, z_steps, m_steps, t_steps) in self.data_set:
                        y_steps, h, c, last_t = self.local_model(z_steps, h, c, t_steps, last_t,
                                                                 self.conf["delta"])
                        outputs.append(y_steps)
                    comp = torch.cat(outputs, dim=1)

                elif self.conf["model_name"] == 'Mamba_DMF':
                    delta_t = build_mamba_delta_t(self.data_set.timestamp, next(self.local_model.parameters()).device)
                    comp, _h, _c, _last_t = self.local_model(delta_t)

                else:
                    raise ValueError("Unknown model_name")

            return comp * (1 - self.data_set.M) + self.data_set.sparse_X
        finally:
            self.local_model.train(was_training)

    def active_query(self, completion):
        budget = self.get_query_budget()

        if not self.conf.get("al_enable", False) or budget <= 0:
            return 0

        local_scores = self.compute_local_score_map(completion)
        rows, cols = self.select_topk_indices(local_scores["score_map"], budget)
        return self.apply_selected_queries((rows, cols))


def clone_model_gradients_for_attack(net):
    gradient_snapshot = {}
    for name, parameter in net.named_parameters():
        if parameter.grad is None:
            gradient_snapshot[name] = torch.zeros_like(parameter.detach()).cpu()
        else:
            gradient_snapshot[name] = parameter.grad.detach().cpu().clone()
    return gradient_snapshot


def attach_output_gradient_for_attack(gradient_snapshot, output_tensor, loss_numel=None):
    effective_loss_numel = float(output_tensor.numel() if loss_numel is None else loss_numel)
    gradient_snapshot["__attack_output_prediction__"] = output_tensor.detach().cpu().clone()
    gradient_snapshot["__attack_output_loss_numel__"] = torch.full_like(output_tensor.detach(), effective_loss_numel).cpu()
    if output_tensor.grad is None:
        gradient_snapshot["__attack_output_gradient__"] = torch.zeros_like(output_tensor.detach()).cpu()
    else:
        gradient_snapshot["__attack_output_gradient__"] = output_tensor.grad.detach().cpu().clone()
    return gradient_snapshot


def train_epoch_dmf(data_set, net, optimizer, criterion, capture_single_step_gradient=False):
    completion = net().T
    if capture_single_step_gradient:
        completion.retain_grad()

    truth = data_set.sparse_X
    m = data_set.M

    loss = criterion(truth.to(torch.float32), (completion * m).to(torch.float32))

    optimizer.zero_grad()
    loss.backward()
    single_step_gradient = None
    if capture_single_step_gradient:
        single_step_gradient = clone_model_gradients_for_attack(net)
        single_step_gradient = attach_output_gradient_for_attack(single_step_gradient, completion)
    optimizer.step()

    return loss.item(), completion * (1 - data_set.M) + data_set.sparse_X, single_step_gradient


def build_mamba_delta_t(timestamp, target_device):
    if timestamp is None:
        raise ValueError("Mamba_DMF requires timestamp when data_form='unequal_sliced'")
    timestamp = torch.as_tensor(timestamp, device=target_device, dtype=torch.float32).flatten()
    if timestamp.numel() == 0:
        raise ValueError("Mamba_DMF requires a non-empty timestamp sequence")
    previous_t = torch.cat([
        torch.full((1,), -1.0, device=target_device, dtype=torch.float32),
        timestamp[:-1],
    ])
    return (timestamp - previous_t).detach()


def train_epoch_mamba(data_set, net, optimizer, criterion, capture_single_step_gradient=False):
    delta_t = build_mamba_delta_t(data_set.timestamp, next(net.parameters()).device)
    completion, _h, _c, _last_t = net(delta_t)
    if capture_single_step_gradient:
        completion.retain_grad()

    truth = data_set.sparse_X
    m = data_set.M
    loss = criterion(truth.to(torch.float32), (completion * m).to(torch.float32))

    optimizer.zero_grad()
    loss.backward()
    single_step_gradient = None
    if capture_single_step_gradient:
        single_step_gradient = clone_model_gradients_for_attack(net)
        single_step_gradient = attach_output_gradient_for_attack(single_step_gradient, completion)
    optimizer.step()

    return loss.item(), completion * (1 - data_set.M) + data_set.sparse_X, single_step_gradient


def init_state(length):
    h = torch.zeros((1, length)).to(device)
    h.requires_grad = False
    return h


def train_epoch_rnn(conf, data_set, net, optimizer, criterion, capture_single_step_gradient=False):
    loss_list = []
    outputs = []
    single_step_gradient = None
    output_prediction_chunks = []
    output_gradient_chunks = []
    output_loss_numel_chunks = []
    h = init_state(conf["z_size"])
    for (x_steps, z_steps, m_steps, _) in data_set:
        y_steps, h = net(z_steps, h)

        loss = criterion(x_steps.to(torch.float32), (y_steps * m_steps).to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        if capture_single_step_gradient and single_step_gradient is None:
            single_step_gradient = clone_model_gradients_for_attack(net)
        optimizer.step()

        outputs.append(y_steps)
        loss_list.append(loss.item())

    completion = torch.cat(outputs, dim=1)
    return mean(loss_list), completion * (1 - data_set.M) + data_set.sparse_X, single_step_gradient


def train_epoch_unequal_lstm(conf, data_set, net, optimizer, criterion, capture_single_step_gradient=False):
    loss_list = []
    outputs = []
    single_step_gradient = None

    last_t = -1

    h = init_state(conf["z_size"])
    c = init_state(conf["c_size"])

    for (x_steps, z_steps, m_steps, t_steps) in data_set:
        y_steps, h, c, last_t = net(z_steps, h, c, t_steps, last_t, conf["delta"])

        loss = criterion(x_steps.to(torch.float32),
                         (y_steps * m_steps).to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        if capture_single_step_gradient and single_step_gradient is None:
            single_step_gradient = clone_model_gradients_for_attack(net)
        optimizer.step()

        outputs.append(y_steps)
        loss_list.append(loss.item())

    completion = torch.cat(outputs, dim=1)

    return mean(loss_list), completion * (1 - data_set.M) + data_set.sparse_X, single_step_gradient


def build_output_gradient_closed_form_snapshot(gradient_snapshot, output_prediction_chunks, output_gradient_chunks, output_loss_numel_chunks):
    if not output_prediction_chunks:
        return gradient_snapshot
    gradient_snapshot["__attack_output_prediction__"] = torch.cat(output_prediction_chunks, dim=1)
    gradient_snapshot["__attack_output_gradient__"] = torch.cat(output_gradient_chunks, dim=1)
    gradient_snapshot["__attack_output_loss_numel__"] = torch.cat(output_loss_numel_chunks, dim=1)
    return gradient_snapshot


def train_epoch_rnn(conf, data_set, net, optimizer, criterion, capture_single_step_gradient=False):
    loss_list = []
    outputs = []
    single_step_gradient = None
    output_prediction_chunks = []
    output_gradient_chunks = []
    output_loss_numel_chunks = []
    h = init_state(conf["z_size"])
    for (x_steps, z_steps, m_steps, _) in data_set:
        y_steps, h = net(z_steps, h)
        if capture_single_step_gradient:
            y_steps.retain_grad()
        loss = criterion(x_steps.to(torch.float32), (y_steps * m_steps).to(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        if capture_single_step_gradient:
            output_prediction_chunks.append(y_steps.detach().cpu().clone())
            if y_steps.grad is None:
                output_gradient_chunks.append(torch.zeros_like(y_steps.detach()).cpu())
            else:
                output_gradient_chunks.append(y_steps.grad.detach().cpu().clone())
            output_loss_numel_chunks.append(torch.full_like(y_steps.detach(), float(y_steps.numel())).cpu())
        if capture_single_step_gradient and single_step_gradient is None:
            single_step_gradient = clone_model_gradients_for_attack(net)
        optimizer.step()
        outputs.append(y_steps)
        loss_list.append(loss.item())
    completion = torch.cat(outputs, dim=1)
    if capture_single_step_gradient and single_step_gradient is not None:
        single_step_gradient = build_output_gradient_closed_form_snapshot(
            single_step_gradient,
            output_prediction_chunks,
            output_gradient_chunks,
            output_loss_numel_chunks,
        )
    return mean(loss_list), completion * (1 - data_set.M) + data_set.sparse_X, single_step_gradient


def train_epoch_unequal_lstm(conf, data_set, net, optimizer, criterion, capture_single_step_gradient=False):
    loss_list = []
    outputs = []
    single_step_gradient = None
    output_prediction_chunks = []
    output_gradient_chunks = []
    output_loss_numel_chunks = []
    last_t = -1
    h = init_state(conf["z_size"])
    c = init_state(conf["c_size"])
    for (x_steps, z_steps, m_steps, t_steps) in data_set:
        y_steps, h, c, last_t = net(z_steps, h, c, t_steps, last_t, conf["delta"])
        if capture_single_step_gradient:
            y_steps.retain_grad()
        loss = criterion(x_steps.to(torch.float32), (y_steps * m_steps).to(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        if capture_single_step_gradient:
            output_prediction_chunks.append(y_steps.detach().cpu().clone())
            if y_steps.grad is None:
                output_gradient_chunks.append(torch.zeros_like(y_steps.detach()).cpu())
            else:
                output_gradient_chunks.append(y_steps.grad.detach().cpu().clone())
            output_loss_numel_chunks.append(torch.full_like(y_steps.detach(), float(y_steps.numel())).cpu())
        if capture_single_step_gradient and single_step_gradient is None:
            single_step_gradient = clone_model_gradients_for_attack(net)
        optimizer.step()
        outputs.append(y_steps)
        loss_list.append(loss.item())
    completion = torch.cat(outputs, dim=1)
    if capture_single_step_gradient and single_step_gradient is not None:
        single_step_gradient = build_output_gradient_closed_form_snapshot(
            single_step_gradient,
            output_prediction_chunks,
            output_gradient_chunks,
            output_loss_numel_chunks,
        )
    return mean(loss_list), completion * (1 - data_set.M) + data_set.sparse_X, single_step_gradient


def validate(completion, ground_truth):
    criterion = torch.nn.MSELoss()
    loss = torch.sqrt(criterion(completion.to(torch.float32), ground_truth.to(torch.float32)))
    return loss.item()


def validate_on_mask(completion, ground_truth, mask):
    mask = mask.to(device=completion.device, dtype=torch.bool)
    if mask.sum().item() == 0:
        return 0.0
    diff = completion.to(torch.float32)[mask] - ground_truth.to(device=completion.device, dtype=torch.float32)[mask]
    return torch.sqrt(torch.mean(diff.pow(2))).item()


def mae_on_mask(completion, ground_truth, mask):
    mask = mask.to(device=completion.device, dtype=torch.bool)
    if mask.sum().item() == 0:
        return 0.0
    diff = completion.to(torch.float32)[mask] - ground_truth.to(device=completion.device, dtype=torch.float32)[mask]
    return torch.mean(torch.abs(diff)).item()
