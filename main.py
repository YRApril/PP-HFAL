import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import torch

import attack
import process_conf
from completion_outputs import evaluate_completion_arrays, save_array, save_common_completion_outputs
from console_utils import NAN, emit_console, format_console_record
from client import Client
from server import Server


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


def set_global_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir:
        os.makedirs(prepare_filesystem_path(parent_dir), exist_ok=True)


def build_atomic_temp_path(target_path):
    target_abs_path = os.path.abspath(target_path)
    target_dir = os.path.dirname(target_abs_path)
    target_digest = hashlib.sha1(target_abs_path.encode("utf-8")).hexdigest()[:12]
    temp_name = f".tmp.json.{target_digest}.{os.getpid()}.{time.time_ns()}"
    return os.path.join(target_dir, temp_name)


def replace_file_atomically(temp_path, target_path):
    os.replace(prepare_filesystem_path(temp_path), prepare_filesystem_path(os.path.abspath(target_path)))


def sanitize_json_value(value):
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_conf(conf_path="conf.json"):
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), conf_path)
    with open(prepare_filesystem_path(conf_path), "r", encoding="utf-8") as f:
        conf = json.load(f)
    set_global_seed(conf.get("seed"))
    process_conf.process(conf)
    return conf


def reset_output_file(out_file_path):
    ensure_parent_dir(out_file_path)
    with open(prepare_filesystem_path(out_file_path), "w", encoding="utf-8") as f:
        f.write("")


def append_output_text(out_file_path, content):
    ensure_parent_dir(out_file_path)
    with open(prepare_filesystem_path(out_file_path), "a", encoding="utf-8") as f:
        f.write(content)


def append_output_console(out_file_path, algorithm, event, *, blank_after=0, **fields):
    append_output_text(out_file_path, format_console_record(algorithm, event, blank_after=blank_after, **fields))


def emit_and_append_output(out_file_path, algorithm, event, *, blank_after=0, **fields):
    emit_console(algorithm, event, blank_after=blank_after, **fields)
    append_output_console(out_file_path, algorithm, event, blank_after=blank_after, **fields)


def write_conf_snapshot(snapshot_path, conf):
    target_path = os.path.abspath(snapshot_path)
    temp_path = build_atomic_temp_path(target_path)
    ensure_parent_dir(target_path)
    snapshot = process_conf.build_serializable_conf_snapshot(conf, excluded_keys=process_conf.SNAPSHOT_EXCLUDED_KEYS)
    snapshot["output_txt_name"] = os.path.basename(conf["out_file_path"])
    snapshot["output_json_name"] = os.path.basename(target_path)
    if conf.get("out_summary_path"):
        snapshot["output_summary_name"] = os.path.basename(conf["out_summary_path"])
    safe_snapshot = sanitize_json_value(snapshot)
    try:
        with open(prepare_filesystem_path(temp_path), "w", encoding="utf-8") as f:
            json.dump(safe_snapshot, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        replace_file_atomically(temp_path, target_path)
    finally:
        if os.path.exists(prepare_filesystem_path(temp_path)):
            os.remove(prepare_filesystem_path(temp_path))


def initialize_server(conf):
    return Server(conf)


def run_server_pretrain(server):
    return server.pretrain_global_model()


def get_server_pretrain_time_cost(pretrain_result):
    return max(float(pretrain_result.get("time_cost", 0.0)), 0.0)


def log_server_pretrain_summary(out_file_path, pretrain_result):
    if not pretrain_result["enabled"]:
        status = "disabled"
    elif not pretrain_result["executed"]:
        status = "skipped"
    else:
        status = "done"
    emit_and_append_output(
        out_file_path,
        "Ours",
        "server_pretrain",
        status=status,
        history_slice_count=pretrain_result["history_slice_count"],
        train_loss=pretrain_result.get("train_loss", NAN),
        validate_loss=pretrain_result.get("validate_loss", NAN),
        time_cost=get_server_pretrain_time_cost(pretrain_result),
    )


def initialize_clients(conf):
    clients = []
    for client_id in range(conf["num_clients"]):
        clients.append(Client(conf, client_id))
    return clients


def initialize_server_and_clients(conf, out_file_path):
    server = initialize_server(conf)
    pretrain_result = run_server_pretrain(server)
    log_server_pretrain_summary(out_file_path, pretrain_result)
    clients = initialize_clients(conf)
    return server, clients, pretrain_result


def select_participating_clients(clients, conf):
    return clients


def create_weight_accumulator(global_model):
    weight_accumulator = {}
    for name, params in global_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params)
    return weight_accumulator


def should_run_attack_eval(conf):
    return bool(conf.get("attack_eval_enable", False))


def should_capture_attack_eval_round(conf, round_idx):
    if not should_run_attack_eval(conf):
        return False
    target_rounds = conf.get("attack_eval_rounds", "last")
    if target_rounds == "last":
        return round_idx == conf["global_epochs"] - 1
    if isinstance(target_rounds, int) and not isinstance(target_rounds, bool):
        return round_idx == target_rounds
    if isinstance(target_rounds, list):
        return round_idx in target_rounds
    return False


def should_capture_attack_eval_client(conf, client):
    target_clients = conf.get("attack_eval_clients", "all")
    if target_clients == "all":
        return True
    return int(client.client_id) in set(target_clients)


def should_capture_single_step_gradient_for_attack(conf, client):
    gradient_capture_methods = {"gradient_matching", "output_gradient_closed_form"}
    if conf.get("attack_eval_method", "output_gradient_closed_form") not in gradient_capture_methods:
        return False
    if conf.get("attack_eval_gradient_source", "single_step_gradient") != "single_step_gradient":
        return False
    return should_capture_attack_eval_client(conf, client)


def clone_tensor_for_attack(value):
    return value.detach().cpu().clone()


def clone_state_dict_for_attack(state_dict):
    return {name: clone_tensor_for_attack(value) for name, value in state_dict.items()}


def build_attack_eval_artifact(conf, client, server_model, uploaded_update, round_idx, single_step_gradient=None):
    artifact = {
        "client_id": int(client.client_id),
        "round_idx": int(round_idx),
        "server_model_state": clone_state_dict_for_attack(server_model.state_dict()),
        "uploaded_update": clone_state_dict_for_attack(uploaded_update),
        "ground_truth": clone_tensor_for_attack(client.ground_truth),
        "sparse_X": clone_tensor_for_attack(client.sparse_X),
        "mask": clone_tensor_for_attack(client.M),
    }
    if single_step_gradient is not None:
        artifact["single_step_gradient"] = clone_state_dict_for_attack(single_step_gradient)
    if conf["data_form"] == "unequal_sliced":
        timestamp = conf.get("timestamp")
        if isinstance(timestamp, torch.Tensor):
            artifact["timestamp"] = clone_tensor_for_attack(timestamp)
    return artifact


def collect_attack_eval_artifact_if_needed(conf, artifacts, client, server_model, client_result, round_idx):
    if not should_capture_attack_eval_client(conf, client):
        return
    artifacts.append(
        build_attack_eval_artifact(
            conf,
            client,
            server_model,
            client_result["diff"],
            round_idx,
            single_step_gradient=client_result.get("single_step_gradient"),
        )
    )


def get_active_learning_timing(conf):
    return conf.get("al_timing") or conf.get("al_when", "per_local_epoch")


def should_run_local_pre_round_active_learning(conf):
    return (
        bool(conf.get("al_enable", False)) and
        not bool(conf.get("cloud_al_enable", False)) and
        get_active_learning_timing(conf) == "per_global_round"
    )


def should_run_local_epoch_active_learning(conf):
    return (
        bool(conf.get("al_enable", False)) and
        not bool(conf.get("cloud_al_enable", False)) and
        get_active_learning_timing(conf) == "per_local_epoch"
    )


def build_empty_local_pre_round_query_result():
    return {
        "active_learning_policy": None,
        "queried_count": 0,
        "local_candidate_count": 0,
        "global_candidate_count": 0,
        "intersection_candidate_count": 0,
        "fusion_candidate_count": 0,
        "metric_candidate_count": None,
        "active_learning_time_cost": 0.0,
    }


def run_local_pre_round_active_query(client, server_model, round_idx):
    _ = round_idx
    start_time = time.time()
    prepare_client_for_pre_training_selection(client, server_model)
    local_scores = client.compute_local_score_map_for_fusion()
    final_budget = get_final_query_budget(client.conf)
    rows, cols = client.select_topk_indices(local_scores["score_map"], final_budget)
    queried_count = client.apply_selected_queries((rows, cols))
    if queried_count > 0:
        client.build_dataset()
    finite_local_candidate_count = int(torch.isfinite(local_scores["score_map"]).sum().item())
    return {
        "active_learning_policy": "local_only_pre_global_round",
        "queried_count": queried_count,
        "local_candidate_count": count_candidate_mask(local_scores["candidate_mask"]),
        "global_candidate_count": 0,
        "intersection_candidate_count": 0,
        "fusion_candidate_count": finite_local_candidate_count,
        "metric_candidate_count": None,
        "active_learning_time_cost": time.time() - start_time,
    }


def train_one_client_for_round(client, server_model, round_idx, capture_single_step_gradient=False):
    pre_round_query_result = build_empty_local_pre_round_query_result()
    if should_run_local_pre_round_active_learning(client.conf):
        pre_round_query_result = run_local_pre_round_active_query(client, server_model, round_idx)

    client_result = client.train_local_epochs(
        server_model,
        round_idx,
        run_active_learning=should_run_local_epoch_active_learning(client.conf),
        capture_single_step_gradient=capture_single_step_gradient,
    )
    client_result.update(pre_round_query_result)
    return client_result


def accumulate_client_diff(weight_accumulator, global_model, diff):
    for name, _ in global_model.state_dict().items():
        weight_accumulator[name].add_(diff[name])


def format_epoch_progress(epoch_idx, total_epochs):
    current_epoch = int(epoch_idx) + 1
    if total_epochs is None:
        return current_epoch
    return f"{current_epoch}/{int(total_epochs)}"


def mean_available_metric(values):
    finite_values = [finite_float_or_none(value) for value in values]
    finite_values = [value for value in finite_values if value is not None]
    if not finite_values:
        return None
    return float(mean(finite_values))


def run_legacy_training_round(server, clients, round_idx, capture_attack_artifacts=False):
    num_participants = len(clients)
    if num_participants == 0:
        raise ValueError("At least one client must participate in each training round")

    weight_accumulator = create_weight_accumulator(server.global_model)
    train_loss_list = []
    validate_loss_list = []
    initial_missing_validate_loss_list = []
    missing_rmse_list = []
    missing_mae_list = []
    observed_rmse_list = []
    time_cost_list = []
    queried_count_list = []
    local_candidate_count_list = []
    global_candidate_count_list = []
    intersection_candidate_count_list = []
    fusion_candidate_count_list = []
    metric_candidate_count_list = []
    active_learning_time_cost_list = []
    active_learning_policy = None
    attack_artifacts = []

    for client in clients:
        capture_client_gradient = capture_attack_artifacts and should_capture_single_step_gradient_for_attack(server.conf, client)
        client_result = train_one_client_for_round(
            client,
            server.global_model,
            round_idx,
            capture_single_step_gradient=capture_client_gradient,
        )
        train_loss_list.append(client_result["train_loss"])
        validate_loss_list.append(client_result["validate_loss"])
        initial_missing_validate_loss_list.append(client_result.get("initial_missing_validate_loss", client_result["validate_loss"]))
        missing_rmse_list.append(client_result.get("missing_rmse", client_result.get("initial_missing_validate_loss", client_result["validate_loss"])))
        missing_mae_list.append(client_result.get("missing_mae"))
        observed_rmse_list.append(client_result.get("observed_rmse"))
        time_cost_list.append(client_result["time_cost"])
        queried_count_list.append(int(client_result.get("queried_count", 0)))
        local_candidate_count_list.append(int(client_result.get("local_candidate_count", 0)))
        global_candidate_count_list.append(int(client_result.get("global_candidate_count", 0)))
        intersection_candidate_count_list.append(int(client_result.get("intersection_candidate_count", 0)))
        fusion_candidate_count_list.append(int(client_result.get("fusion_candidate_count", 0)))
        metric_candidate_count_list.append(client_result.get("metric_candidate_count"))
        active_learning_time_cost_list.append(float(client_result.get("active_learning_time_cost", 0.0)))
        if client_result.get("active_learning_policy") is not None:
            active_learning_policy = client_result["active_learning_policy"]
        if capture_attack_artifacts:
            collect_attack_eval_artifact_if_needed(server.conf, attack_artifacts, client, server.global_model, client_result, round_idx)
        accumulate_client_diff(weight_accumulator, server.global_model, client_result["diff"])

    server.model_aggregate(weight_accumulator, num_participants)
    round_time_cost = sum(time_cost_list)
    return {
        "avg_train_loss": mean(train_loss_list),
        "avg_validate_loss": mean(validate_loss_list),
        "avg_initial_missing_validate_loss": mean(initial_missing_validate_loss_list),
        "missing_rmse": mean_available_metric(missing_rmse_list),
        "missing_mae": mean_available_metric(missing_mae_list),
        "observed_rmse": mean_available_metric(observed_rmse_list),
        "round_time_cost": round_time_cost,
        "train_loss_list": train_loss_list,
        "validate_loss_list": validate_loss_list,
        "initial_missing_validate_loss_list": initial_missing_validate_loss_list,
        "missing_rmse_list": missing_rmse_list,
        "missing_mae_list": missing_mae_list,
        "observed_rmse_list": observed_rmse_list,
        "time_cost_list": time_cost_list,
        "num_participants": num_participants,
        "queried_count_list": queried_count_list,
        "total_queried_count": sum(queried_count_list),
        "local_candidate_count_list": local_candidate_count_list,
        "global_candidate_count_list": global_candidate_count_list,
        "intersection_candidate_count_list": intersection_candidate_count_list,
        "fusion_candidate_count_list": fusion_candidate_count_list,
        "metric_candidate_count_list": metric_candidate_count_list,
        "active_learning_time_cost_list": active_learning_time_cost_list,
        "fusion_policy": active_learning_policy or "none",
        "active_learning_policy": active_learning_policy,
        "attack_artifacts": attack_artifacts,
    }


def log_round_summary(out_file_path, round_idx, round_result, total_epochs=None, blank_after=2):
    emit_and_append_output(
        out_file_path,
        "Ours",
        "progress",
        blank_after=blank_after,
        epoch=format_epoch_progress(round_idx, total_epochs),
        avg_train_loss=round_result["avg_train_loss"],
        avg_validate_loss=round_result["avg_validate_loss"],
        train_loss=round_result["avg_train_loss"],
        validate_loss=round_result["avg_validate_loss"],
        missing_rmse=round_result.get("missing_rmse", round_result.get("avg_initial_missing_validate_loss")),
        missing_mae=round_result.get("missing_mae"),
        observed_rmse=round_result.get("observed_rmse"),
        round_time_cost=round_result["round_time_cost"],
        step_time_sec=NAN,
        elapsed_sec=NAN,
    )


def build_early_stop_state(conf):
    patience = int(conf.get("early_stop_patience", 50))
    min_delta = float(conf.get("early_stop_min_delta", 0.05))
    min_rounds = int(conf.get("early_stop_min_rounds", 1))
    return {
        "enabled": bool(conf.get("early_stop_enable", False)),
        "patience": max(1, patience),
        "min_delta": max(0.0, min_delta),
        "min_rounds": max(1, min_rounds),
        "best_epoch": None,
        "best_validate_loss": None,
        "rounds_without_min_delta": 0,
        "triggered": False,
        "stop_epoch": None,
        "reason": None,
    }


def update_early_stop_state(early_stop_state, round_idx, validate_loss):
    if not early_stop_state.get("enabled", False):
        return False
    current_value = finite_float_or_none(validate_loss)
    if current_value is None:
        return False
    epoch = round_idx + 1
    best_value = early_stop_state.get("best_validate_loss")
    min_delta = early_stop_state["min_delta"]
    if best_value is None or current_value <= best_value - min_delta:
        early_stop_state["best_validate_loss"] = current_value
        early_stop_state["best_epoch"] = epoch
        early_stop_state["rounds_without_min_delta"] = 0
        return False

    early_stop_state["rounds_without_min_delta"] = epoch - int(early_stop_state["best_epoch"])
    can_stop_by_patience = early_stop_state["rounds_without_min_delta"] >= early_stop_state["patience"]
    can_stop_by_min_rounds = epoch >= early_stop_state["min_rounds"]
    if can_stop_by_patience and can_stop_by_min_rounds:
        early_stop_state["triggered"] = True
        early_stop_state["stop_epoch"] = epoch
        early_stop_state["reason"] = (
            "no avg_validate_loss/RMSE improvement >= %.4f for %d consecutive rounds"
            % (early_stop_state["min_delta"], early_stop_state["patience"])
        )
        return True
    return False


def log_early_stop_summary(out_file_path, early_stop_state):
    emit_and_append_output(
        out_file_path,
        "Ours",
        "early_stop",
        blank_after=1,
        stop_epoch=early_stop_state["stop_epoch"],
        best_epoch=early_stop_state["best_epoch"],
        best_validate_loss=early_stop_state["best_validate_loss"],
        patience=early_stop_state["patience"],
        min_delta=early_stop_state["min_delta"],
        rounds_without_min_delta=early_stop_state["rounds_without_min_delta"],
    )


def build_early_prune_state(conf):
    check_epoch = int(conf.get("early_prune_epoch", 30))
    max_validate_loss = finite_float_or_none(conf.get("early_prune_max_validate_loss"))
    return {
        "enabled": bool(conf.get("early_prune_enable", False)) and max_validate_loss is not None,
        "check_epoch": max(1, check_epoch),
        "max_validate_loss": max_validate_loss,
        "triggered": False,
        "stop_epoch": None,
        "observed_validate_loss": None,
        "reason": None,
    }


def update_early_prune_state(early_prune_state, round_idx, validate_loss):
    if not early_prune_state.get("enabled", False):
        return False
    epoch = round_idx + 1
    if epoch != early_prune_state["check_epoch"]:
        return False
    current_value = finite_float_or_none(validate_loss)
    early_prune_state["observed_validate_loss"] = current_value
    if current_value is not None and current_value < early_prune_state["max_validate_loss"]:
        return False
    early_prune_state["triggered"] = True
    early_prune_state["stop_epoch"] = epoch
    early_prune_state["reason"] = (
        "avg_validate_loss/RMSE at epoch %d is not < %.4f"
        % (early_prune_state["check_epoch"], early_prune_state["max_validate_loss"])
    )
    return True


def log_early_prune_summary(out_file_path, early_prune_state):
    observed_value = early_prune_state.get("observed_validate_loss")
    emit_and_append_output(
        out_file_path,
        "Ours",
        "early_prune",
        blank_after=1,
        stop_epoch=early_prune_state["stop_epoch"],
        check_epoch=early_prune_state["check_epoch"],
        observed_validate_loss=observed_value,
        max_validate_loss=early_prune_state["max_validate_loss"],
        reason=early_prune_state["reason"],
    )


def finalize_training_log(out_file_path, all_time_cost, final_round_result=None):
    final_round_result = final_round_result or {}
    emit_and_append_output(
        out_file_path,
        "Ours",
        "done",
        blank_after=1,
        missing_rmse=final_round_result.get("missing_rmse", final_round_result.get("avg_initial_missing_validate_loss")),
        missing_mae=final_round_result.get("missing_mae"),
        observed_rmse=final_round_result.get("observed_rmse"),
        relative_frobenius=NAN,
        total_time_sec=all_time_cost,
        output=out_file_path,
    )


def log_attack_eval_summary(out_file_path, attack_summary):
    if not attack_summary or not attack_summary.get("enabled", False):
        return
    emit_and_append_output(
        out_file_path,
        "Ours",
        "attack_eval",
        blank_after=1,
        method=attack_summary.get("method"),
        clients=f"{attack_summary.get('num_completed', 0)}/{attack_summary.get('num_artifacts', 0)}",
        ASR=attack_summary.get("mean_asr"),
        MSE=attack_summary.get("mean_mse"),
        RMSE=attack_summary.get("mean_rmse"),
        PSNR=attack_summary.get("mean_psnr"),
        attack_time_cost=attack_summary.get("time_cost", 0.0),
    )


def should_run_score_map_active_learning(conf):
    return bool(conf.get("al_enable", False) and conf.get("cloud_al_enable", False))


def get_final_query_budget(conf):
    budget = conf.get("final_query_budget")
    if budget is None:
        budget = conf.get("al_budget", 0)
    return max(int(budget), 0)


def build_cloud_client_context(client, round_idx, local_candidate_mask=None):
    return {
        "client_id": client.client_id,
        "round_idx": round_idx,
        "data_list": client.get_data(),
        "local_candidate_mask": local_candidate_mask,
    }


def count_candidate_mask(candidate_mask):
    return int(candidate_mask.sum().item())


def normalize_score_map(score_map, valid_mask, norm_method):
    score_map = score_map.to(torch.float32)
    normalized_score_map = torch.zeros_like(score_map, dtype=torch.float32)
    valid_values = score_map[valid_mask]

    if valid_values.numel() == 0:
        return normalized_score_map

    if norm_method == "none":
        normalized_values = valid_values
    elif norm_method == "minmax":
        min_value = valid_values.min()
        max_value = valid_values.max()
        span = max_value - min_value
        if span.abs().item() <= 1e-12:
            normalized_values = torch.zeros_like(valid_values)
        else:
            normalized_values = (valid_values - min_value) / span
    elif norm_method == "zscore":
        mean_value = valid_values.mean()
        std_value = valid_values.std(unbiased=False)
        if std_value.abs().item() <= 1e-12:
            normalized_values = torch.zeros_like(valid_values)
        else:
            normalized_values = (valid_values - mean_value) / std_value
    elif norm_method == "rank":
        normalized_values = torch.zeros_like(valid_values)
        order = torch.argsort(valid_values, descending=True)
        if valid_values.numel() == 1:
            normalized_values[order] = 1.0
        else:
            normalized_values[order] = torch.linspace(
                1.0, 0.0, steps=valid_values.numel(), device=score_map.device, dtype=torch.float32
            )
    else:
        raise ValueError("Unknown score_norm")

    normalized_score_map[valid_mask] = normalized_values
    return normalized_score_map


def build_intersection_fusion(local_scores, global_scores):
    local_candidate_mask = local_scores["candidate_mask"].to(dtype=torch.bool)
    global_candidate_mask = global_scores["candidate_mask"].to(dtype=torch.bool)
    intersection_mask = local_candidate_mask & global_candidate_mask
    local_score_map = local_scores["score_map"].to(torch.float32)
    global_score_map = global_scores["score_map"].to(torch.float32)
    fused_score_map = torch.full_like(local_score_map, float("-inf"))
    fused_score_map = torch.where(intersection_mask, local_score_map + global_score_map, fused_score_map)

    return {
        "score_map": fused_score_map,
        "candidate_mask": intersection_mask,
        "meta": {
            "fusion_policy": "intersection",
            "local_candidate_count": count_candidate_mask(local_candidate_mask),
            "global_candidate_count": count_candidate_mask(global_candidate_mask),
            "intersection_candidate_count": count_candidate_mask(intersection_mask),
            "fusion_candidate_count": count_candidate_mask(intersection_mask),
            "metric_candidate_count": global_scores.get("meta", {}).get("metric_candidate_count"),
            "score_rule": "local_score_plus_global_score_inside_intersection",
        },
    }


def build_weighted_sum_fusion(local_scores, global_scores, conf):
    local_score_map = local_scores["score_map"].to(torch.float32)
    global_score_map = global_scores["score_map"].to(torch.float32)
    local_valid_mask = torch.isfinite(local_score_map)
    global_valid_mask = torch.isfinite(global_score_map)
    fusion_candidate_mask = local_valid_mask & global_valid_mask
    local_candidate_mask = local_scores["candidate_mask"].to(dtype=torch.bool)
    global_candidate_mask = global_scores["candidate_mask"].to(dtype=torch.bool)
    intersection_mask = local_candidate_mask & global_candidate_mask

    score_norm = conf.get("score_norm", "minmax")
    fusion_alpha = float(conf.get("fusion_alpha", 0.5))
    fusion_beta = float(conf.get("fusion_beta", 0.5))
    local_norm_score_map = normalize_score_map(local_score_map, fusion_candidate_mask, score_norm)
    global_norm_score_map = normalize_score_map(global_score_map, fusion_candidate_mask, score_norm)
    fused_score_map = fusion_alpha * local_norm_score_map + fusion_beta * global_norm_score_map
    fused_score_map = fused_score_map.masked_fill(~fusion_candidate_mask, float("-inf"))

    return {
        "score_map": fused_score_map,
        "candidate_mask": fusion_candidate_mask,
        "meta": {
            "fusion_policy": "weighted_sum",
            "local_candidate_count": count_candidate_mask(local_candidate_mask),
            "global_candidate_count": count_candidate_mask(global_candidate_mask),
            "intersection_candidate_count": count_candidate_mask(intersection_mask),
            "fusion_candidate_count": count_candidate_mask(fusion_candidate_mask),
            "metric_candidate_count": global_scores.get("meta", {}).get("metric_candidate_count"),
            "score_rule": "alpha_local_norm_plus_beta_global_norm",
            "score_norm": score_norm,
            "fusion_alpha": fusion_alpha,
            "fusion_beta": fusion_beta,
        },
    }


def normalize_feature_matrix_for_cosine(features):
    features = torch.nan_to_num(features.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = features.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return features / norm


def compute_dynamic_redundancy_values(normalized_features, selected_candidate_positions):
    if not selected_candidate_positions:
        return torch.zeros(normalized_features.size(0), device=normalized_features.device, dtype=torch.float32)
    selected_positions = torch.tensor(
        selected_candidate_positions, device=normalized_features.device, dtype=torch.long
    )
    selected_features = normalized_features[selected_positions]
    cosine_to_selected = torch.matmul(normalized_features, selected_features.T)
    return cosine_to_selected.max(dim=1).values


def run_cloud_joint_greedy_selection(candidate_rows, candidate_cols, candidate_features, utility_values,
                                     representativeness_values, anomaly_values, conf):
    final_budget = get_final_query_budget(conf)
    candidate_count = candidate_rows.numel()
    if candidate_count == 0 or final_budget <= 0:
        empty = torch.empty(0, device=candidate_rows.device, dtype=torch.long)
        return {
            "rows": empty,
            "cols": empty,
            "selected_candidate_positions": [],
            "selected_costs": [],
            "last_cost_values": torch.empty(0, device=candidate_rows.device, dtype=torch.float32),
        }

    utility_weight = float(conf.get("cloud_joint_utility_weight", 1.0))
    represent_weight = float(conf.get("cloud_joint_represent_weight", 0.5))
    anomaly_weight = float(conf.get("cloud_joint_anomaly_weight", 0.3))
    redundancy_weight = float(conf.get("cloud_joint_redundancy_weight", 0.5))
    normalized_features = normalize_feature_matrix_for_cosine(candidate_features)
    remaining_mask = torch.ones(candidate_count, device=candidate_rows.device, dtype=torch.bool)
    selected_candidate_positions = []
    selected_costs = []
    last_cost_values = torch.full((candidate_count,), float("inf"), device=candidate_rows.device, dtype=torch.float32)

    for _ in range(min(final_budget, candidate_count)):
        redundancy_values = compute_dynamic_redundancy_values(
            normalized_features, selected_candidate_positions
        )
        cost_values = (
            -utility_weight * utility_values -
            represent_weight * representativeness_values +
            anomaly_weight * anomaly_values +
            redundancy_weight * redundancy_values
        )
        cost_values = cost_values.masked_fill(~remaining_mask, float("inf"))
        selected_position = int(torch.argmin(cost_values).item())
        selected_cost = cost_values[selected_position]
        if not torch.isfinite(selected_cost):
            break
        selected_candidate_positions.append(selected_position)
        selected_costs.append(float(selected_cost.detach().cpu().item()))
        remaining_mask[selected_position] = False
        last_cost_values = cost_values

    if not selected_candidate_positions:
        empty = torch.empty(0, device=candidate_rows.device, dtype=torch.long)
        return {
            "rows": empty,
            "cols": empty,
            "selected_candidate_positions": [],
            "selected_costs": [],
            "last_cost_values": last_cost_values,
        }

    selected_positions_tensor = torch.tensor(
        selected_candidate_positions, device=candidate_rows.device, dtype=torch.long
    )
    return {
        "rows": candidate_rows[selected_positions_tensor],
        "cols": candidate_cols[selected_positions_tensor],
        "selected_candidate_positions": selected_candidate_positions,
        "selected_costs": selected_costs,
        "last_cost_values": last_cost_values,
    }


def build_cloud_joint_greedy_fusion(local_scores, global_scores, conf):
    if "joint" not in global_scores:
        raise ValueError("fusion_policy='cloud_joint_greedy' requires global_strategy='cloud_joint_cost'")

    local_score_map = local_scores["score_map"].to(torch.float32)
    global_score_map = global_scores["score_map"].to(torch.float32)
    local_candidate_mask = local_scores["candidate_mask"].to(dtype=torch.bool)
    global_candidate_mask = global_scores["candidate_mask"].to(dtype=torch.bool)
    intersection_mask = local_candidate_mask & global_candidate_mask
    candidate_source = conf.get("cloud_joint_candidate_source", "rank_union")

    if not global_scores["joint"]:
        empty = torch.empty(0, device=local_score_map.device, dtype=torch.long)
        empty_candidate_mask = torch.zeros_like(local_candidate_mask, dtype=torch.bool)
        return {
            "score_map": torch.full_like(local_score_map, float("-inf")),
            "candidate_mask": empty_candidate_mask,
            "selected_rows": empty,
            "selected_cols": empty,
            "meta": {
                "fusion_policy": "cloud_joint_greedy",
                "local_candidate_count": count_candidate_mask(local_candidate_mask),
                "global_candidate_count": count_candidate_mask(global_candidate_mask),
                "intersection_candidate_count": count_candidate_mask(intersection_mask),
                "fusion_candidate_count": 0,
                "metric_candidate_count": global_scores.get("meta", {}).get("metric_candidate_count", 0),
                "cloud_joint_candidate_source": candidate_source,
                "selected_count": 0,
                "score_rule": "empty_cloud_joint_missing_domain",
            },
        }

    joint_candidate_mask = global_scores["joint"]["candidate_mask"].to(dtype=torch.bool)
    base_candidate_mask = torch.isfinite(local_score_map) & torch.isfinite(global_score_map) & joint_candidate_mask
    if candidate_source == "rank_union":
        fusion_candidate_mask = base_candidate_mask
    elif candidate_source == "full_missing":
        fusion_candidate_mask = base_candidate_mask
    else:
        raise ValueError("cloud_joint_candidate_source must be 'rank_union' or 'full_missing'")

    candidate_rows, candidate_cols = torch.where(fusion_candidate_mask)
    if candidate_rows.numel() == 0:
        empty = torch.empty(0, device=local_score_map.device, dtype=torch.long)
        return {
            "score_map": torch.full_like(local_score_map, float("-inf")),
            "candidate_mask": fusion_candidate_mask,
            "selected_rows": empty,
            "selected_cols": empty,
            "meta": {
                "fusion_policy": "cloud_joint_greedy",
                "local_candidate_count": count_candidate_mask(local_candidate_mask),
                "global_candidate_count": count_candidate_mask(global_candidate_mask),
                "intersection_candidate_count": count_candidate_mask(intersection_mask),
                "fusion_candidate_count": 0,
                "metric_candidate_count": global_scores.get("meta", {}).get("metric_candidate_count"),
                "cloud_joint_candidate_source": candidate_source,
                "selected_count": 0,
                "score_rule": "empty_joint_candidate_domain",
            },
        }

    score_norm = conf.get("score_norm", "minmax")
    local_norm_score_map = normalize_score_map(local_score_map, fusion_candidate_mask, score_norm)
    cloud_utility_map = global_scores["joint"]["utility_score_map"].to(torch.float32)
    fusion_alpha = float(conf.get("fusion_alpha", 0.5))
    fusion_beta = float(conf.get("fusion_beta", 0.5))
    utility_denominator = max(fusion_alpha + fusion_beta, 1e-12)
    joint_utility_map = (
        fusion_alpha * local_norm_score_map + fusion_beta * cloud_utility_map
    ) / utility_denominator

    utility_values = joint_utility_map[candidate_rows, candidate_cols].to(torch.float32)
    representativeness_values = global_scores["joint"]["representativeness_score_map"][
        candidate_rows, candidate_cols
    ].to(torch.float32)
    anomaly_values = global_scores["joint"]["anomaly_score_map"][
        candidate_rows, candidate_cols
    ].to(torch.float32)
    candidate_features = global_scores["joint"]["feature_map"][
        candidate_rows, candidate_cols
    ].to(torch.float32)

    selection = run_cloud_joint_greedy_selection(
        candidate_rows,
        candidate_cols,
        candidate_features,
        utility_values,
        representativeness_values,
        anomaly_values,
        conf,
    )

    fused_score_map = torch.full_like(local_score_map, float("-inf"))
    base_cost_map = (
        -float(conf.get("cloud_joint_utility_weight", 1.0)) * joint_utility_map -
        float(conf.get("cloud_joint_represent_weight", 0.5)) * global_scores["joint"]["representativeness_score_map"].to(torch.float32) +
        float(conf.get("cloud_joint_anomaly_weight", 0.3)) * global_scores["joint"]["anomaly_score_map"].to(torch.float32)
    )
    fused_score_map[fusion_candidate_mask] = -base_cost_map[fusion_candidate_mask]
    if selection["rows"].numel() > 0:
        selected_cost_tensor = torch.tensor(
            selection["selected_costs"], device=local_score_map.device, dtype=torch.float32
        )
        fused_score_map[selection["rows"], selection["cols"]] = -selected_cost_tensor

    return {
        "score_map": fused_score_map,
        "candidate_mask": fusion_candidate_mask,
        "selected_rows": selection["rows"],
        "selected_cols": selection["cols"],
        "meta": {
            "fusion_policy": "cloud_joint_greedy",
            "local_candidate_count": count_candidate_mask(local_candidate_mask),
            "global_candidate_count": count_candidate_mask(global_candidate_mask),
            "intersection_candidate_count": count_candidate_mask(intersection_mask),
            "fusion_candidate_count": count_candidate_mask(fusion_candidate_mask),
            "metric_candidate_count": global_scores.get("meta", {}).get("metric_candidate_count"),
            "cloud_joint_candidate_source": candidate_source,
            "selected_count": int(selection["rows"].numel()),
            "score_rule": "minimize_joint_cost_with_dynamic_redundancy",
            "score_norm": score_norm,
            "fusion_alpha": fusion_alpha,
            "fusion_beta": fusion_beta,
            "cloud_joint_utility_weight": float(conf.get("cloud_joint_utility_weight", 1.0)),
            "cloud_joint_represent_weight": float(conf.get("cloud_joint_represent_weight", 0.5)),
            "cloud_joint_anomaly_weight": float(conf.get("cloud_joint_anomaly_weight", 0.3)),
            "cloud_joint_redundancy_weight": float(conf.get("cloud_joint_redundancy_weight", 0.5)),
            "selected_costs": selection["selected_costs"],
        },
    }


def fuse_score_maps(local_scores, global_scores, conf):
    policy = conf.get("fusion_policy", "intersection")
    if policy == "intersection":
        return build_intersection_fusion(local_scores, global_scores)
    if policy == "weighted_sum":
        return build_weighted_sum_fusion(local_scores, global_scores, conf)
    if policy == "cloud_joint_greedy":
        return build_cloud_joint_greedy_fusion(local_scores, global_scores, conf)
    raise NotImplementedError("Unsupported fusion_policy")


def apply_fused_queries(client, fused_scores, conf):
    final_budget = get_final_query_budget(conf)
    if "selected_rows" in fused_scores and "selected_cols" in fused_scores:
        rows = fused_scores["selected_rows"][:final_budget]
        cols = fused_scores["selected_cols"][:final_budget]
    else:
        rows, cols = client.select_topk_indices(fused_scores["score_map"], final_budget)
    queried_count = client.apply_selected_queries((rows, cols))
    if queried_count > 0:
        client.build_dataset()
    return {
        "rows": rows,
        "cols": cols,
        "queried_count": queried_count,
    }


def build_selected_indices_history_payload(selection_result):
    rows = selection_result["rows"].detach().cpu().tolist()
    cols = selection_result["cols"].detach().cpu().tolist()
    return list(zip(rows, cols))


def build_score_snapshot(round_idx, local_scores, global_scores, fused_scores, selection_result):
    return {
        "round_idx": round_idx,
        "local_meta": local_scores["meta"],
        "global_meta": global_scores["meta"],
        "fusion_meta": fused_scores["meta"],
        "queried_count": selection_result["queried_count"],
    }


def build_cloud_history_summary(server):
    cloud_history = getattr(server, "cloud_history", None)
    if not cloud_history:
        return None

    proxy_data = cloud_history.get("proxy_data")
    proxy_shape = list(proxy_data.shape) if hasattr(proxy_data, "shape") else None
    per_client_summary = {}
    for client_id, client_history in cloud_history.get("per_client", {}).items():
        selected_indices = client_history.get("selected_indices", [])
        score_snapshots = client_history.get("score_snapshots", [])
        per_client_summary[str(client_id)] = {
            "num_selected_rounds": len(selected_indices),
            "num_score_snapshots": len(score_snapshots),
            "selected_indices": process_conf.serialize_conf_value(selected_indices),
            "score_snapshots": process_conf.serialize_conf_value(score_snapshots),
        }

    return {
        "source": cloud_history.get("source"),
        "meta": process_conf.serialize_conf_value(cloud_history.get("meta", {})),
        "proxy_data_type": type(proxy_data).__name__ if proxy_data is not None else None,
        "proxy_data_shape": proxy_shape,
        "per_client": per_client_summary,
    }


def prepare_client_for_pre_training_selection(client, server_model):
    client.sync_with_server(server_model)
    client.build_dataset()


def train_one_client_for_score_map_round(client, server, round_idx, capture_single_step_gradient=False):
    client_round_start_time = time.time()
    prepare_client_for_pre_training_selection(client, server.global_model)
    local_scores = client.compute_local_score_map_for_fusion()
    missing_mask = client.get_missing_mask()
    client_context = build_cloud_client_context(client, round_idx, local_candidate_mask=local_scores["candidate_mask"])
    global_scores = server.compute_global_score_map(client.client_id, missing_mask, client_context)
    fused_scores = fuse_score_maps(local_scores, global_scores, server.conf)
    selection_result = apply_fused_queries(client, fused_scores, server.conf)
    server.update_cloud_history(
        client.client_id,
        selected_indices=build_selected_indices_history_payload(selection_result),
        score_snapshot=build_score_snapshot(round_idx, local_scores, global_scores, fused_scores, selection_result),
    )

    client_result = client.train_local_epochs(
        server.global_model,
        round_idx,
        run_active_learning=False,
        capture_single_step_gradient=capture_single_step_gradient,
    )
    end_to_end_time_cost = time.time() - client_round_start_time
    training_time_cost = float(client_result["time_cost"])
    active_learning_time_cost = max(end_to_end_time_cost - training_time_cost, 0.0)

    client_result["queried_count"] = selection_result["queried_count"]
    client_result["local_candidate_count"] = fused_scores["meta"]["local_candidate_count"]
    client_result["global_candidate_count"] = fused_scores["meta"]["global_candidate_count"]
    client_result["intersection_candidate_count"] = fused_scores["meta"]["intersection_candidate_count"]
    client_result["fusion_candidate_count"] = fused_scores["meta"]["fusion_candidate_count"]
    client_result["metric_candidate_count"] = fused_scores["meta"].get("metric_candidate_count")
    client_result["training_time_cost"] = training_time_cost
    client_result["active_learning_time_cost"] = active_learning_time_cost
    client_result["end_to_end_time_cost"] = end_to_end_time_cost
    return client_result


def run_score_map_training_round(server, clients, round_idx, capture_attack_artifacts=False):
    num_participants = len(clients)
    if num_participants == 0:
        raise ValueError("At least one client must participate in each training round")

    weight_accumulator = create_weight_accumulator(server.global_model)
    train_loss_list = []
    validate_loss_list = []
    initial_missing_validate_loss_list = []
    missing_rmse_list = []
    missing_mae_list = []
    observed_rmse_list = []
    time_cost_list = []
    training_time_cost_list = []
    active_learning_time_cost_list = []
    queried_count_list = []
    local_candidate_count_list = []
    global_candidate_count_list = []
    intersection_candidate_count_list = []
    fusion_candidate_count_list = []
    metric_candidate_count_list = []
    attack_artifacts = []

    for client in clients:
        capture_client_gradient = capture_attack_artifacts and should_capture_single_step_gradient_for_attack(server.conf, client)
        client_result = train_one_client_for_score_map_round(
            client,
            server,
            round_idx,
            capture_single_step_gradient=capture_client_gradient,
        )
        train_loss_list.append(client_result["train_loss"])
        validate_loss_list.append(client_result["validate_loss"])
        initial_missing_validate_loss_list.append(client_result.get("initial_missing_validate_loss", client_result["validate_loss"]))
        missing_rmse_list.append(client_result.get("missing_rmse", client_result.get("initial_missing_validate_loss", client_result["validate_loss"])))
        missing_mae_list.append(client_result.get("missing_mae"))
        observed_rmse_list.append(client_result.get("observed_rmse"))
        time_cost_list.append(client_result["end_to_end_time_cost"])
        training_time_cost_list.append(client_result["training_time_cost"])
        active_learning_time_cost_list.append(client_result["active_learning_time_cost"])
        queried_count_list.append(client_result["queried_count"])
        local_candidate_count_list.append(client_result["local_candidate_count"])
        global_candidate_count_list.append(client_result["global_candidate_count"])
        intersection_candidate_count_list.append(client_result["intersection_candidate_count"])
        fusion_candidate_count_list.append(client_result["fusion_candidate_count"])
        metric_candidate_count_list.append(client_result.get("metric_candidate_count"))
        if capture_attack_artifacts:
            collect_attack_eval_artifact_if_needed(server.conf, attack_artifacts, client, server.global_model, client_result, round_idx)
        accumulate_client_diff(weight_accumulator, server.global_model, client_result["diff"])

    server.model_aggregate(weight_accumulator, num_participants)
    round_time_cost = sum(time_cost_list)
    return {
        "avg_train_loss": mean(train_loss_list),
        "avg_validate_loss": mean(validate_loss_list),
        "avg_initial_missing_validate_loss": mean(initial_missing_validate_loss_list),
        "missing_rmse": mean_available_metric(missing_rmse_list),
        "missing_mae": mean_available_metric(missing_mae_list),
        "observed_rmse": mean_available_metric(observed_rmse_list),
        "round_time_cost": round_time_cost,
        "train_loss_list": train_loss_list,
        "validate_loss_list": validate_loss_list,
        "initial_missing_validate_loss_list": initial_missing_validate_loss_list,
        "missing_rmse_list": missing_rmse_list,
        "missing_mae_list": missing_mae_list,
        "observed_rmse_list": observed_rmse_list,
        "time_cost_list": time_cost_list,
        "training_time_cost_list": training_time_cost_list,
        "active_learning_time_cost_list": active_learning_time_cost_list,
        "num_participants": num_participants,
        "queried_count_list": queried_count_list,
        "total_queried_count": sum(queried_count_list),
        "local_candidate_count_list": local_candidate_count_list,
        "global_candidate_count_list": global_candidate_count_list,
        "intersection_candidate_count_list": intersection_candidate_count_list,
        "fusion_candidate_count_list": fusion_candidate_count_list,
        "metric_candidate_count_list": metric_candidate_count_list,
        "fusion_policy": server.conf.get("fusion_policy", "intersection"),
        "attack_artifacts": attack_artifacts,
    }


def log_score_map_round_query_summary(out_file_path, round_idx, round_result, total_epochs=None, blank_after=2):
    emit_and_append_output(
        out_file_path,
        "Ours",
        "al",
        blank_after=blank_after,
        epoch=format_epoch_progress(round_idx, total_epochs),
        policy=round_result["fusion_policy"],
        total_queried=round_result["total_queried_count"],
        fusion_candidates=round_result["fusion_candidate_count_list"],
        metric_reference_candidates=round_result.get("metric_candidate_count_list"),
        intersection_candidates=round_result["intersection_candidate_count_list"],
    )


def build_score_map_round_context(server, clients, round_idx):
    return {
        "round_idx": round_idx,
        "server": server,
        "clients": clients,
        "global_model": server.global_model,
    }


def run_score_map_training_round_reserved(server, clients, round_idx):
    _ = build_score_map_round_context(server, clients, round_idx)
    raise NotImplementedError("Future score-map orchestration is reserved but not wired into main flow yet.")


def run_training_reserved(conf, server, clients):
    all_time_cost = 0
    last_round_result = None
    for round_idx in range(conf["global_epochs"]):
        participating_clients = select_participating_clients(clients, conf)
        round_result = run_legacy_training_round(server, participating_clients, round_idx)
        last_round_result = round_result
        all_time_cost += round_result["round_time_cost"]
        log_round_summary(conf["out_file_path"], round_idx, round_result, total_epochs=conf["global_epochs"])
    finalize_training_log(conf["out_file_path"], all_time_cost, last_round_result)
    return all_time_cost


def run_training(conf, server, clients, pretrain_result=None):
    all_time_cost = get_server_pretrain_time_cost(pretrain_result or {})
    use_score_map_flow = should_run_score_map_active_learning(conf)
    round_runner = run_score_map_training_round if use_score_map_flow else run_legacy_training_round
    round_results = []
    attack_artifacts = []
    early_stop_state = build_early_stop_state(conf)
    early_prune_state = build_early_prune_state(conf)

    for round_idx in range(conf["global_epochs"]):
        participating_clients = select_participating_clients(clients, conf)
        capture_attack_artifacts = should_capture_attack_eval_round(conf, round_idx)
        round_result = round_runner(
            server,
            participating_clients,
            round_idx,
            capture_attack_artifacts=capture_attack_artifacts,
        )
        attack_artifacts.extend(round_result.pop("attack_artifacts", []))
        all_time_cost += round_result["round_time_cost"]
        round_snapshot = dict(round_result)
        round_snapshot["round_idx"] = round_idx
        round_snapshot["num_participants"] = len(participating_clients)
        round_results.append(round_snapshot)
        has_al_summary = use_score_map_flow or round_result.get("active_learning_policy") is not None
        log_round_summary(conf["out_file_path"], round_idx, round_result, total_epochs=conf["global_epochs"], blank_after=0 if has_al_summary else 2)
        if has_al_summary:
            log_score_map_round_query_summary(conf["out_file_path"], round_idx, round_result, total_epochs=conf["global_epochs"], blank_after=2)
        if update_early_prune_state(early_prune_state, round_idx, round_result["avg_validate_loss"]):
            log_early_prune_summary(conf["out_file_path"], early_prune_state)
            break
        if update_early_stop_state(early_stop_state, round_idx, round_result["avg_validate_loss"]):
            log_early_stop_summary(conf["out_file_path"], early_stop_state)
            break

    attack_eval_summary = attack.run_attack_evaluation(conf, attack_artifacts)
    log_attack_eval_summary(conf["out_file_path"], attack_eval_summary)
    finalize_training_log(conf["out_file_path"], all_time_cost, round_results[-1] if round_results else None)
    cloud_history_summary = build_cloud_history_summary(server) if use_score_map_flow else None
    return {
        "all_time_cost": all_time_cost,
        "round_results": round_results,
        "use_score_map_flow": use_score_map_flow,
        "cloud_history_summary": cloud_history_summary,
        "attack_eval": attack_eval_summary,
        "early_stop": early_stop_state,
        "early_prune": early_prune_state,
    }


def get_current_time_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def tensor_to_numpy(value):
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def write_ours_common_completion_outputs(conf, server, clients, training_result):
    if not clients:
        return None
    public_dir = Path(conf.get("output_run_dir") or Path(conf["out_file_path"]).parent)

    client_completed = []
    client_initial_observed_masks = []
    client_current_observed_masks = []
    per_client_metrics = []

    with torch.no_grad():
        for client in clients:
            client.sync_with_server(server.global_model)
            client.build_dataset()
            completion = client.predict_completion_only()
            completion_np = tensor_to_numpy(completion).astype(np.float32)
            full_np = tensor_to_numpy(client.ground_truth).astype(np.float32)
            initial_observed_np = (~client.initial_missing_mask).detach().cpu().numpy().astype(bool)
            current_observed_np = tensor_to_numpy(client.M).astype(bool)
            metrics = evaluate_completion_arrays(full_np, initial_observed_np, completion_np)
            per_client_metrics.append(
                {
                    "client_id": int(client.client_id),
                    "initial_observed_count": int(initial_observed_np.sum()),
                    "initial_missing_count": int((~initial_observed_np).sum()),
                    "current_observed_count": int(current_observed_np.sum()),
                    "current_missing_count": int((~current_observed_np).sum()),
                    **metrics,
                }
            )
            client_completed.append(completion_np)
            client_initial_observed_masks.append(initial_observed_np)
            client_current_observed_masks.append(current_observed_np)

    client_completed_stack = np.stack(client_completed, axis=0)
    initial_observed_stack = np.stack(client_initial_observed_masks, axis=0)
    current_observed_stack = np.stack(client_current_observed_masks, axis=0)
    full_matrix = tensor_to_numpy(clients[0].ground_truth).astype(np.float32)
    public_completed = np.mean(client_completed_stack, axis=0).astype(np.float32)
    public_observed_mask = np.any(initial_observed_stack, axis=0)
    client_initial_observed_counts = [int(mask.sum()) for mask in client_initial_observed_masks]
    client_initial_missing_counts = [int((~mask).sum()) for mask in client_initial_observed_masks]
    client_current_observed_counts = [int(mask.sum()) for mask in client_current_observed_masks]
    client_current_missing_counts = [int((~mask).sum()) for mask in client_current_observed_masks]

    def summarize_client_mask_count(counts):
        if not counts:
            return None
        if len(set(counts)) == 1:
            return int(counts[0])
        return int(round(float(np.mean(counts))))

    initial_observed_count_metric = summarize_client_mask_count(client_initial_observed_counts)
    initial_missing_count_metric = summarize_client_mask_count(client_initial_missing_counts)

    client_completed_path = public_dir / "ours_client_completed.npy"
    initial_mask_path = public_dir / "ours_client_initial_observed_mask.npy"
    current_mask_path = public_dir / "ours_client_current_observed_mask.npy"
    save_array(client_completed_path, client_completed_stack)
    save_array(initial_mask_path, initial_observed_stack, mask=True)
    save_array(current_mask_path, current_observed_stack, mask=True)

    round_results = training_result.get("round_results", [])
    final_round = round_results[-1] if round_results else {}
    attack_eval_summary = training_result.get("attack_eval") or {}
    attack_artifact_file = attack_eval_summary.get("artifact_file")
    config_snapshot = process_conf.build_serializable_conf_snapshot(
        conf,
        excluded_keys=process_conf.SNAPSHOT_EXCLUDED_KEYS,
    )
    artifacts = save_common_completion_outputs(
        public_dir,
        algorithm="Ours",
        config=config_snapshot,
        metrics={
            "missing_rmse": final_round.get("missing_rmse", final_round.get("avg_initial_missing_validate_loss")),
            "missing_mae": final_round.get("missing_mae"),
            "missing_count": initial_missing_count_metric,
            "observed_count": initial_observed_count_metric,
            "observed_rmse": final_round.get("observed_rmse"),
            "train_loss": final_round.get("avg_train_loss"),
            "validate_loss": final_round.get("avg_validate_loss"),
            "total_time_sec": training_result.get("all_time_cost"),
            "num_rounds": len(round_results),
            "round_results": round_results,
            "aggregation_method": "client_average",
            "metric_count_source": "per_client_initial_masks",
            "client_initial_missing_count_list": client_initial_missing_counts,
            "client_initial_observed_count_list": client_initial_observed_counts,
            "client_current_missing_count_list": client_current_missing_counts,
            "client_current_observed_count_list": client_current_observed_counts,
            "client_total_initial_missing_count": int(sum(client_initial_missing_counts)),
            "client_total_initial_observed_count": int(sum(client_initial_observed_counts)),
            "client_total_current_missing_count": int(sum(client_current_missing_counts)),
            "client_total_current_observed_count": int(sum(client_current_observed_counts)),
            "attack_eval_enabled": bool(attack_eval_summary.get("enabled", False)),
            "attack_eval_mean_asr": attack_eval_summary.get("mean_asr"),
            "attack_eval_mean_mse": attack_eval_summary.get("mean_mse"),
            "attack_eval_mean_rmse": attack_eval_summary.get("mean_rmse"),
            "attack_eval_mean_psnr": attack_eval_summary.get("mean_psnr"),
            "attack_artifacts_npz": attack_artifact_file,
        },
        history=round_results,
        full_matrix=full_matrix,
        observed_mask=public_observed_mask,
        completed=public_completed,
        per_client_metrics=per_client_metrics,
        legacy_artifacts={
            "txt": conf["out_file_path"],
            "snapshot_json": conf["out_conf_snapshot_path"],
            "summary_json": conf["out_summary_path"],
            "client_completed_npy": str(client_completed_path),
            "client_initial_observed_mask_npy": str(initial_mask_path),
            "client_current_observed_mask_npy": str(current_mask_path),
            "attack_artifacts_npz": attack_artifact_file,
        },
        notes={
            "completed_npy": "Mean of final global-model completions over all clients.",
            "observed_mask_npy": "Union of clients' initial observed masks; per-client masks are saved separately.",
            "metric_source": "Top-level Ours metrics prefer per-client initial-mask metrics because clients have different masks.",
            "missing_count": "Top-level missing_count is the per-client initial missing count aligned with missing_rmse; computed_from_saved_matrices still uses the public union observed_mask.",
            "raw_prediction": "Not saved because Ours exposes completion with observed entries restored, not a separate raw prediction matrix.",
        },
        prefer_provided_metrics=True,
    )
    return artifacts


def safe_mean(values):
    if not values:
        return None
    finite_values = [finite_float_or_none(value) for value in values]
    if any(value is None for value in finite_values):
        return None
    return float(mean(finite_values))


def finite_float_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    float_value = float(value)
    return float_value if math.isfinite(float_value) else None


def safe_min(values):
    if not values:
        return None
    finite_values = [finite_float_or_none(value) for value in values]
    if any(value is None for value in finite_values):
        return None
    return min(finite_values)


def build_run_summary(conf, pretrain_result, training_result, started_at, finished_at):
    round_results = training_result.get("round_results", [])
    avg_train_loss_list = [float(item["avg_train_loss"]) for item in round_results]
    avg_validate_loss_list = [float(item["avg_validate_loss"]) for item in round_results]
    validate_loss_epoch30 = avg_validate_loss_list[29] if len(avg_validate_loss_list) >= 30 else None
    last_window_size = min(5, len(avg_validate_loss_list))
    last_validate_window = avg_validate_loss_list[-last_window_size:] if last_window_size > 0 else []
    last_train_window = avg_train_loss_list[-last_window_size:] if last_window_size > 0 else []
    config_snapshot = process_conf.build_serializable_conf_snapshot(conf, excluded_keys=process_conf.SNAPSHOT_EXCLUDED_KEYS)
    serialized_round_results = process_conf.serialize_conf_value(round_results)
    serialized_pretrain_result = process_conf.serialize_conf_value(pretrain_result)
    serialized_cloud_history_summary = process_conf.serialize_conf_value(
        training_result.get("cloud_history_summary")
    )
    attack_eval_summary = training_result.get("attack_eval") or {"enabled": False}
    serialized_attack_eval_summary = process_conf.serialize_conf_value(attack_eval_summary)
    early_stop_summary = training_result.get("early_stop") or build_early_stop_state({})
    serialized_early_stop_summary = process_conf.serialize_conf_value(early_stop_summary)
    early_prune_summary = training_result.get("early_prune") or build_early_prune_state({})
    serialized_early_prune_summary = process_conf.serialize_conf_value(early_prune_summary)
    ranking_metrics_valid = bool(avg_validate_loss_list) and all(
        finite_float_or_none(value) is not None for value in avg_validate_loss_list
    )
    assignment_signature = conf.get("batch_assignment_signature")
    assignment = None
    if assignment_signature is not None:
        assignment = {
            "manifest_hash": conf.get("batch_manifest_hash"),
            "assignment_signature": assignment_signature,
            "num_shards": conf.get("batch_num_shards"),
            "shard_weights": process_conf.serialize_conf_value(conf.get("batch_shard_weights")),
            "shard_id": conf.get("batch_shard_id"),
            "shard_weight": conf.get("batch_shard_weight"),
            "shard_mode": conf.get("batch_shard_mode"),
        }

    gpu_name = None
    current_cuda_index = None
    if torch.cuda.is_available():
        current_cuda_index = int(torch.cuda.current_device())
        gpu_name = torch.cuda.get_device_name(current_cuda_index)

    return {
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "batch_id": conf.get("batch_id"),
        "stage_name": conf.get("stage_name"),
        "trial_id": conf.get("trial_id"),
        "run_id": conf.get("run_id"),
        "machine_tag": conf.get("machine_tag"),
        "seed": conf.get("seed"),
        "config_hash_id": conf.get("config_hash_id"),
        "experiment_hash_id": conf.get("experiment_hash_id"),
        "assignment": assignment,
        "use_score_map_flow": bool(training_result.get("use_score_map_flow", False)),
        "device": {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "current_cuda_index": current_cuda_index,
            "gpu_name": gpu_name,
        },
        "output_paths": {
            "txt": os.path.abspath(conf["out_file_path"]),
            "snapshot_json": os.path.abspath(conf["out_conf_snapshot_path"]),
            "summary_json": os.path.abspath(conf["out_summary_path"]),
            "public_output_dir": (training_result.get("public_output") or {}).get("output_dir"),
            "public_artifacts_json": (
                ((training_result.get("public_output") or {}).get("public_files") or {}).get("artifacts")
            ),
        },
        "metrics": {
            "num_rounds": len(round_results),
            "ranking_metrics_valid": ranking_metrics_valid,
            "server_pretrain_time_cost": finite_float_or_none(get_server_pretrain_time_cost(pretrain_result or {})),
            "total_time_cost": finite_float_or_none(training_result.get("all_time_cost", 0.0)),
            "validate_loss_epoch30": finite_float_or_none(validate_loss_epoch30),
            "mean_last5_validate_loss": safe_mean(last_validate_window),
            "mean_last5_train_loss": safe_mean(last_train_window),
            "best_validate_loss": safe_min(avg_validate_loss_list),
            "best_train_loss": safe_min(avg_train_loss_list),
            "final_round_validate_loss": finite_float_or_none(avg_validate_loss_list[-1]) if avg_validate_loss_list else None,
            "final_round_train_loss": finite_float_or_none(avg_train_loss_list[-1]) if avg_train_loss_list else None,
            "early_stop_enabled": bool(early_stop_summary.get("enabled", False)),
            "early_stop_triggered": bool(early_stop_summary.get("triggered", False)),
            "early_stop_stop_epoch": early_stop_summary.get("stop_epoch"),
            "early_stop_best_epoch": early_stop_summary.get("best_epoch"),
            "early_stop_best_validate_loss": finite_float_or_none(early_stop_summary.get("best_validate_loss")),
            "early_stop_rounds_without_min_delta": early_stop_summary.get("rounds_without_min_delta"),
            "early_stop_patience": early_stop_summary.get("patience"),
            "early_stop_min_delta": early_stop_summary.get("min_delta"),
            "early_prune_enabled": bool(early_prune_summary.get("enabled", False)),
            "early_prune_triggered": bool(early_prune_summary.get("triggered", False)),
            "early_prune_stop_epoch": early_prune_summary.get("stop_epoch"),
            "early_prune_check_epoch": early_prune_summary.get("check_epoch"),
            "early_prune_max_validate_loss": finite_float_or_none(early_prune_summary.get("max_validate_loss")),
            "early_prune_observed_validate_loss": finite_float_or_none(early_prune_summary.get("observed_validate_loss")),
            "attack_eval_enabled": bool(attack_eval_summary.get("enabled", False)),
            "attack_eval_mean_asr": finite_float_or_none(attack_eval_summary.get("mean_asr")),
            "attack_eval_mean_mse": finite_float_or_none(attack_eval_summary.get("mean_mse")),
            "attack_eval_mean_rmse": finite_float_or_none(attack_eval_summary.get("mean_rmse")),
            "attack_eval_mean_psnr": finite_float_or_none(attack_eval_summary.get("mean_psnr")),
            "attack_eval_time_cost": finite_float_or_none(attack_eval_summary.get("time_cost")),
            "attack_eval_gradient_source": attack_eval_summary.get("gradient_source"),
            "attack_eval_gradient_match_mode": attack_eval_summary.get("gradient_match_mode"),
        },
        "pretrain_result": serialized_pretrain_result,
        "config_snapshot": config_snapshot,
        "round_results": serialized_round_results,
        "cloud_history_summary": serialized_cloud_history_summary,
        "attack_eval": serialized_attack_eval_summary,
        "early_stop": serialized_early_stop_summary,
        "early_prune": serialized_early_prune_summary,
        "public_output": process_conf.serialize_conf_value(training_result.get("public_output")),
    }


def write_run_summary(summary_path, summary):
    target_path = os.path.abspath(summary_path)
    temp_path = build_atomic_temp_path(target_path)
    ensure_parent_dir(target_path)
    safe_summary = sanitize_json_value(summary)
    try:
        with open(prepare_filesystem_path(temp_path), "w", encoding="utf-8") as f:
            json.dump(safe_summary, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        replace_file_atomically(temp_path, target_path)
    finally:
        if os.path.exists(prepare_filesystem_path(temp_path)):
            os.remove(prepare_filesystem_path(temp_path))


def run_single_experiment(conf_path="conf.json"):
    started_at = get_current_time_iso()
    conf = load_conf(conf_path)
    reset_output_file(conf["out_file_path"])
    write_conf_snapshot(conf["out_conf_snapshot_path"], conf)
    server, clients, pretrain_result = initialize_server_and_clients(conf, conf["out_file_path"])
    training_result = run_training(conf, server, clients, pretrain_result=pretrain_result)
    training_result["public_output"] = write_ours_common_completion_outputs(conf, server, clients, training_result)
    finished_at = get_current_time_iso()
    summary = build_run_summary(conf, pretrain_result, training_result, started_at, finished_at)
    write_run_summary(conf["out_summary_path"], summary)
    return {
        "conf": conf,
        "pretrain_result": pretrain_result,
        "training_result": training_result,
        "summary": summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run one federated matrix completion experiment.")
    parser.add_argument("--conf", default="conf.json", help="Path to the experiment config JSON file.")
    return parser.parse_args()


def main():
    args = parse_args()
    return run_single_experiment(args.conf)


if __name__ == "__main__":
    main()
