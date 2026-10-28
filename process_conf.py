import hashlib
import json
import math
import os
import random
from datetime import datetime

import pandas as pd
import torch



MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_LAPLACE_SENSITIVITY_MULTIPLIER = 2.0


MODEL_DATA_FORM_MAP = {
    "DMF": "integrated",
    "RNN_DMF": "unequal_sliced",
    "TIME_DMF": "unequal_sliced",
    "Mamba_DMF": "unequal_sliced",
}


INDEXED_INPUT_DATASETS = {
    "temperature_use": "location_time",
    "PEMS03_data_use_T": "time_location",
}


DEFAULT_CONF_VALUES = {
    "al_enable": False,
    "al_when": "per_local_epoch",
    "al_budget": 0,
    "al_strategy": "mc_dropout_var",
    "al_mc_passes": 8,
    "al_diversity_lambda": 0.0,
    "dropout_rate": 0.1,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "mamba_z_init": "randn",
    "mamba_residual": False,
    "mamba_residual_scale": 1.0,
    "mamba_layer_norm": False,
    "mamba_head_activation": "elu",
    "al_timing": None,
    "local_strategy": None,
    "global_strategy": "history_proxy_placeholder",
    "fusion_policy": "intersection",
    "fusion_alpha": 0.20,
    "fusion_beta": 0.80,
    "score_norm": "rank",
    "local_candidate_size": None,
    "global_candidate_size": None,
    "final_query_budget": None,
    "cloud_al_enable": False,
    "cloud_history_source": "ground_truth_proxy",
    "cloud_feature_mode": "manual",
    "cloud_similarity": "rbf",
    "cloud_rbf_tau": 0.75,
    "cloud_similarity_chunk_size": 512,
    "cloud_density_radius": 1,
    "cloud_spatial_neighbor_radius": 1,
    "cloud_temporal_neighbor_radius": 1,
    "cloud_anomaly_history_weight": 1.0,
    "cloud_anomaly_spatial_weight": 1.0,
    "cloud_anomaly_temporal_weight": 1.0,
    "cloud_joint_utility_weight": 1.5,
    "cloud_joint_represent_weight": 0.05,
    "cloud_joint_anomaly_weight": 0.001,
    "cloud_joint_redundancy_weight": 0.05,
    "cloud_joint_candidate_source": "rank_union",
    "cloud_joint_candidate_size": None,
    "server_pretrain_enable": True,
    "server_pretrain_epochs": 10,
    "server_pretrain_ratio": 0.5,
    "server_pretrain_split_mode": "time_first_half",
    "server_history_source": "train_data_half_proxy",
    "server_pretrain_optimizer": None,
    "server_pretrain_lr": None,
    "seed": None,
    "batch_id": None,
    "stage_name": None,
    "trial_id": None,
    "run_id": None,
    "machine_tag": None,
    "output_root": "data/output",
    "dp_clip_norm": 1.0,
    "dp_sensitivity_multiplier": FIXED_LAPLACE_SENSITIVITY_MULTIPLIER,
    "attack_eval_enable": False,
    "attack_eval_method": "output_gradient_closed_form",
    "attack_eval_rounds": "last",
    "attack_eval_clients": "all",
    "attack_eval_scope": "observed",
    "attack_eval_asr_threshold": 0.1,
    "attack_eval_data_range": "auto",
    "attack_eval_device": "auto",
    "attack_eval_gradient_steps": 80,
    "attack_eval_gradient_lr": 0.1,
    "attack_eval_gradient_init_scale": 1.0,
    "attack_eval_gradient_source": "single_step_gradient",
    "attack_eval_gradient_match_mode": "raw",
}


OUTPUT_FILENAME_FIELDS = [
    ("m", "model_name"),
    ("t", "task_name"),
    ("sr", "sparse_ratio"),
    ("n", "num_clients"),
    ("p", "privacy"),
    ("eps", "epsilon"),
    ("al", "al_enable"),
    ("cal", "cloud_al_enable"),
    ("at", "al_timing"),
    ("ab", "al_budget"),
    ("ls", "local_strategy"),
    ("gs", "global_strategy"),
    ("fp", "fusion_policy"),
    ("fa", "fusion_alpha"),
    ("fb", "fusion_beta"),
    ("sn", "score_norm"),
    ("lcs", "local_candidate_size"),
    ("gcs", "global_candidate_size"),
    ("fqb", "final_query_budget"),
    ("spr", "server_pretrain_ratio"),
    ("cfm", "cloud_feature_mode"),
    ("csim", "cloud_similarity"),
    ("cjsrc", "cloud_joint_candidate_source"),
    ("cjcs", "cloud_joint_candidate_size"),
    ("cuw", "cloud_joint_utility_weight"),
    ("crw", "cloud_joint_represent_weight"),
    ("caw", "cloud_joint_anomaly_weight"),
    ("cdw", "cloud_joint_redundancy_weight"),
]


SHORT_OUTPUT_FILENAME_FIELDS = [
    ("m", "model_name"),
    ("t", "task_name"),
    ("sr", "sparse_ratio"),
    ("n", "num_clients"),
    ("p", "privacy"),
    ("eps", "epsilon"),
    ("al", "al_enable"),
    ("cal", "cloud_al_enable"),
    ("ab", "al_budget"),
    ("fqb", "final_query_budget"),
    ("lcs", "local_candidate_size"),
    ("gcs", "global_candidate_size"),
    ("cjcs", "cloud_joint_candidate_size"),
    ("spr", "server_pretrain_ratio"),
    ("fa", "fusion_alpha"),
    ("fb", "fusion_beta"),
    ("cuw", "cloud_joint_utility_weight"),
    ("crw", "cloud_joint_represent_weight"),
    ("caw", "cloud_joint_anomaly_weight"),
    ("cdw", "cloud_joint_redundancy_weight"),
]


OPTIONAL_OUTPUT_FILENAME_FIELDS = [
    ("sd", "seed"),
]


SHORT_OPTIONAL_OUTPUT_FILENAME_FIELDS = [
    ("tr", "trial_id"),
    ("sd", "seed"),
]


FILENAME_VALUE_ALIASES = {
    ("task_name", "MyTraffic5"): "MT5",
    ("privacy", "laplace"): "lap",
    ("al_timing", "per_global_round"): "pgr",
    ("al_when", "per_global_round"): "pgr",
    ("al_when", "per_local_epoch"): "ple",
    ("local_strategy", "mc_dropout_var"): "mcdv",
    ("local_strategy", "residual"): "res",
    ("local_strategy", "random"): "rand",
    ("al_strategy", "mc_dropout_var"): "mcdv",
    ("al_strategy", "residual"): "res",
    ("al_strategy", "random"): "rand",
    ("global_strategy", "history_proxy_placeholder"): "hpp",
    ("global_strategy", "global_prediction_abs"): "gpa",
    ("global_strategy", "cloud_joint_cost"): "cjc",
    ("fusion_policy", "intersection"): "isect",
    ("fusion_policy", "weighted_sum"): "ws",
    ("fusion_policy", "cloud_joint_greedy"): "cjg",
    ("score_norm", "minmax"): "mm",
    ("cloud_feature_mode", "manual"): "man",
    ("cloud_feature_mode", "model_embedding"): "emb",
    ("cloud_similarity", "clipped_cosine"): "ccos",
    ("cloud_joint_candidate_source", "rank_union"): "ru",
    ("cloud_joint_candidate_source", "full_missing"): "fm",
}


COMPACT_FILENAME_VALUE_ALIASES = {
    ("model_name", "TIME_DMF"): "TDMF",
    ("model_name", "RNN_DMF"): "RDMF",
    ("model_name", "Mamba_DMF"): "MDMF",
}


SUMMARY_FILENAME_EXTENSION = ".summary.json"
MAX_FILENAME_COMPONENT_LENGTH = 255
MAX_OUTPUT_DIRECTORY_COMPONENT_LENGTH = 96


BATCH_ASSIGNMENT_RUNTIME_KEYS = {
    "batch_manifest_hash",
    "batch_assignment_signature",
    "batch_num_shards",
    "batch_shard_weights",
    "batch_shard_id",
    "batch_shard_weight",
    "batch_shard_mode",
}


HASH_EXCLUDED_KEYS = {
    "config_hash_id",
    "experiment_hash_id",
    "out_file_path",
    "out_conf_snapshot_path",
    "out_summary_path",
    "out_file_stem",
    "output_run_dir",
    "pd_data",
    "timestamp",
    "output_root",
    "batch_id",
    "stage_name",
    "trial_id",
    "run_id",
    "machine_tag",
} | BATCH_ASSIGNMENT_RUNTIME_KEYS


EXPERIMENT_HASH_EXCLUDED_KEYS = HASH_EXCLUDED_KEYS | {
    "seed",
}


SNAPSHOT_EXCLUDED_KEYS = {
    "pd_data",
    "timestamp",
} | BATCH_ASSIGNMENT_RUNTIME_KEYS


def is_plain_int(value):
    return type(value) is int


def is_finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_runtime_path_text(path_value):
    text = str(path_value).strip()
    if os.name != "nt":
        text = text.replace("\\", os.sep)
    return os.path.normpath(text)


def resolve_module_relative_path(path_value):
    normalized_path = normalize_runtime_path_text(path_value)
    if os.path.isabs(normalized_path):
        return os.path.normpath(normalized_path)
    return os.path.normpath(os.path.join(MODULE_DIR, normalized_path))


def del_columns(pd_data, del_slices_ratio):
    if not is_finite_number(del_slices_ratio) or not 0 <= del_slices_ratio <= 1:
        raise ValueError("del_slice_ratio must be between 0 and 1")

    num_slices = pd_data.shape[1]
    del_slices = int(num_slices * del_slices_ratio)
    max_deletable = max(num_slices - 1, 0)
    if del_slices <= 0 or max_deletable == 0:
        return pd_data
    del_slices = min(del_slices, max_deletable)
    index_list = random.sample(range(1, num_slices), del_slices)
    drop_columns = pd_data.columns[index_list]
    pd_data = pd_data.drop(columns=drop_columns)
    return pd_data


def set_default_if_missing(conf, key, default_value):
    if key not in conf:
        conf[key] = default_value


def apply_default_conf_values(conf):
    for key, value in DEFAULT_CONF_VALUES.items():
        set_default_if_missing(conf, key, value)


def infer_data_form(conf):
    model_name = conf["model_name"]
    return MODEL_DATA_FORM_MAP.get(model_name, conf.get("data_form"))


def sync_future_active_learning_aliases(conf):
    if conf.get("al_timing") is None:
        conf["al_timing"] = conf["al_when"]
    if conf.get("local_strategy") is None:
        conf["local_strategy"] = conf["al_strategy"]
    if conf.get("final_query_budget") is None:
        conf["final_query_budget"] = conf["al_budget"]
    if conf.get("local_candidate_size") is None:
        conf["local_candidate_size"] = conf["al_budget"]
    if conf.get("global_candidate_size") is None:
        conf["global_candidate_size"] = conf["final_query_budget"]


def get_effective_local_active_learning_strategy(conf):
    if not conf.get("al_enable", False):
        return None
    if conf.get("cloud_al_enable", False):
        return conf.get("local_strategy") or conf.get("al_strategy")
    return conf.get("al_strategy")


def validate_privacy_conf(conf):
    privacy = conf.get("privacy")
    if privacy not in {"no", "clip", "laplace"}:
        raise ValueError("privacy must be 'no', 'clip', or 'laplace'")

    if privacy == "no":
        conf["dp_sensitivity_multiplier"] = FIXED_LAPLACE_SENSITIVITY_MULTIPLIER
        return

    if privacy == "clip":
        dp_clip_norm = conf.get("dp_clip_norm")
        if not is_finite_number(dp_clip_norm) or dp_clip_norm <= 0:
            raise ValueError("dp_clip_norm must be a positive number when privacy='clip'")
        conf["dp_sensitivity_multiplier"] = FIXED_LAPLACE_SENSITIVITY_MULTIPLIER
        return

    epsilon = conf.get("epsilon")
    if not is_finite_number(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive number when privacy='laplace'")

    dp_clip_norm = conf.get("dp_clip_norm")
    if not is_finite_number(dp_clip_norm) or dp_clip_norm <= 0:
        raise ValueError("dp_clip_norm must be a positive number when privacy='laplace'")

    dp_sensitivity_multiplier = conf.get("dp_sensitivity_multiplier")
    if not is_finite_number(dp_sensitivity_multiplier) or dp_sensitivity_multiplier <= 0:
        raise ValueError("dp_sensitivity_multiplier must be a positive number when privacy='laplace'")
    conf["dp_sensitivity_multiplier"] = FIXED_LAPLACE_SENSITIVITY_MULTIPLIER


def validate_attack_eval_conf(conf):
    if not isinstance(conf.get("attack_eval_enable"), bool):
        raise ValueError("attack_eval_enable must be a boolean")

    method = conf.get("attack_eval_method")
    if method not in {"gradient_matching", "update_reconstruction", "uploaded_model_prediction", "output_gradient_closed_form"}:
        raise ValueError("attack_eval_method must be 'gradient_matching', 'update_reconstruction', 'uploaded_model_prediction', or 'output_gradient_closed_form'")

    gradient_source = conf.get("attack_eval_gradient_source")
    if gradient_source not in {"single_step_gradient", "uploaded_update_proxy"}:
        raise ValueError("attack_eval_gradient_source must be 'single_step_gradient' or 'uploaded_update_proxy'")

    gradient_match_mode = conf.get("attack_eval_gradient_match_mode")
    if gradient_match_mode not in {"raw", "direction"}:
        raise ValueError("attack_eval_gradient_match_mode must be 'raw' or 'direction'")

    rounds = conf.get("attack_eval_rounds")
    global_epochs = conf.get("global_epochs")
    if rounds != "last":
        if is_plain_int(rounds) and rounds >= 0:
            round_values = [rounds]
        elif isinstance(rounds, list) and all(is_plain_int(item) and item >= 0 for item in rounds):
            round_values = rounds
        else:
            raise ValueError("attack_eval_rounds must be 'last', a non-negative integer, or a list of non-negative integers")
        if is_plain_int(global_epochs) and any(item >= global_epochs for item in round_values):
            raise ValueError("attack_eval_rounds contains a round index outside [0, global_epochs)")

    clients = conf.get("attack_eval_clients")
    if clients != "all":
        if not isinstance(clients, list) or not all(is_plain_int(item) and item >= 0 for item in clients):
            raise ValueError("attack_eval_clients must be 'all' or a list of non-negative 0-based client ids")
        num_clients = conf.get("num_clients")
        if is_plain_int(num_clients) and any(item >= num_clients for item in clients):
            raise ValueError("attack_eval_clients contains a client id outside [0, num_clients)")

    scope = conf.get("attack_eval_scope")
    if scope not in {"all", "missing", "observed"}:
        raise ValueError("attack_eval_scope must be 'all', 'missing', or 'observed'")
    if conf.get("attack_eval_enable", False) and method == "update_reconstruction" and scope != "missing":
        raise ValueError("attack_eval_method='update_reconstruction' requires attack_eval_scope='missing'")

    asr_threshold = conf.get("attack_eval_asr_threshold")
    if not is_finite_number(asr_threshold) or asr_threshold < 0:
        raise ValueError("attack_eval_asr_threshold must be a non-negative finite number")

    data_range = conf.get("attack_eval_data_range")
    if isinstance(data_range, str):
        data_range = data_range.strip().lower()
        conf["attack_eval_data_range"] = data_range
    if data_range != "auto":
        if not is_finite_number(data_range) or data_range <= 0:
            raise ValueError("attack_eval_data_range must be 'auto' or a positive finite number")

    gradient_steps = conf.get("attack_eval_gradient_steps")
    if not is_plain_int(gradient_steps) or gradient_steps <= 0:
        raise ValueError("attack_eval_gradient_steps must be a positive integer")

    gradient_lr = conf.get("attack_eval_gradient_lr")
    if not is_finite_number(gradient_lr) or gradient_lr <= 0:
        raise ValueError("attack_eval_gradient_lr must be a positive finite number")

    gradient_init_scale = conf.get("attack_eval_gradient_init_scale")
    if not is_finite_number(gradient_init_scale) or gradient_init_scale <= 0:
        raise ValueError("attack_eval_gradient_init_scale must be a positive finite number")

    device_name = conf.get("attack_eval_device")
    if not isinstance(device_name, str):
        raise ValueError("attack_eval_device must be a string")
    device_name = device_name.strip().lower()
    if device_name not in {"auto", "cpu", "cuda"} and not device_name.startswith("cuda:"):
        raise ValueError("attack_eval_device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    if device_name.startswith("cuda:"):
        device_index_text = device_name.split(":", 1)[1]
        if not device_index_text.isdigit():
            raise ValueError("attack_eval_device cuda index must be a non-negative integer")
        device_index = int(device_index_text)
        if conf.get("attack_eval_enable", False) and torch.cuda.is_available() and device_index >= torch.cuda.device_count():
            raise ValueError("attack_eval_device cuda index is outside the visible CUDA device count")
    if conf.get("attack_eval_enable", False) and device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("attack_eval_device requires CUDA, but torch.cuda.is_available() is False")
    conf["attack_eval_device"] = device_name


def validate_attack_eval_conf(conf):
    if not isinstance(conf.get("attack_eval_enable"), bool):
        raise ValueError("attack_eval_enable must be a boolean")

    method = conf.get("attack_eval_method")
    if method not in {"gradient_matching", "update_reconstruction", "uploaded_model_prediction", "output_gradient_closed_form"}:
        raise ValueError("attack_eval_method must be 'gradient_matching', 'update_reconstruction', 'uploaded_model_prediction', or 'output_gradient_closed_form'")

    gradient_source = conf.get("attack_eval_gradient_source")
    if gradient_source not in {"single_step_gradient", "uploaded_update_proxy"}:
        raise ValueError("attack_eval_gradient_source must be 'single_step_gradient' or 'uploaded_update_proxy'")

    gradient_match_mode = conf.get("attack_eval_gradient_match_mode")
    if gradient_match_mode not in {"raw", "direction"}:
        raise ValueError("attack_eval_gradient_match_mode must be 'raw' or 'direction'")

    rounds = conf.get("attack_eval_rounds")
    global_epochs = conf.get("global_epochs")
    if rounds != "last":
        if is_plain_int(rounds) and rounds >= 0:
            round_values = [rounds]
        elif isinstance(rounds, list) and all(is_plain_int(item) and item >= 0 for item in rounds):
            round_values = rounds
        else:
            raise ValueError("attack_eval_rounds must be 'last', a non-negative integer, or a list of non-negative integers")
        if is_plain_int(global_epochs) and any(item >= global_epochs for item in round_values):
            raise ValueError("attack_eval_rounds contains a round index outside [0, global_epochs)")

    clients = conf.get("attack_eval_clients")
    if clients != "all":
        if not isinstance(clients, list) or not all(is_plain_int(item) and item >= 0 for item in clients):
            raise ValueError("attack_eval_clients must be 'all' or a list of non-negative 0-based client ids")
        num_clients = conf.get("num_clients")
        if is_plain_int(num_clients) and any(item >= num_clients for item in clients):
            raise ValueError("attack_eval_clients contains a client id outside [0, num_clients)")

    scope = conf.get("attack_eval_scope")
    if scope not in {"all", "missing", "observed"}:
        raise ValueError("attack_eval_scope must be 'all', 'missing', or 'observed'")
    if conf.get("attack_eval_enable", False) and method == "update_reconstruction" and scope != "missing":
        raise ValueError("attack_eval_method='update_reconstruction' requires attack_eval_scope='missing'")

    asr_threshold = conf.get("attack_eval_asr_threshold")
    if not is_finite_number(asr_threshold) or asr_threshold < 0:
        raise ValueError("attack_eval_asr_threshold must be a non-negative finite number")

    data_range = conf.get("attack_eval_data_range")
    if isinstance(data_range, str):
        data_range = data_range.strip().lower()
        conf["attack_eval_data_range"] = data_range
    if data_range != "auto":
        if not is_finite_number(data_range) or data_range <= 0:
            raise ValueError("attack_eval_data_range must be 'auto' or a positive finite number")

    gradient_steps = conf.get("attack_eval_gradient_steps")
    if not is_plain_int(gradient_steps) or gradient_steps <= 0:
        raise ValueError("attack_eval_gradient_steps must be a positive integer")

    gradient_lr = conf.get("attack_eval_gradient_lr")
    if not is_finite_number(gradient_lr) or gradient_lr <= 0:
        raise ValueError("attack_eval_gradient_lr must be a positive finite number")

    gradient_init_scale = conf.get("attack_eval_gradient_init_scale")
    if not is_finite_number(gradient_init_scale) or gradient_init_scale <= 0:
        raise ValueError("attack_eval_gradient_init_scale must be a positive finite number")

    device_name = conf.get("attack_eval_device")
    if not isinstance(device_name, str):
        raise ValueError("attack_eval_device must be a string")
    device_name = device_name.strip().lower()
    if device_name not in {"auto", "cpu", "cuda"} and not device_name.startswith("cuda:"):
        raise ValueError("attack_eval_device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    if device_name.startswith("cuda:"):
        device_index_text = device_name.split(":", 1)[1]
        if not device_index_text.isdigit():
            raise ValueError("attack_eval_device cuda index must be a non-negative integer")
        device_index = int(device_index_text)
        if conf.get("attack_eval_enable", False) and torch.cuda.is_available() and device_index >= torch.cuda.device_count():
            raise ValueError("attack_eval_device cuda index is outside the visible CUDA device count")
    if conf.get("attack_eval_enable", False) and device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("attack_eval_device requires CUDA, but torch.cuda.is_available() is False")
    conf["attack_eval_device"] = device_name


def validate_active_learning_conf(conf):
    active_local_strategy = get_effective_local_active_learning_strategy(conf)
    al_when = conf.get("al_when")
    al_timing = conf.get("al_timing")
    if active_local_strategy is not None and active_local_strategy not in {"mc_dropout_var", "residual", "random"}:
        raise ValueError("local active learning strategy must be 'mc_dropout_var', 'residual', or 'random'")

    if al_when not in {"per_local_epoch", "per_global_round"}:
        raise ValueError("al_when must be 'per_local_epoch' or 'per_global_round'")
    if al_timing not in {"per_local_epoch", "per_global_round"}:
        raise ValueError("al_timing must be 'per_local_epoch' or 'per_global_round'")

    if conf.get("al_enable", False) and al_timing == "per_global_round":
        final_query_budget = conf.get("final_query_budget")
        if not is_plain_int(final_query_budget):
            raise ValueError("final_query_budget must be an integer when al_timing='per_global_round'")
        if final_query_budget < 0:
            raise ValueError("final_query_budget must be non-negative when al_timing='per_global_round'")

    if active_local_strategy == "mc_dropout_var":
        mc_passes = conf.get("al_mc_passes")
        if not is_plain_int(mc_passes):
            raise ValueError("al_mc_passes must be an integer when mc_dropout_var is enabled")
        if mc_passes < 2:
            raise ValueError("mc_dropout_var requires al_mc_passes >= 2 when active learning is enabled")


def validate_cloud_active_learning_conf(conf):
    if not conf.get("cloud_al_enable", False):
        return

    if not conf.get("al_enable", False):
        raise ValueError("cloud_al_enable=True requires al_enable=True")

    fusion_policy = conf.get("fusion_policy")
    if fusion_policy not in {"intersection", "weighted_sum", "cloud_joint_greedy"}:
        raise ValueError("fusion_policy must be 'intersection', 'weighted_sum', or 'cloud_joint_greedy'")

    final_query_budget = conf.get("final_query_budget")
    if not is_plain_int(final_query_budget):
        raise ValueError("final_query_budget must be an integer when cloud_al_enable=True")
    if final_query_budget < 0:
        raise ValueError("final_query_budget must be non-negative when cloud_al_enable=True")

    for key in ("local_candidate_size", "global_candidate_size"):
        value = conf.get(key)
        if not is_plain_int(value):
            raise ValueError(f"{key} must be an integer when cloud_al_enable=True")
        if value < 0:
            raise ValueError(f"{key} must be non-negative when cloud_al_enable=True")

    global_strategy = conf.get("global_strategy")
    if global_strategy not in {"history_proxy_placeholder", "global_prediction_abs", "cloud_joint_cost"}:
        raise ValueError("global_strategy must be 'history_proxy_placeholder', 'global_prediction_abs', or 'cloud_joint_cost'")
    if fusion_policy == "cloud_joint_greedy" and global_strategy != "cloud_joint_cost":
        raise ValueError("fusion_policy='cloud_joint_greedy' requires global_strategy='cloud_joint_cost'")

    if fusion_policy in {"weighted_sum", "cloud_joint_greedy"}:
        fusion_alpha = conf.get("fusion_alpha")
        fusion_beta = conf.get("fusion_beta")
        if not is_finite_number(fusion_alpha):
            raise ValueError("fusion_alpha must be numeric when fusion_policy uses score weighting")
        if not is_finite_number(fusion_beta):
            raise ValueError("fusion_beta must be numeric when fusion_policy uses score weighting")
        if fusion_alpha < 0 or fusion_beta < 0:
            raise ValueError("fusion_alpha and fusion_beta must be non-negative when fusion_policy uses score weighting")
        if fusion_alpha + fusion_beta <= 0:
            raise ValueError("fusion_alpha + fusion_beta must be positive when fusion_policy uses score weighting")
        if conf.get("score_norm") not in {"none", "minmax", "zscore", "rank"}:
            raise ValueError("score_norm must be 'none', 'minmax', 'zscore', or 'rank'")

    if global_strategy == "cloud_joint_cost":
        if conf.get("cloud_feature_mode") not in {"manual", "model_embedding"}:
            raise ValueError("cloud_feature_mode must be 'manual' or 'model_embedding'")
        if conf.get("cloud_joint_candidate_source") not in {"rank_union", "full_missing"}:
            raise ValueError("cloud_joint_candidate_source must be 'rank_union' or 'full_missing'")
        cloud_joint_candidate_size = conf.get("cloud_joint_candidate_size")
        if cloud_joint_candidate_size is not None:
            if not is_plain_int(cloud_joint_candidate_size) or cloud_joint_candidate_size <= 0:
                raise ValueError("cloud_joint_candidate_size must be a positive integer or None")
        if conf.get("cloud_similarity") not in {"rbf", "clipped_cosine"}:
            raise ValueError("cloud_similarity must be 'rbf' or 'clipped_cosine'")
        if not is_finite_number(conf.get("cloud_rbf_tau")) or conf.get("cloud_rbf_tau") <= 0:
            raise ValueError("cloud_rbf_tau must be positive")
        if not is_plain_int(conf.get("cloud_similarity_chunk_size")) or conf.get("cloud_similarity_chunk_size") <= 0:
            raise ValueError("cloud_similarity_chunk_size must be a positive integer")
        for key in ("cloud_density_radius", "cloud_spatial_neighbor_radius", "cloud_temporal_neighbor_radius"):
            value = conf.get(key)
            if not is_plain_int(value) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        for key in ("cloud_anomaly_history_weight", "cloud_anomaly_spatial_weight", "cloud_anomaly_temporal_weight"):
            value = conf.get(key)
            if not is_finite_number(value) or value < 0:
                raise ValueError(f"{key} must be a non-negative number")
        anomaly_weight_sum = (
            conf.get("cloud_anomaly_history_weight") +
            conf.get("cloud_anomaly_spatial_weight") +
            conf.get("cloud_anomaly_temporal_weight")
        )
        if anomaly_weight_sum <= 0:
            raise ValueError("at least one cloud anomaly component weight must be positive")
        for key in (
                "cloud_joint_utility_weight",
                "cloud_joint_represent_weight",
                "cloud_joint_anomaly_weight",
                "cloud_joint_redundancy_weight"):
            value = conf.get(key)
            if not is_finite_number(value) or value < 0:
                raise ValueError(f"{key} must be a non-negative number")
        joint_weight_sum = (
            conf.get("cloud_joint_utility_weight") +
            conf.get("cloud_joint_represent_weight") +
            conf.get("cloud_joint_anomaly_weight") +
            conf.get("cloud_joint_redundancy_weight")
        )
        if joint_weight_sum <= 0:
            raise ValueError("at least one cloud joint cost weight must be positive")


def validate_server_pretrain_conf(conf):
    server_pretrain_ratio = conf.get("server_pretrain_ratio", 0.5)
    if not is_finite_number(server_pretrain_ratio):
        raise ValueError("server_pretrain_ratio must be a finite number between 0 and 1")
    if not 0 <= server_pretrain_ratio <= 1:
        raise ValueError("server_pretrain_ratio must be between 0 and 1")

    server_pretrain_epochs = conf.get("server_pretrain_epochs", 0)
    if not is_plain_int(server_pretrain_epochs):
        raise ValueError("server_pretrain_epochs must be an integer")
    if server_pretrain_epochs < 0:
        raise ValueError("server_pretrain_epochs must be non-negative")

    split_mode = conf.get("server_pretrain_split_mode", "time_first_half")
    if split_mode not in {"time_first_half"}:
        raise ValueError("server_pretrain_split_mode must be 'time_first_half'")

    server_pretrain_optimizer = conf.get("server_pretrain_optimizer")
    if server_pretrain_optimizer not in {None, "Rprop"}:
        raise ValueError("server_pretrain_optimizer must be None or 'Rprop'")

    server_pretrain_lr = conf.get("server_pretrain_lr")
    if server_pretrain_lr is not None:
        if not is_finite_number(server_pretrain_lr) or server_pretrain_lr <= 0:
            raise ValueError("server_pretrain_lr must be a finite positive number when provided")


def format_compact_float_for_filename(value):
    if not math.isfinite(value):
        return str(value)
    if float(int(value)) == float(value):
        return str(int(value))
    compact_text = format(value, ".6g")
    return "0" if compact_text == "-0" else compact_text


def normalize_filename_value(value, conf_key=None, compact=False):
    if compact and (conf_key, value) in COMPACT_FILENAME_VALUE_ALIASES:
        text = COMPACT_FILENAME_VALUE_ALIASES[(conf_key, value)]
    elif (conf_key, value) in FILENAME_VALUE_ALIASES:
        text = FILENAME_VALUE_ALIASES[(conf_key, value)]
    elif value is None:
        text = "na"
    elif isinstance(value, bool):
        text = ("1" if value else "0") if compact else str(value).lower()
    elif isinstance(value, float):
        text = format_compact_float_for_filename(value) if compact else str(value)
    else:
        text = str(value)
    for invalid_char in '<>:"/\\|?*':
        text = text.replace(invalid_char, "-")
    return text


def serialize_conf_value(value):
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return None
    if isinstance(value, dict):
        return {k: serialize_conf_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_conf_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_serializable_conf_snapshot(conf, excluded_keys=None):
    excluded_keys = set() if excluded_keys is None else set(excluded_keys)
    snapshot = {}
    for key in sorted(conf.keys()):
        if key in excluded_keys:
            continue
        snapshot[key] = serialize_conf_value(conf[key])
    return snapshot


def clear_derived_output_path_fields(conf):
    derived_output_keys = (
        "out_file_path",
        "out_conf_snapshot_path",
        "out_summary_path",
        "out_file_stem",
        "output_run_dir",
        "config_hash_id",
        "experiment_hash_id",
    )
    for key in derived_output_keys:
        conf.pop(key, None)


def validate_batch_execution_conf(conf):
    seed = conf.get("seed")
    if seed is not None and not is_plain_int(seed):
        raise ValueError("seed must be an integer or None")

    for key in ("batch_id", "stage_name", "trial_id", "run_id", "machine_tag"):
        value = conf.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be a string or None")

    output_root = conf.get("output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError("output_root must be a non-empty string")
    conf["output_root"] = normalize_runtime_path_text(output_root)


def build_config_hash_id(conf):
    hash_snapshot = build_serializable_conf_snapshot(conf, excluded_keys=HASH_EXCLUDED_KEYS)
    normalized_text = json.dumps(hash_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:8]


def build_experiment_hash_id(conf):
    experiment_snapshot = build_serializable_conf_snapshot(conf, excluded_keys=EXPERIMENT_HASH_EXCLUDED_KEYS)
    normalized_text = json.dumps(experiment_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:8]


def build_output_file_stem(conf):
    def build_stem_with_mode(compact):
        parts = []
        for output_key, conf_key in OUTPUT_FILENAME_FIELDS:
            value_text = normalize_filename_value(conf.get(conf_key), conf_key, compact=compact)
            parts.append(f"{output_key}={value_text}")
        for output_key, conf_key in OPTIONAL_OUTPUT_FILENAME_FIELDS:
            if conf.get(conf_key) is not None:
                value_text = normalize_filename_value(conf.get(conf_key), conf_key, compact=compact)
                parts.append(f"{output_key}={value_text}")
        parts.append(f"id={build_config_hash_id(conf)}")
        return "__".join(parts)

    normal_stem = build_stem_with_mode(compact=False)
    if len(normal_stem + SUMMARY_FILENAME_EXTENSION) <= MAX_FILENAME_COMPONENT_LENGTH:
        return normal_stem

    compact_stem = build_stem_with_mode(compact=True)
    if len(compact_stem + SUMMARY_FILENAME_EXTENSION) <= MAX_FILENAME_COMPONENT_LENGTH:
        return compact_stem

    short_parts = []
    for output_key, conf_key in SHORT_OUTPUT_FILENAME_FIELDS:
        value_text = normalize_filename_value(conf.get(conf_key), conf_key, compact=True)
        short_parts.append(f"{output_key}={value_text}")
    for output_key, conf_key in SHORT_OPTIONAL_OUTPUT_FILENAME_FIELDS:
        if conf.get(conf_key) is not None:
            value_text = normalize_filename_value(conf.get(conf_key), conf_key, compact=True)
            short_parts.append(f"{output_key}={value_text}")
    short_parts.append(f"id={build_config_hash_id(conf)}")
    short_stem = "__".join(short_parts)
    if len(short_stem + SUMMARY_FILENAME_EXTENSION) <= MAX_FILENAME_COMPONENT_LENGTH:
        return short_stem

    hash_only_stem = f"tr={normalize_filename_value(conf.get('trial_id'), 'trial_id', compact=True)}__id={build_config_hash_id(conf)}"
    if len(hash_only_stem + SUMMARY_FILENAME_EXTENSION) <= MAX_FILENAME_COMPONENT_LENGTH:
        return hash_only_stem

    raise ValueError(
        "output filename stem is still too long after compact normalization; reduce filename fields or output root depth"
    )


def shorten_output_directory_component(text, max_length=MAX_OUTPUT_DIRECTORY_COMPONENT_LENGTH):
    text = str(text)
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[: max_length - 10]}__{digest}"


def build_output_paths(conf):
    output_root = resolve_module_relative_path(conf.get("output_root", DEFAULT_CONF_VALUES["output_root"]))
    out_file_stem = build_output_file_stem(conf)
    data_dir_name = normalize_filename_value(conf.get("task_name"), compact=False)
    timestamp_dir_name = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    config_dir_name = shorten_output_directory_component(out_file_stem)
    output_run_dir = os.path.join(output_root, data_dir_name, config_dir_name, timestamp_dir_name)
    out_txt_path = os.path.join(output_run_dir, "run.txt")
    out_json_path = os.path.join(output_run_dir, "snapshot.json")
    out_summary_path = os.path.join(output_run_dir, "run.summary.json")
    return out_file_stem, output_run_dir, out_txt_path, out_json_path, out_summary_path


def load_input_dataframe(conf):
    csv_path = resolve_module_relative_path("data/input/{}.csv".format(conf["task_name"]))
    pd_data = read_input_csv_by_task(csv_path, conf["task_name"])
    if conf["del_slice_ratio"] != 0:
        pd_data = del_columns(pd_data, conf["del_slice_ratio"])
    return pd_data


def read_input_csv_by_task(csv_path, task_name):
    indexed_layout = INDEXED_INPUT_DATASETS.get(task_name)
    if indexed_layout is None:
        return pd.read_csv(csv_path, header=None)

    raw_data = pd.read_csv(csv_path, header=0, index_col=0)
    if indexed_layout == "time_location":
        time_labels = raw_data.index
        pd_data = raw_data.T
    elif indexed_layout == "location_time":
        time_labels = raw_data.columns
        pd_data = raw_data
    else:
        raise ValueError("Unsupported indexed input layout: {}".format(indexed_layout))

    pd_data = pd_data.apply(pd.to_numeric, errors="coerce")
    non_empty_time_mask = ~pd_data.isna().all(axis=0)
    pd_data = pd_data.loc[:, non_empty_time_mask]
    time_labels = pd.Index(time_labels)[non_empty_time_mask.to_numpy()]

    remaining_nan_count = int(pd_data.isna().sum().sum())
    if remaining_nan_count != 0:
        raise ValueError("{} contains {} missing values after dropping empty time columns".format(task_name, remaining_nan_count))

    pd_data = pd_data.astype("float32")
    pd_data.columns = build_numeric_time_axis(time_labels)
    return pd_data


def build_numeric_time_axis(time_labels):
    parsed_time = pd.to_datetime(time_labels, errors="coerce")
    if parsed_time.isna().any():
        return pd.RangeIndex(start=0, stop=len(time_labels), step=1).to_numpy(dtype="float32")
    elapsed_hours = (parsed_time - parsed_time[0]).total_seconds() / 3600.0
    return elapsed_hours.astype("float32")


def attach_timestamp_if_needed(conf, pd_data):
    if conf["data_form"] == "unequal_sliced":
        conf["timestamp"] = torch.from_numpy(pd_data.columns.values)


def attach_data_shape_fields(conf, pd_data):
    conf["num_locations"] = pd_data.shape[0]
    conf["num_slices"] = pd_data.shape[1]
    conf["num_steps"] = conf["num_slices"]
    conf["pd_data"] = pd_data


def process(conf):
    process_reserved(conf)


def process_reserved(conf):
    clear_derived_output_path_fields(conf)
    apply_default_conf_values(conf)
    conf["data_form"] = infer_data_form(conf)
    sync_future_active_learning_aliases(conf)
    validate_batch_execution_conf(conf)
    validate_privacy_conf(conf)
    validate_attack_eval_conf(conf)
    validate_active_learning_conf(conf)
    validate_server_pretrain_conf(conf)
    validate_cloud_active_learning_conf(conf)
    out_file_stem, output_run_dir, out_txt_path, out_json_path, out_summary_path = build_output_paths(conf)
    conf["out_file_stem"] = out_file_stem
    conf["output_run_dir"] = output_run_dir
    conf["config_hash_id"] = build_config_hash_id(conf)
    conf["experiment_hash_id"] = build_experiment_hash_id(conf)
    conf["out_file_path"] = out_txt_path
    conf["out_conf_snapshot_path"] = out_json_path
    conf["out_summary_path"] = out_summary_path
    pd_data = load_input_dataframe(conf)
    attach_timestamp_if_needed(conf, pd_data)
    attach_data_shape_fields(conf, pd_data)
