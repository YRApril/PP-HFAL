from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "completion-output-v1"
MAX_PATH_COMPONENT_LENGTH = 96
_INVALID_PATH_COMPONENT_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WHITESPACE = re.compile(r"\s+")

COMMON_METRIC_KEYS = {
    "missing_rmse",
    "missing_mae",
    "missing_mse",
    "missing_count",
    "observed_rmse",
    "observed_mae",
    "observed_mse",
    "observed_count",
    "all_rmse",
    "all_mae",
    "all_mse",
    "all_count",
    "relative_frobenius",
    "total_time_sec",
    "train_loss",
    "validate_loss",
    "best_epoch",
    "num_rounds",
    "num_iterations",
    "stop_reason",
}


def sanitize_path_component(value: Any, *, default: str = "run", max_length: int = MAX_PATH_COMPONENT_LENGTH) -> str:
    if value is None:
        text = default
    else:
        text = str(value)
    text = _INVALID_PATH_COMPONENT_CHARS.sub("-", text)
    text = _WHITESPACE.sub("_", text.strip())
    text = text.strip(" ._")
    if not text:
        text = default
    if len(text) > max_length:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        text = f"{text[: max_length - 10]}__{digest}"
    return text


def completion_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")


def build_completion_config_name(*parts: Any, max_length: int = MAX_PATH_COMPONENT_LENGTH, **fields: Any) -> str:
    tokens: list[str] = []
    for part in parts:
        if part is not None and str(part) != "":
            tokens.append(str(part))
    for key, value in fields.items():
        if value is None:
            continue
        tokens.append(f"{key}={value}")
    return sanitize_path_component("__".join(tokens), default="config", max_length=max_length)


def make_completion_run_dir(
    output_root: Path | str,
    data_name: Any,
    config_name: Any,
    *,
    timestamp: str | None = None,
) -> Path:
    parent = (
        Path(output_root).resolve()
        / sanitize_path_component(data_name, default="data", max_length=80)
        / sanitize_path_component(config_name, default="config", max_length=MAX_PATH_COMPONENT_LENGTH)
    )
    time_component = sanitize_path_component(timestamp or completion_timestamp(), default="time", max_length=80)
    for suffix in ["", *[f"_{idx:02d}" for idx in range(1, 100)]]:
        run_dir = parent / f"{time_component}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique output directory under {parent}")


def prepare_filesystem_path(path_value: Path | str) -> str:
    normalized_path = os.path.normpath(os.path.abspath(os.fspath(path_value)))
    if os.name != "nt":
        return normalized_path
    if normalized_path.startswith("\\\\?\\"):
        return normalized_path
    if normalized_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + normalized_path.lstrip("\\")
    if len(normalized_path) >= 240:
        return "\\\\?\\" + normalized_path
    return normalized_path


def ensure_dir(path_value: Path | str) -> None:
    os.makedirs(prepare_filesystem_path(path_value), exist_ok=True)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if hasattr(value, "detach") and callable(value.detach):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(prepare_filesystem_path(path), "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def normalize_history_row(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        row = asdict(row)
    elif not isinstance(row, dict):
        row = {"value": row}
    return {str(key): json_safe(value) for key, value in row.items()}


def write_history_csv(path: Path, history: list[Any] | tuple[Any, ...] | None) -> None:
    ensure_dir(path.parent)
    rows = [normalize_history_row(row) for row in (history or [])]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(prepare_filesystem_path(path), "w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})


def csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)
    return value


def masked_regression_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = mask.astype(bool) & np.isfinite(prediction) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return {"count": 0, "mse": None, "rmse": None, "mae": None}
    diff = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    mse = float(np.mean(diff * diff))
    return {
        "count": count,
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
    }


def evaluate_completion_arrays(
    full_matrix: Any,
    observed_mask: Any,
    completed: Any,
) -> dict[str, Any]:
    full = to_numpy(full_matrix).astype(np.float64, copy=False)
    observed = to_numpy(observed_mask).astype(bool, copy=False)
    comp = to_numpy(completed).astype(np.float64, copy=False)
    if full.shape != comp.shape or observed.shape != full.shape:
        return {
            "metric_error": "shape_mismatch",
            "full_shape": list(full.shape),
            "observed_mask_shape": list(observed.shape),
            "completed_shape": list(comp.shape),
        }

    missing = ~observed
    all_valid = np.ones_like(observed, dtype=bool)
    missing_metrics = masked_regression_metrics(comp, full, missing)
    observed_metrics = masked_regression_metrics(comp, full, observed)
    all_metrics = masked_regression_metrics(comp, full, all_valid)

    finite = np.isfinite(comp) & np.isfinite(full)
    if int(finite.sum()) == 0:
        relative_frobenius = None
    else:
        full_values = full[finite].astype(np.float64, copy=False)
        diff_values = (comp - full)[finite].astype(np.float64, copy=False)
        denom = math.sqrt(float(np.sum(full_values * full_values)))
        numerator = math.sqrt(float(np.sum(diff_values * diff_values)))
        relative_frobenius = None if denom <= 1e-12 else float(numerator / denom)

    return {
        "missing_count": missing_metrics["count"],
        "missing_mse": missing_metrics["mse"],
        "missing_rmse": missing_metrics["rmse"],
        "missing_mae": missing_metrics["mae"],
        "observed_count": observed_metrics["count"],
        "observed_mse": observed_metrics["mse"],
        "observed_rmse": observed_metrics["rmse"],
        "observed_mae": observed_metrics["mae"],
        "all_count": all_metrics["count"],
        "all_mse": all_metrics["mse"],
        "all_rmse": all_metrics["rmse"],
        "all_mae": all_metrics["mae"],
        "relative_frobenius": relative_frobenius,
    }


def extract_common_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {key: metrics[key] for key in COMMON_METRIC_KEYS if key in metrics}


def save_array(path: Path, value: Any, *, mask: bool = False) -> None:
    ensure_dir(path.parent)
    array = to_numpy(value)
    if mask:
        array = array.astype(np.uint8)
    np.save(prepare_filesystem_path(path), array)


def save_common_completion_outputs(
    output_dir: Path | str,
    *,
    algorithm: str,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    history: list[Any] | tuple[Any, ...] | None = None,
    full_matrix: Any | None = None,
    observed_mask: Any | None = None,
    raw_prediction: Any | None = None,
    completed: Any | None = None,
    per_client_metrics: Any | None = None,
    legacy_artifacts: dict[str, Any] | None = None,
    notes: dict[str, Any] | None = None,
    prefer_provided_metrics: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    public_files: dict[str, str] = {}
    config_path = out_dir / "config.json"
    config_payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "algorithm": algorithm}
    if config:
        config_payload.update(config)
    write_json(config_path, config_payload)
    public_files["config"] = str(config_path)

    if history is not None:
        history_path = out_dir / "history.csv"
        write_history_csv(history_path, history)
        public_files["history"] = str(history_path)

    if full_matrix is not None:
        full_path = out_dir / "full_matrix.npy"
        save_array(full_path, full_matrix)
        public_files["full_matrix"] = str(full_path)
    if observed_mask is not None:
        mask_path = out_dir / "observed_mask.npy"
        save_array(mask_path, observed_mask, mask=True)
        public_files["observed_mask"] = str(mask_path)
    if raw_prediction is not None:
        raw_path = out_dir / "raw_prediction.npy"
        save_array(raw_path, raw_prediction)
        public_files["raw_prediction"] = str(raw_path)
    if completed is not None:
        completed_path = out_dir / "completed.npy"
        save_array(completed_path, completed)
        public_files["completed"] = str(completed_path)

    per_client_path = None
    if per_client_metrics is not None:
        per_client_path = out_dir / "per_client_metrics.json"
        write_json(per_client_path, {"schema_version": SCHEMA_VERSION, "algorithm": algorithm, "per_client": per_client_metrics})
        public_files["per_client_metrics"] = str(per_client_path)

    computed_metrics = None
    if full_matrix is not None and observed_mask is not None and completed is not None:
        computed_metrics = evaluate_completion_arrays(full_matrix, observed_mask, completed)

    provided_common = extract_common_metrics(metrics)
    common_metrics: dict[str, Any] = {}
    if computed_metrics:
        common_metrics.update(computed_metrics)
    if prefer_provided_metrics:
        common_metrics.update(provided_common)
    else:
        for key, value in provided_common.items():
            common_metrics.setdefault(key, value)

    metrics_path = out_dir / "metrics.json"
    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": algorithm,
        "metric_source": "provided" if prefer_provided_metrics else "saved_matrices",
        **common_metrics,
        "computed_from_saved_matrices": computed_metrics,
        "algorithm_metrics": metrics or {},
    }
    write_json(metrics_path, metrics_payload)
    public_files["metrics"] = str(metrics_path)

    artifacts_path = out_dir / "artifacts.json"
    public_files["artifacts"] = str(artifacts_path)
    artifacts_payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": algorithm,
        "output_dir": str(out_dir),
        "public_files": public_files,
        "legacy_artifacts": legacy_artifacts or {},
        "notes": notes or {},
    }
    write_json(artifacts_path, artifacts_payload)
    return artifacts_payload


def aggregate_windows(
    windows: np.ndarray,
    starts: list[int] | tuple[int, ...],
    output_shape: tuple[int, ...],
    warm_up_steps: tuple[int, int] = (0, 0),
) -> np.ndarray:
    result = np.zeros(output_shape, dtype=np.float64)
    counts = np.zeros(output_shape, dtype=np.float64)
    left, right = warm_up_steps
    for window, start in zip(windows, starts):
        end = window.shape[0] - right if right > 0 else window.shape[0]
        if left >= end:
            continue
        trimmed = window[left:end]
        target_start = int(start) + left
        target_end = target_start + trimmed.shape[0]
        result[target_start:target_end] += trimmed
        counts[target_start:target_end] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = result / counts
    averaged[counts == 0] = np.nan
    return averaged.astype(np.float32)


def aggregate_window_masks(
    windows: np.ndarray,
    starts: list[int] | tuple[int, ...],
    output_shape: tuple[int, ...],
    warm_up_steps: tuple[int, int] = (0, 0),
) -> np.ndarray:
    counts = np.zeros(output_shape, dtype=np.int32)
    left, right = warm_up_steps
    for window, start in zip(windows.astype(bool), starts):
        end = window.shape[0] - right if right > 0 else window.shape[0]
        if left >= end:
            continue
        trimmed = window[left:end]
        target_start = int(start) + left
        target_end = target_start + trimmed.shape[0]
        counts[target_start:target_end] += trimmed.astype(np.int32)
    return counts > 0
