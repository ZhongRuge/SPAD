"""实验输出管理：run_id、variant name、输出目录、JSON 写入。"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def build_run_id(now=None):
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def build_timestamp(now=None):
    now = now or datetime.now()
    return now.isoformat(timespec="seconds")


def dataset_name_from_paths(dataset_paths):
    return Path(dataset_paths["data_path"]).stem


def _slugify(value):
    normalized = str(value).strip().lower().replace("_", "-")
    normalized = re.sub(r"[^a-z0-9\-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "default"


def _flatten_params(params):
    if not params:
        return [("default", "default")]
    flat = []
    for key in sorted(params):
        val = params[key]
        if isinstance(val, dict):
            for ck, cv in _flatten_params(val):
                flat.append((f"{key}.{ck}", cv))
        else:
            flat.append((key, val))
    return flat


def build_variant_name(algorithm_id, params):
    norm_id = _slugify(algorithm_id)
    parts = []
    for key, val in _flatten_params(params):
        parts.append("default" if key == "default" else f"{_slugify(key)}-{_slugify(val)}")
    param_label = "__".join(parts)

    payload = json.dumps(params or {}, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{norm_id}__{param_label}__{digest}"


def prepare_run_output_root(compression_output_dir, dataset_name, run_id):
    run_dir = Path(compression_output_dir) / "runs" / dataset_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_algorithm_output_dir(run_dir, variant_name):
    alg_dir = Path(run_dir) / variant_name
    alg_dir.mkdir(parents=True, exist_ok=True)
    return alg_dir


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def find_latest_run_dir(compression_output_dir, dataset_name):
    dataset_runs_dir = Path(compression_output_dir) / "runs" / dataset_name
    if not dataset_runs_dir.exists():
        return None
    run_dirs = [e for e in dataset_runs_dir.iterdir() if e.is_dir()]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda e: e.name)[-1]
