"""项目配置加载、验证与路径解析。

从原 simulation_io.py 拆出的纯配置层，不包含任何 I/O 操作。
"""

import math
import os
from pathlib import Path

import yaml

from spad import PROJECT_ROOT

# ============== 常量 ==============

SCENES_WITH_GROUND_TRUTH = {"moving_circle"}

DEFAULT_ALGORITHMS = [
    "rle",
    "delta_rle",
    "delta_sparse",
    "delta_sparse_varint_zlib",
    "delta_sparse_zlib",
    "packbits_zlib",
    "global_event_stream",
    "aer",
    "temporal_binning",
]

DEFAULT_COMPRESSED_VISUALIZATION_ALGORITHM = "delta_sparse"

REFERENCE_CROSSTALK_ORTHOGONAL = 1.1e-3
REFERENCE_CROSSTALK_DIAGONAL = 1.5e-4


# ============== 配置加载 ==============

def load_config(path=None):
    """加载并验证 YAML 配置文件。

    路径解析优先级: 传入参数 > 项目根目录 config.yaml > data_generate/config.yaml
    """
    if path is not None:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
    else:
        path_obj = PROJECT_ROOT / "config.yaml"
        if not path_obj.exists():
            path_obj = PROJECT_ROOT / "data_generate" / "config.yaml"

    with open(path_obj, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_config_dir"] = str(path_obj.parent)
    validate_config(config)
    return config


# ============== 配置验证 ==============

def validate_config(config):
    sensor = config["sensor"]
    simulation = config["simulation"]
    scene = config["scene"]
    noise = config.get("noise", {})
    io_config = config["io"]
    runtime = config.get("runtime", {})
    compression = config.get("compression", {})
    evaluation = config.get("evaluation", {})
    benchmark = config.get("benchmark", {})

    _require_positive_int(sensor["width"], "sensor.width")
    _require_positive_int(sensor["height"], "sensor.height")
    _require_positive_int(sensor["fps"], "sensor.fps")
    _require_probability(sensor.get("pde", 1.0), "sensor.pde")

    if float(sensor.get("dead_time_ns", 0.0)) < 0:
        raise ValueError("sensor.dead_time_ns 必须大于等于 0")

    _require_positive_int(simulation["batch_size"], "simulation.batch_size")
    if simulation.get("total_seconds") is None:
        raise KeyError("缺少 simulation.total_seconds")
    if float(simulation["total_seconds"]) <= 0:
        raise ValueError("simulation.total_seconds 必须大于 0")

    total_frame_count = float(simulation["total_seconds"]) * int(sensor["fps"])
    rounded = round(total_frame_count)
    if not math.isclose(total_frame_count, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"simulation.total_seconds * sensor.fps 必须对应整数帧数，当前得到 {total_frame_count}"
        )

    if scene["type"] not in {"uniform_poisson", "moving_circle"}:
        raise ValueError("scene.type 仅支持 uniform_poisson 或 moving_circle")

    _require_non_negative(scene["background_cps"], "scene.background_cps")
    _require_non_negative(scene.get("signal_cps", 0.0), "scene.signal_cps")
    _require_non_negative(scene.get("target_radius", 0), "scene.target_radius")
    _require_non_negative(scene.get("velocity_pps", 0.0), "scene.velocity_pps")

    _require_non_negative(noise.get("dcr_cps", 0.0), "noise.dcr_cps")
    for key in ("crosstalk_prob", "crosstalk_orthogonal_prob", "crosstalk_diagonal_prob"):
        if key in noise:
            _require_probability(noise.get(key, 0.0), f"noise.{key}")
    _require_probability(noise.get("afterpulsing_prob", 0.0), "noise.afterpulsing_prob")

    if not str(io_config["filename"]).strip():
        raise ValueError("io.filename 不能为空")
    resolve_save_as_bits(io_config)

    runtime_batch_size = runtime.get("batch_size", {})
    if runtime_batch_size:
        if not isinstance(runtime_batch_size, dict):
            raise ValueError("runtime.batch_size must be a mapping")
        for stage, bs in runtime_batch_size.items():
            _require_positive_int(bs, f"runtime.batch_size.{stage}")

    algorithms = compression.get("algorithms")
    if algorithms is not None:
        if not isinstance(algorithms, list) or not algorithms:
            raise ValueError("compression.algorithms must be a non-empty list")
        for alg in algorithms:
            if not str(alg).strip():
                raise ValueError("compression.algorithms cannot contain blank values")

    algorithm_params = compression.get("algorithm_params", {})
    if algorithm_params and not isinstance(algorithm_params, dict):
        raise ValueError("compression.algorithm_params must be a mapping")

    verify_lossless = evaluation.get("verify_lossless")
    if verify_lossless is not None and not isinstance(verify_lossless, bool):
        raise ValueError("evaluation.verify_lossless must be a boolean")

    _validate_benchmark_config(benchmark)


def _validate_benchmark_config(benchmark):
    noise_sweep = benchmark.get("noise_sweep", {})
    cases = noise_sweep.get("cases")
    if cases is None:
        return
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark.noise_sweep.cases must be a non-empty list")
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"benchmark.noise_sweep.cases[{i}] must be a mapping")
        if not str(case.get("name", "")).strip():
            raise ValueError(f"benchmark.noise_sweep.cases[{i}].name cannot be blank")
        if "background_cps" not in case:
            raise ValueError(f"benchmark.noise_sweep.cases[{i}].background_cps is required")
        if "dcr_cps" not in case:
            raise ValueError(f"benchmark.noise_sweep.cases[{i}].dcr_cps is required")
        _require_non_negative(case.get("background_cps", 0.0), f"benchmark.noise_sweep.cases[{i}].background_cps")
        _require_non_negative(case.get("dcr_cps", 0.0), f"benchmark.noise_sweep.cases[{i}].dcr_cps")


# ============== 验证辅助函数 ==============

def _require_positive_int(value, field_name):
    v = float(value)
    if not v.is_integer() or int(v) <= 0:
        raise ValueError(f"{field_name} 必须为正整数")


def _require_non_negative(value, field_name):
    if float(value) < 0:
        raise ValueError(f"{field_name} 必须大于等于 0")


def _require_probability(value, field_name):
    p = float(value)
    if p < 0 or p > 1:
        raise ValueError(f"{field_name} 必须在 [0, 1] 区间内")


# ============== 路径解析 ==============

def resolve_save_as_bits(io_config):
    val = io_config.get("save_as_bits", True)
    if isinstance(val, bool):
        return val
    raise ValueError("io.save_as_bits 必须为布尔值 true 或 false")


def resolve_total_frames(config):
    total_seconds = float(config["simulation"]["total_seconds"])
    fps = int(config["sensor"]["fps"])
    return int(round(total_seconds * fps))


def _resolve_relative_dir(config, raw_dir):
    path_obj = Path(raw_dir)
    if not path_obj.is_absolute():
        path_obj = Path(config["_config_dir"]) / path_obj
    os.makedirs(path_obj, exist_ok=True)
    return path_obj


def resolve_dataset_paths(config):
    paths_config = config.get("paths", {})
    dataset_config = paths_config.get("dataset", {})

    output_dir = dataset_config.get("output_dir", config["io"]["output_dir"])
    filename = dataset_config.get("filename", config["io"]["filename"])
    if not str(filename).strip():
        raise ValueError("dataset filename cannot be blank")

    output_dir = _resolve_relative_dir(config, output_dir)
    data_path = os.path.join(str(output_dir), filename)
    base_name, _ = os.path.splitext(filename)
    meta_path = os.path.join(str(output_dir), f"{base_name}.meta.json")
    ground_truth_path = os.path.join(str(output_dir), f"{base_name}.ground_truth.jsonl")

    return {
        "data_path": data_path,
        "meta_path": meta_path,
        "ground_truth_path": ground_truth_path,
    }


resolve_output_paths = resolve_dataset_paths


def resolve_compression_output_dir(config):
    paths_config = config.get("paths", {})
    compression_paths = paths_config.get("compression", {})
    output_dir = compression_paths.get("output_dir", config["io"]["output_dir"])
    return str(_resolve_relative_dir(config, output_dir))


def resolve_runtime_batch_size(config, stage):
    runtime_batch_size = config.get("runtime", {}).get("batch_size", {})
    if isinstance(runtime_batch_size, dict) and runtime_batch_size.get(stage) is not None:
        return int(runtime_batch_size[stage])
    if stage in {"generate", "compress"}:
        return int(config["simulation"]["batch_size"])
    raise ValueError(f"Unknown runtime stage: {stage}")


def resolve_enabled_algorithms(config):
    algorithms = config.get("compression", {}).get("algorithms")
    if not algorithms:
        return list(DEFAULT_ALGORITHMS)
    return [str(alg).strip() for alg in algorithms]


def resolve_algorithm_params(config, algorithm_id):
    params = config.get("compression", {}).get("algorithm_params", {}).get(algorithm_id, {})
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError(f"compression.algorithm_params.{algorithm_id} must be a mapping")
    return dict(params)


def resolve_evaluator_config(config):
    evaluation = config.get("evaluation", {})
    return {
        "batch_size": resolve_runtime_batch_size(config, "compress"),
        "verify_lossless": bool(evaluation.get("verify_lossless", True)),
    }


def resolve_visualization_algorithm(config):
    compressed = config.get("visualization", {}).get("compressed", {})
    return str(compressed.get("algorithm", DEFAULT_COMPRESSED_VISUALIZATION_ALGORITHM)).strip()


# ============== 噪声参数解析 ==============

def resolve_crosstalk_probabilities(noise_config):
    orthogonal = noise_config.get("crosstalk_orthogonal_prob")
    diagonal = noise_config.get("crosstalk_diagonal_prob")

    if orthogonal is not None or diagonal is not None:
        o = float(orthogonal or 0.0)
        d = float(diagonal or 0.0)
        return {
            "orthogonal_prob": o,
            "diagonal_prob": d,
            "total_prob": 4.0 * o + 4.0 * d,
            "ratio_source": "explicit",
        }

    total = float(noise_config.get("crosstalk_prob", 0.0))
    if total <= 0:
        return {
            "orthogonal_prob": 0.0,
            "diagonal_prob": 0.0,
            "total_prob": 0.0,
            "ratio_source": "disabled",
        }

    ratio = REFERENCE_CROSSTALK_ORTHOGONAL / REFERENCE_CROSSTALK_DIAGONAL
    d = total / (4.0 * (ratio + 1.0))
    o = ratio * d
    return {
        "orthogonal_prob": o,
        "diagonal_prob": d,
        "total_prob": total,
        "ratio_source": "derived_from_total",
    }


def scene_has_ground_truth(scene_type):
    return scene_type in SCENES_WITH_GROUND_TRUTH
