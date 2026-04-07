import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DATA_GENERATE_DIR = PROJECT_DIR / "data_generate"
COMPRESS_DIR = PROJECT_DIR / "compress"
CONFIG_PATH = DATA_GENERATE_DIR / "config.yaml"
RESULTS_JSON_PATH = PROJECT_DIR / "noise_compression_results.json"
RESULTS_CSV_PATH = PROJECT_DIR / "noise_compression_results.csv"
SWEEP_OUTPUT_DIR = PROJECT_DIR / "data" / "noise_sweep"

for import_path in (str(DATA_GENERATE_DIR), str(COMPRESS_DIR)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from data_generate.main_datagenerate import run_simulation
from data_generate.simulation_io import resolve_output_paths
from compress.io_manager import SpadIOManager
from compress.algorithms import (
    AerCompressor,
    DeltaRleCompressor,
    DeltaSparseCompressor,
    DeltaSparseVarintZlibCompressor,
    DeltaSparseZlibCompressor,
    GlobalEventStreamCompressor,
    PackBitsZlibCompressor,
    RleCompressor,
    TemporalBinningCompressor,
)


def cps_from_expected_hits_per_frame(expected_hits_per_frame: float, pixels_per_frame: int, fps: int) -> float:
    per_pixel_hit_probability = expected_hits_per_frame / pixels_per_frame
    if per_pixel_hit_probability < 0 or per_pixel_hit_probability >= 1:
        raise ValueError("expected_hits_per_frame must map to a per-pixel probability in [0, 1)")
    if per_pixel_hit_probability == 0:
        return 0.0
    return -fps * math.log1p(-per_pixel_hit_probability)


def expected_hits_per_frame_from_cps(cps: float, pixels_per_frame: int, fps: int) -> float:
    return pixels_per_frame * (1.0 - math.exp(-float(cps) / fps))


def _configured_noise_cases(base_config: dict):
    benchmark_config = base_config.get("benchmark", {})
    noise_sweep_config = benchmark_config.get("noise_sweep", {})
    configured_cases = noise_sweep_config.get("cases")
    if not configured_cases:
        return None

    cases = []
    for case in configured_cases:
        if not isinstance(case, dict):
            raise ValueError("benchmark.noise_sweep.cases entries must be mappings")

        name = str(case.get("name", "")).strip()
        if not name:
            raise ValueError("benchmark.noise_sweep.cases[].name cannot be blank")

        background_cps = float(case["background_cps"])
        dcr_cps = float(case["dcr_cps"])
        if background_cps < 0 or dcr_cps < 0:
            raise ValueError("benchmark.noise_sweep case cps values must be non-negative")

        cases.append(
            {
                "name": name,
                "description": str(case.get("description", name)).strip() or name,
                "background_cps": background_cps,
                "dcr_cps": dcr_cps,
            }
        )

    return cases


def build_noise_cases(base_config: dict):
    fps = int(base_config["sensor"]["fps"])
    pixels_per_frame = int(base_config["sensor"]["width"]) * int(base_config["sensor"]["height"])

    cases = _configured_noise_cases(base_config)
    if cases is None:
        baseline_background_cps = float(base_config["scene"]["background_cps"])
        baseline_dcr_cps = float(base_config["noise"]["dcr_cps"])
        cases = [
            {
                "name": "baseline",
                "description": "Current config baseline noise",
                "background_cps": baseline_background_cps,
                "dcr_cps": baseline_dcr_cps,
            },
            {
                "name": "low_noise",
                "description": "Balanced low-noise setting",
                "background_cps": cps_from_expected_hits_per_frame(8.0, pixels_per_frame, fps),
                "dcr_cps": cps_from_expected_hits_per_frame(8.0, pixels_per_frame, fps),
            },
            {
                "name": "medium_noise",
                "description": "Balanced medium-noise setting",
                "background_cps": cps_from_expected_hits_per_frame(32.0, pixels_per_frame, fps),
                "dcr_cps": cps_from_expected_hits_per_frame(32.0, pixels_per_frame, fps),
            },
            {
                "name": "high_noise",
                "description": "Balanced high-noise setting",
                "background_cps": cps_from_expected_hits_per_frame(128.0, pixels_per_frame, fps),
                "dcr_cps": cps_from_expected_hits_per_frame(128.0, pixels_per_frame, fps),
            },
        ]

    for case in cases:
        case["expected_background_hits_per_frame"] = expected_hits_per_frame_from_cps(
            case["background_cps"],
            pixels_per_frame,
            fps,
        )
        case["expected_dcr_hits_per_frame"] = expected_hits_per_frame_from_cps(
            case["dcr_cps"],
            pixels_per_frame,
            fps,
        )
        case["expected_total_independent_noise_hits_per_frame"] = (
            case["expected_background_hits_per_frame"] + case["expected_dcr_hits_per_frame"]
        )

    return cases


def build_test_algorithms():
    return [
        AerCompressor(use_delta=False),
        RleCompressor(),
        DeltaRleCompressor(),
        DeltaSparseCompressor(),
        DeltaSparseVarintZlibCompressor(),
        PackBitsZlibCompressor(),
        DeltaSparseZlibCompressor(),
        TemporalBinningCompressor(bin_size=255),
        GlobalEventStreamCompressor(),
    ]


def write_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def summarize_dataset(io_manager: SpadIOManager):
    total_active_pixels = 0
    total_frames = 0
    max_active_pixels = 0
    pixels_per_frame = io_manager.width * io_manager.height

    for batch_pixels in io_manager.stream_batches(batch_size=1000):
        active_counts = np.sum(batch_pixels, axis=(1, 2), dtype=np.int64)
        total_active_pixels += int(np.sum(active_counts, dtype=np.int64))
        total_frames += int(batch_pixels.shape[0])
        max_active_pixels = max(max_active_pixels, int(np.max(active_counts)))

    average_active_pixels = total_active_pixels / total_frames if total_frames else 0.0
    average_sparsity = average_active_pixels / pixels_per_frame if pixels_per_frame else 0.0
    return {
        "average_active_pixels_per_frame": average_active_pixels,
        "average_sparsity": average_sparsity,
        "max_active_pixels_per_frame": max_active_pixels,
    }


def evaluate_compressor(io_manager: SpadIOManager, compressor, output_path: Path):
    io_manager.init_compressed_file(str(output_path))
    total_encode_time = 0.0
    total_decode_time = 0.0
    total_original_bytes = io_manager.get_original_size_bytes()
    batch_size = 1000
    mismatch_pixels = 0
    total_pixels = 0
    decode_success = True

    for batch_pixels in io_manager.stream_batches(batch_size=batch_size):
        start_time = time.time()
        compressed_bytes = compressor.encode(batch_pixels)
        total_encode_time += time.time() - start_time
        io_manager.append_compressed_chunk(str(output_path), compressed_bytes, batch_pixels.shape[0])

    total_compressed_bytes = os.path.getsize(output_path)

    raw_stream = io_manager.stream_batches(batch_size=batch_size)
    chunk_stream = io_manager.stream_compressed_chunks(str(output_path))
    for raw_pixels, (frame_count, compressed_chunk) in zip(raw_stream, chunk_stream):
        batch_shape = io_manager.get_batch_shape(frame_count)
        start_time = time.time()
        decoded_pixels = compressor.decode(compressed_chunk, batch_shape)
        total_decode_time += time.time() - start_time

        compared_pixels = int(np.prod(batch_shape, dtype=np.int64))
        total_pixels += compared_pixels
        mismatch_pixels += int(np.count_nonzero(decoded_pixels != raw_pixels))
        if compressor.is_lossless and mismatch_pixels > 0:
            decode_success = False

    compression_ratio = total_original_bytes / total_compressed_bytes if total_compressed_bytes else 0.0
    mismatch_ratio = mismatch_pixels / total_pixels if total_pixels else 0.0

    return {
        "algorithm_class": compressor.__class__.__name__,
        "algorithm_name": compressor.algorithm_name,
        "is_lossless": compressor.is_lossless,
        "original_bytes": total_original_bytes,
        "compressed_bytes": total_compressed_bytes,
        "compression_ratio": compression_ratio,
        "encode_seconds": total_encode_time,
        "decode_seconds": total_decode_time,
        "lossless_passed": decode_success if compressor.is_lossless else None,
        "mismatch_ratio": mismatch_ratio,
    }


def run_case(base_config: dict, case: dict):
    config = copy.deepcopy(base_config)
    config["scene"]["background_cps"] = float(case["background_cps"])
    config["noise"]["dcr_cps"] = float(case["dcr_cps"])
    config["io"]["output_dir"] = "../data/noise_sweep"
    config["io"]["filename"] = f"spad_dataset_{case['name']}.bin"
    config.setdefault("paths", {})
    config["paths"].setdefault("dataset", {})
    config["paths"]["dataset"]["output_dir"] = "../data/noise_sweep"
    config["paths"]["dataset"]["filename"] = f"spad_dataset_{case['name']}.bin"
    write_config(config)

    print(
        f"\n===== Starting noise case: {case['name']} | "
        f"background_cps={case['background_cps']:.3f}, dcr_cps={case['dcr_cps']:.3f} ====="
    )

    run_simulation()

    paths = resolve_output_paths(config)
    io_manager = SpadIOManager(paths["meta_path"], paths["data_path"])
    dataset_summary = summarize_dataset(io_manager)

    case_output_dir = SWEEP_OUTPUT_DIR / case["name"]
    case_output_dir.mkdir(parents=True, exist_ok=True)

    algorithm_results = []
    for compressor in build_test_algorithms():
        output_path = case_output_dir / f"{compressor.__class__.__name__}_compressed.bin"
        result = evaluate_compressor(io_manager, compressor, output_path)
        algorithm_results.append(result)
        print(
            f"{compressor.__class__.__name__}: "
            f"CR={result['compression_ratio']:.4f}x, "
            f"compressed={result['compressed_bytes'] / 1024:.2f} KB, "
            f"mismatch={result['mismatch_ratio']:.4%}"
        )

    return {
        "case": case,
        "dataset_summary": dataset_summary,
        "paths": {key: str(value) for key, value in paths.items()},
        "algorithm_results": algorithm_results,
    }


def save_results(all_results):
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(all_results, json_file, indent=2, ensure_ascii=False)

    with open(RESULTS_CSV_PATH, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "case_name",
                "background_cps",
                "dcr_cps",
                "expected_total_independent_noise_hits_per_frame",
                "average_active_pixels_per_frame",
                "average_sparsity",
                "algorithm_class",
                "algorithm_name",
                "is_lossless",
                "compression_ratio",
                "compressed_bytes",
                "encode_seconds",
                "decode_seconds",
                "lossless_passed",
                "mismatch_ratio",
            ]
        )
        for case_result in all_results:
            case = case_result["case"]
            summary = case_result["dataset_summary"]
            for algorithm_result in case_result["algorithm_results"]:
                writer.writerow(
                    [
                        case["name"],
                        round(case["background_cps"], 6),
                        round(case["dcr_cps"], 6),
                        round(case["expected_total_independent_noise_hits_per_frame"], 6),
                        round(summary["average_active_pixels_per_frame"], 6),
                        round(summary["average_sparsity"], 8),
                        algorithm_result["algorithm_class"],
                        algorithm_result["algorithm_name"],
                        algorithm_result["is_lossless"],
                        round(algorithm_result["compression_ratio"], 6),
                        algorithm_result["compressed_bytes"],
                        round(algorithm_result["encode_seconds"], 6),
                        round(algorithm_result["decode_seconds"], 6),
                        algorithm_result["lossless_passed"],
                        round(algorithm_result["mismatch_ratio"], 8),
                    ]
                )


def main():
    original_config_text = CONFIG_PATH.read_text(encoding="utf-8")
    base_config = yaml.safe_load(original_config_text)
    base_config["_config_dir"] = str(DATA_GENERATE_DIR)
    cases = build_noise_cases(base_config)
    all_results = []

    try:
        for case in cases:
            all_results.append(run_case(base_config, case))
    finally:
        CONFIG_PATH.write_text(original_config_text, encoding="utf-8")

    save_results(all_results)
    print(f"\nResults written to: {RESULTS_JSON_PATH}")
    print(f"Results written to: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
