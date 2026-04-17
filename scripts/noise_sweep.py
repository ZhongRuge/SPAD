"""噪声扫描实验脚本。

关键改进：
  - 不再写磁盘修改 config.yaml → 直接 deepcopy config dict 传入 run_simulation
  - 复用 spad.compress.build_compressor 和统一评估逻辑
  - 不再需要 try/finally 恢复配置文件
"""

import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from spad import PROJECT_ROOT
from spad.compress import build_compressor, list_algorithms
from spad.compress.aer import AerCompressor
from spad.compress.delta_rle import DeltaRleCompressor
from spad.compress.delta_sparse import DeltaSparseCompressor
from spad.compress.delta_sparse_varint_zlib import DeltaSparseVarintZlibCompressor
from spad.compress.delta_sparse_zlib import DeltaSparseZlibCompressor
from spad.compress.global_event_stream import GlobalEventStreamCompressor
from spad.compress.packbits_zlib import PackBitsZlibCompressor
from spad.compress.rle import RleCompressor
from spad.compress.temporal_binning import TemporalBinningCompressor
from spad.config import load_config, resolve_output_paths
from spad.generate import run_simulation
from spad.io import CompressedWriter, SpadReader, stream_compressed_chunks

RESULTS_JSON = PROJECT_ROOT / "noise_compression_results.json"
RESULTS_CSV = PROJECT_ROOT / "noise_compression_results.csv"
SWEEP_DIR = PROJECT_ROOT / "data" / "noise_sweep"


def cps_from_expected_hits(expected_hits: float, pixels: int, fps: int) -> float:
    prob = expected_hits / pixels
    if prob < 0 or prob >= 1:
        raise ValueError("expected_hits must map to probability in [0, 1)")
    if prob == 0:
        return 0.0
    return -fps * math.log1p(-prob)


def hits_from_cps(cps: float, pixels: int, fps: int) -> float:
    return pixels * (1.0 - math.exp(-float(cps) / fps))


def _configured_cases(config):
    cases_cfg = config.get("benchmark", {}).get("noise_sweep", {}).get("cases")
    if not cases_cfg:
        return None
    result = []
    for c in cases_cfg:
        if not isinstance(c, dict):
            raise ValueError("noise_sweep cases must be mappings")
        name = str(c.get("name", "")).strip()
        if not name:
            raise ValueError("case name cannot be blank")
        result.append({
            "name": name,
            "description": str(c.get("description", name)).strip() or name,
            "background_cps": float(c["background_cps"]),
            "dcr_cps": float(c["dcr_cps"]),
        })
    return result


def build_noise_cases(config):
    fps = int(config["sensor"]["fps"])
    pixels = int(config["sensor"]["width"]) * int(config["sensor"]["height"])

    cases = _configured_cases(config)
    if cases is None:
        bg = float(config["scene"]["background_cps"])
        dcr = float(config["noise"]["dcr_cps"])
        cases = [
            {"name": "baseline", "description": "Config baseline", "background_cps": bg, "dcr_cps": dcr},
            {"name": "low_noise", "description": "Low noise", "background_cps": cps_from_expected_hits(8, pixels, fps), "dcr_cps": cps_from_expected_hits(8, pixels, fps)},
            {"name": "medium_noise", "description": "Medium noise", "background_cps": cps_from_expected_hits(32, pixels, fps), "dcr_cps": cps_from_expected_hits(32, pixels, fps)},
            {"name": "high_noise", "description": "High noise", "background_cps": cps_from_expected_hits(128, pixels, fps), "dcr_cps": cps_from_expected_hits(128, pixels, fps)},
        ]

    for c in cases:
        c["expected_background_hits_per_frame"] = hits_from_cps(c["background_cps"], pixels, fps)
        c["expected_dcr_hits_per_frame"] = hits_from_cps(c["dcr_cps"], pixels, fps)
        c["expected_total_independent_noise_hits_per_frame"] = (
            c["expected_background_hits_per_frame"] + c["expected_dcr_hits_per_frame"]
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


def summarize_dataset(reader: SpadReader):
    total_active = 0
    total_frames = 0
    max_active = 0
    ppf = reader.width * reader.height

    for batch in reader.stream_batches(batch_size=1000):
        counts = np.sum(batch, axis=(1, 2), dtype=np.int64)
        total_active += int(np.sum(counts))
        total_frames += batch.shape[0]
        max_active = max(max_active, int(np.max(counts)))

    avg = total_active / total_frames if total_frames else 0.0
    return {
        "average_active_pixels_per_frame": avg,
        "average_sparsity": avg / ppf if ppf else 0.0,
        "max_active_pixels_per_frame": max_active,
    }


def evaluate_compressor(reader: SpadReader, compressor, output_path: Path):
    """单遍评估（与 CompressorEvaluator 逻辑一致但返回 sweep 所需格式）。"""
    total_encode = 0.0
    total_decode = 0.0
    total_orig = reader.get_original_size_bytes()
    batch_size = 1000
    mismatch = 0
    total_px = 0

    with CompressedWriter(str(output_path)) as writer:
        for batch in reader.stream_batches(batch_size):
            t0 = time.perf_counter()
            compressed = compressor.encode(batch)
            total_encode += time.perf_counter() - t0

            writer.write_chunk(compressed, batch.shape[0])

            shape = batch.shape
            t0 = time.perf_counter()
            decoded = compressor.decode(compressed, shape)
            total_decode += time.perf_counter() - t0

            n = int(np.prod(shape, dtype=np.int64))
            total_px += n
            mismatch += int(np.count_nonzero(decoded != batch))

    compressed_bytes = os.path.getsize(output_path)
    cr = total_orig / compressed_bytes if compressed_bytes else 0.0

    return {
        "algorithm_class": compressor.__class__.__name__,
        "algorithm_name": compressor.algorithm_name,
        "is_lossless": compressor.is_lossless,
        "original_bytes": total_orig,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": cr,
        "encode_seconds": total_encode,
        "decode_seconds": total_decode,
        "lossless_passed": (mismatch == 0) if compressor.is_lossless else None,
        "mismatch_ratio": mismatch / total_px if total_px else 0.0,
    }


def run_case(base_config, case):
    """运行一个噪声情景：生成数据 + 评估所有算法。不再写磁盘修改 config。"""
    config = copy.deepcopy(base_config)
    config["scene"]["background_cps"] = float(case["background_cps"])
    config["noise"]["dcr_cps"] = float(case["dcr_cps"])

    # 覆盖输出路径
    sweep_dir = str(SWEEP_DIR)
    config["io"]["output_dir"] = sweep_dir
    config["io"]["filename"] = f"spad_dataset_{case['name']}.bin"
    config.setdefault("paths", {})
    config["paths"].setdefault("dataset", {})
    config["paths"]["dataset"]["output_dir"] = sweep_dir
    config["paths"]["dataset"]["filename"] = f"spad_dataset_{case['name']}.bin"
    # 让相对路径基于 sweep_dir 解析
    config["_config_dir"] = sweep_dir

    print(f"\n===== Noise case: {case['name']} | bg={case['background_cps']:.3f}, dcr={case['dcr_cps']:.3f} =====")

    # 直接传入 config dict，不写磁盘
    run_simulation(config=config)

    paths = resolve_output_paths(config)
    reader = SpadReader(paths["meta_path"], paths["data_path"])
    summary = summarize_dataset(reader)

    case_dir = SWEEP_DIR / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for compressor in build_test_algorithms():
        out = case_dir / f"{compressor.__class__.__name__}_compressed.bin"
        result = evaluate_compressor(reader, compressor, out)
        results.append(result)
        print(f"  {compressor.__class__.__name__}: CR={result['compression_ratio']:.4f}x, mismatch={result['mismatch_ratio']:.4%}")

    return {
        "case": case,
        "dataset_summary": summary,
        "paths": {k: str(v) for k, v in paths.items()},
        "algorithm_results": results,
    }


def save_results(all_results):
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(RESULTS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_name", "background_cps", "dcr_cps",
            "expected_total_independent_noise_hits_per_frame",
            "average_active_pixels_per_frame", "average_sparsity",
            "algorithm_class", "algorithm_name", "is_lossless",
            "compression_ratio", "compressed_bytes",
            "encode_seconds", "decode_seconds",
            "lossless_passed", "mismatch_ratio",
        ])
        for cr in all_results:
            case = cr["case"]
            summary = cr["dataset_summary"]
            for ar in cr["algorithm_results"]:
                writer.writerow([
                    case["name"],
                    round(case["background_cps"], 6),
                    round(case["dcr_cps"], 6),
                    round(case["expected_total_independent_noise_hits_per_frame"], 6),
                    round(summary["average_active_pixels_per_frame"], 6),
                    round(summary["average_sparsity"], 8),
                    ar["algorithm_class"], ar["algorithm_name"], ar["is_lossless"],
                    round(ar["compression_ratio"], 6), ar["compressed_bytes"],
                    round(ar["encode_seconds"], 6), round(ar["decode_seconds"], 6),
                    ar["lossless_passed"], round(ar["mismatch_ratio"], 8),
                ])


def main():
    config = load_config()
    cases = build_noise_cases(config)
    all_results = []

    for case in cases:
        all_results.append(run_case(config, case))

    save_results(all_results)
    print(f"\nResults: {RESULTS_JSON}")
    print(f"Results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
