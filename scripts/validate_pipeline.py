"""端到端流水线验证。

验证内容：
  - 数据集文件完整性（大小、shape、dtype、值域）
  - 流式读取与整包读取一致性
  - Ground truth 连续性与边界检查
  - 压缩运行 manifest / metrics / artifact 一致性
  - 每种算法的编码→解码无损往返验证
"""

import argparse
import json
from pathlib import Path

import numpy as np

from spad.config import load_config, resolve_dataset_paths, resolve_compression_output_dir
from spad.compress import build_compressor
from spad.evaluate import dataset_name_from_paths, find_latest_run_dir
from spad.io import SpadReader, read_metadata, load_video_matrix, stream_compressed_chunks


# ── 验证框架 ────────────────────────────────────────────────

class ValidationRunner:
    def __init__(self):
        self.total = 0
        self.failed = 0

    def ok(self, name, detail=""):
        self.total += 1
        print(f"[PASS] {name}" + (f" | {detail}" if detail else ""))

    def fail(self, name, detail):
        self.total += 1
        self.failed += 1
        print(f"[FAIL] {name} | {detail}")

    def run(self, name, func):
        try:
            detail = func()
            self.ok(name, detail if isinstance(detail, str) else "")
        except AssertionError as e:
            self.fail(name, str(e))
        except Exception as e:
            self.fail(name, f"{type(e).__name__}: {e}")

    @property
    def passed(self):
        return self.total - self.failed


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_gt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPAD pipeline validation")
    parser.add_argument("--config", default=None, help="Config YAML path")
    parser.add_argument("--run-dir", default=None, help="Explicit run dir to validate")
    args = parser.parse_args()

    runner = ValidationRunner()
    config = load_config(args.config)
    dp = resolve_dataset_paths(config)
    comp_dir = resolve_compression_output_dir(config)

    data_path = Path(dp["data_path"])
    meta_path = Path(dp["meta_path"])
    gt_path = Path(dp["ground_truth_path"])
    ds_name = dataset_name_from_paths(dp)

    print("=== SPAD Pipeline Validation ===")
    print(f"Dataset: {data_path.resolve()}")

    ctx = {}  # 跨步骤共享状态

    # 1. 文件存在
    runner.run("dataset files exist", lambda: (
        _assert(data_path.exists(), f"missing: {data_path}"),
        _assert(meta_path.exists(), f"missing: {meta_path}"),
        "ok",
    )[-1])

    # 2. 文件大小
    def check_size():
        meta = read_metadata(str(meta_path))
        ctx["meta"] = meta
        import math
        ppf = meta["width"] * meta["height"]
        bpf = math.ceil(ppf / 8) if meta.get("save_as_bits", True) else ppf
        expected = bpf * meta["total_frames"]
        actual = data_path.stat().st_size
        _assert(actual == expected, f"expected={expected}, actual={actual}")
        return f"{actual} bytes"
    runner.run("dataset size matches metadata", check_size)

    # 3. 整包读取
    def check_full_load():
        meta = ctx["meta"]
        vm = load_video_matrix(str(data_path), meta)
        ctx["vm"] = vm
        expected = (meta["total_frames"], meta["height"], meta["width"])
        _assert(vm.shape == expected, f"shape {vm.shape} != {expected}")
        _assert(vm.dtype == np.uint8, f"dtype={vm.dtype}")
        _assert(np.all((vm == 0) | (vm == 1)), "values outside {0,1}")
        return f"shape={vm.shape}, mean={vm.mean():.6f}"
    runner.run("full load valid", check_full_load)

    # 4. 流式 vs 整包一致
    def check_stream():
        vm = ctx["vm"]
        reader = SpadReader(str(meta_path), str(data_path))
        ctx["reader"] = reader
        batches = list(reader.stream_batches(batch_size=777))
        streamed = np.concatenate(batches, axis=0)
        _assert(np.array_equal(streamed, vm), "stream != full load")
        return f"shape={streamed.shape}"
    runner.run("stream read == full load", check_stream)

    # 5. Ground truth
    if gt_path.exists():
        def check_gt():
            recs = _load_gt(str(gt_path))
            ctx["gt"] = recs
            meta = ctx["meta"]
            _assert(len(recs) == meta["total_frames"], f"count {len(recs)} != {meta['total_frames']}")
            for i, r in enumerate(recs):
                _assert(r["frame"] == i, f"gap at {i}")
                _assert(0 <= r["x_center"] < meta["width"], f"x OOB frame {i}")
                _assert(0 <= r["y_center"] < meta["height"], f"y OOB frame {i}")
            return f"{len(recs)} records"
        runner.run("ground truth valid", check_gt)

    # 6. 压缩运行
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(comp_dir, ds_name)
    if run_dir and (run_dir / "run_manifest.json").exists():
        manifest = _load_json(run_dir / "run_manifest.json")
        runner.run("run manifest consistent", lambda: (
            _assert(manifest["dataset"]["name"] == ds_name, "dataset name mismatch"),
            f"run_dir={run_dir.name}, algorithms={len(manifest['algorithms'])}",
        )[-1])

        # 7. 每个算法 artifact
        for entry in manifest["algorithms"]:
            aid = entry["algorithm_id"]

            def validate_artifact(e=entry):
                vm = ctx["vm"]
                meta = ctx["meta"]
                adir = Path(e["artifact_dir"])
                comp_path = adir / "compressed.bin"
                metrics = _load_json(e["metrics_file"])

                _assert(comp_path.exists(), f"missing {comp_path}")
                _assert(metrics["compressed_size_bytes"] == comp_path.stat().st_size, "size mismatch")

                params = _load_json(e["manifest_file"])["algorithm"].get("params", {})
                compressor = build_compressor(e["algorithm_id"], params)

                decoded = []
                for fc, chunk in stream_compressed_chunks(str(comp_path)):
                    decoded.append(compressor.decode(chunk, (fc, meta["height"], meta["width"])))
                decoded = np.concatenate(decoded, axis=0)

                _assert(decoded.shape == vm.shape, f"shape {decoded.shape} != {vm.shape}")
                if compressor.is_lossless:
                    _assert(np.array_equal(decoded, vm), "lossless mismatch")
                    return f"{e['algorithm_id']}: exact roundtrip OK"
                return f"{e['algorithm_id']}: lossy OK, shape={decoded.shape}"

            runner.run(f"artifact [{aid}]", validate_artifact)
    else:
        print("[SKIP] No compression run found")

    print(f"\n=== Passed: {runner.passed} / {runner.total} ===")
    return 1 if runner.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
