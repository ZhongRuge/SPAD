import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DATA_GENERATE_DIR = REPO_ROOT / "data_generate"
COMPRESS_DIR = REPO_ROOT / "compress"

if str(DATA_GENERATE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATE_DIR))
if str(COMPRESS_DIR) not in sys.path:
    sys.path.insert(0, str(COMPRESS_DIR))

from algorithm_registry import build_compressor
from algorithm_registry import build_output_filename
from experiment_output import dataset_name_from_paths
from experiment_output import find_latest_run_dir
from io_manager import SpadIOManager
from simulation_io import expected_storage_bytes
from simulation_io import load_config
from simulation_io import load_video_matrix
from simulation_io import read_metadata
from simulation_io import resolve_compression_output_dir
from simulation_io import resolve_dataset_paths


class ValidationRunner:
    def __init__(self):
        self.total_checks = 0
        self.failed_checks = 0

    def pass_check(self, name, detail=""):
        self.total_checks += 1
        suffix = f" | {detail}" if detail else ""
        print(f"[PASS] {name}{suffix}")

    def fail_check(self, name, detail):
        self.total_checks += 1
        self.failed_checks += 1
        print(f"[FAIL] {name} | {detail}")

    def run(self, name, func):
        try:
            detail = func()
            self.pass_check(name, detail if isinstance(detail, str) else "")
        except AssertionError as exc:
            self.fail_check(name, str(exc))
        except Exception as exc:
            self.fail_check(name, f"{type(exc).__name__}: {exc}")

    @property
    def passed_checks(self):
        return self.total_checks - self.failed_checks


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_ground_truth_records(ground_truth_path):
    with open(ground_truth_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate SPAD data pipeline consistency and experiment artifacts.",
    )
    parser.add_argument(
        "--config",
        default=str(DATA_GENERATE_DIR / "config.yaml"),
        help="Path to the shared project config YAML.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional explicit compression run directory to validate. Defaults to latest run of the configured dataset.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    runner = ValidationRunner()

    config = load_config(Path(args.config))
    dataset_paths = resolve_dataset_paths(config)
    compression_output_dir = resolve_compression_output_dir(config)

    data_path = Path(dataset_paths["data_path"])
    meta_path = Path(dataset_paths["meta_path"])
    ground_truth_path = Path(dataset_paths["ground_truth_path"])
    dataset_name = dataset_name_from_paths(dataset_paths)

    print("=== SPAD Pipeline Validation ===")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Dataset: {data_path.resolve()}")

    shared_state = {}

    runner.run(
        "dataset files exist",
        lambda: (
            _assert(data_path.exists(), f"missing data file: {data_path}"),
            _assert(meta_path.exists(), f"missing metadata file: {meta_path}"),
            "data/meta found",
        )[-1],
    )

    def check_dataset_size():
        meta = read_metadata(meta_path)
        shared_state["meta"] = meta
        expected_bytes = expected_storage_bytes(meta)
        actual_bytes = data_path.stat().st_size
        _assert(
            actual_bytes == expected_bytes,
            f"dataset size mismatch: expected={expected_bytes}, actual={actual_bytes}",
        )
        return f"expected={expected_bytes}, actual={actual_bytes}"

    runner.run("raw dataset file size matches metadata", check_dataset_size)

    def check_full_load_valid():
        meta = shared_state["meta"]
        video_matrix = load_video_matrix(data_path, meta)
        shared_state["video_matrix"] = video_matrix

        expected_shape = (meta["total_frames"], meta["height"], meta["width"])
        _assert(video_matrix.shape == expected_shape, f"shape mismatch: expected={expected_shape}, actual={video_matrix.shape}")
        _assert(video_matrix.dtype == np.uint8, f"dtype mismatch: expected=uint8, actual={video_matrix.dtype}")
        _assert(np.all((video_matrix == 0) | (video_matrix == 1)), "video matrix contains values outside {0,1}")
        return f"shape={video_matrix.shape}, dtype={video_matrix.dtype}, mean={video_matrix.mean():.6f}"

    runner.run("load_video_matrix full read is shape/dtype/value-range valid", check_full_load_valid)

    def check_pack_unpack_roundtrip():
        meta = shared_state["meta"]
        video_matrix = shared_state["video_matrix"]
        disk_bytes = data_path.read_bytes()

        if meta.get("save_as_bits", True):
            repacked = np.packbits(video_matrix.reshape(video_matrix.shape[0], -1), axis=1).reshape(-1).tobytes()
        else:
            repacked = video_matrix.reshape(-1).astype(np.uint8).tobytes()

        _assert(repacked == disk_bytes, "repacked bytes differ from on-disk dataset bytes")
        return f"bytes={len(disk_bytes)}"

    runner.run("raw dataset pack/unpack roundtrip matches disk bytes", check_pack_unpack_roundtrip)

    def check_stream_matches_full():
        video_matrix = shared_state["video_matrix"]
        io_manager = SpadIOManager(str(meta_path), str(data_path))
        shared_state["io_manager"] = io_manager

        stream_batches = list(io_manager.stream_batches(batch_size=777))
        stream_matrix = np.concatenate(stream_batches, axis=0)
        _assert(stream_matrix.shape == video_matrix.shape, f"stream shape mismatch: expected={video_matrix.shape}, actual={stream_matrix.shape}")
        _assert(np.array_equal(stream_matrix, video_matrix), "stream_batches concatenation differs from load_video_matrix output")
        return f"stream_shape={stream_matrix.shape}"

    runner.run("full read and stream read are exactly identical", check_stream_matches_full)

    def check_ground_truth_exists():
        _assert(ground_truth_path.exists(), f"missing ground truth file: {ground_truth_path}")
        records = _load_ground_truth_records(ground_truth_path)
        shared_state["ground_truth_records"] = records
        _assert(records, "ground truth file is empty")
        return f"records={len(records)}"

    runner.run("ground truth file exists and is readable", check_ground_truth_exists)

    def check_ground_truth_count():
        records = shared_state["ground_truth_records"]
        meta = shared_state["meta"]
        _assert(
            len(records) == meta["total_frames"],
            f"ground truth count mismatch: expected={meta['total_frames']}, actual={len(records)}",
        )
        return f"expected={meta['total_frames']}, actual={len(records)}"

    runner.run("ground truth record count matches total frames", check_ground_truth_count)

    def check_ground_truth_sequence():
        records = shared_state["ground_truth_records"]
        required_fields = {"frame", "x_center", "y_center", "radius"}
        _assert(required_fields.issubset(records[0].keys()), f"ground truth missing required fields: {required_fields - set(records[0].keys())}")
        for idx, record in enumerate(records):
            _assert(record["frame"] == idx, f"non-contiguous frame sequence at index={idx}, frame={record['frame']}")
        return "frame sequence is contiguous from 0"

    runner.run("ground truth frame sequence is continuous", check_ground_truth_sequence)

    def check_ground_truth_bounds():
        records = shared_state["ground_truth_records"]
        meta = shared_state["meta"]
        for idx, record in enumerate(records):
            _assert(0 <= record["x_center"] < meta["width"], f"x_center out of bounds at frame={idx}: {record['x_center']}")
            _assert(0 <= record["y_center"] < meta["height"], f"y_center out of bounds at frame={idx}: {record['y_center']}")
        return f"bounds within width={meta['width']}, height={meta['height']}"

    runner.run("ground truth coordinates stay within sensor bounds", check_ground_truth_bounds)

    def resolve_run_dir():
        if args.run_dir:
            return Path(args.run_dir)
        latest_run_dir = find_latest_run_dir(compression_output_dir, dataset_name)
        _assert(latest_run_dir is not None, f"no compression run found under {compression_output_dir} for dataset {dataset_name}")
        return latest_run_dir

    def check_run_manifest():
        run_dir = resolve_run_dir()
        shared_state["run_dir"] = run_dir
        run_manifest_path = run_dir / "run_manifest.json"
        _assert(run_manifest_path.exists(), f"missing run manifest: {run_manifest_path}")
        run_manifest = _load_json(run_manifest_path)
        shared_state["run_manifest"] = run_manifest

        _assert(run_manifest["dataset"]["name"] == dataset_name, f"run manifest dataset mismatch: expected={dataset_name}, actual={run_manifest['dataset']['name']}")
        _assert(Path(run_manifest["dataset"]["data_path"]).resolve() == data_path.resolve(), "run manifest data_path does not match configured dataset")
        _assert(Path(run_manifest["dataset"]["meta_path"]).resolve() == meta_path.resolve(), "run manifest meta_path does not match configured dataset")
        _assert(run_manifest.get("algorithms"), "run manifest has empty algorithms list")
        return f"run_dir={run_dir.resolve()}, algorithms={len(run_manifest['algorithms'])}"

    runner.run("latest compression run manifest is present and consistent", check_run_manifest)

    def validate_algorithm_artifact(algorithm_entry):
        run_dir = shared_state["run_dir"]
        video_matrix = shared_state["video_matrix"]
        meta = shared_state["meta"]
        io_manager = shared_state["io_manager"]

        algorithm_id = algorithm_entry["algorithm_id"]
        artifact_dir = Path(algorithm_entry["artifact_dir"])
        variant_name = algorithm_entry["variant_name"]

        _assert(artifact_dir.exists(), f"artifact_dir missing: {artifact_dir}")
        _assert(artifact_dir.resolve().parent == run_dir.resolve(), f"artifact_dir not under run_dir: {artifact_dir}")

        compressed_path = artifact_dir / "compressed.bin"
        manifest_path = artifact_dir / "manifest.json"
        metrics_path = artifact_dir / "metrics.json"

        _assert(compressed_path.exists(), f"missing compressed.bin: {compressed_path}")
        _assert(manifest_path.exists(), f"missing manifest.json: {manifest_path}")
        _assert(metrics_path.exists(), f"missing metrics.json: {metrics_path}")

        manifest = _load_json(manifest_path)
        metrics = _load_json(metrics_path)

        _assert(manifest["algorithm"]["id"] == algorithm_id, f"manifest algorithm id mismatch: expected={algorithm_id}, actual={manifest['algorithm']['id']}")
        _assert(manifest["algorithm"]["variant_name"] == variant_name, f"manifest variant mismatch: expected={variant_name}, actual={manifest['algorithm']['variant_name']}")
        _assert(Path(manifest["outputs"]["compressed_file"]).resolve() == compressed_path.resolve(), "manifest compressed_file path mismatch")
        _assert(Path(manifest["outputs"]["metrics_file"]).resolve() == metrics_path.resolve(), "manifest metrics_file path mismatch")

        _assert(metrics["algorithm_id"] == algorithm_id, f"metrics algorithm id mismatch: expected={algorithm_id}, actual={metrics['algorithm_id']}")
        _assert(Path(metrics["output_file_path"]).resolve() == compressed_path.resolve(), "metrics output_file_path mismatch")
        _assert(Path(metrics["input_meta_path"]).resolve() == meta_path.resolve(), "metrics input_meta_path mismatch")
        _assert(metrics["compressed_size_bytes"] == compressed_path.stat().st_size, "metrics compressed_size_bytes does not match actual file size")

        params = manifest["algorithm"].get("params", {})
        compressor = build_compressor(algorithm_id, params)
        filename = build_output_filename(compressor)
        _assert(filename == compressed_path.name, f"compressed file name mismatch: expected={filename}, actual={compressed_path.name}")

        decoded_batches = []
        for frame_count, chunk in io_manager.stream_compressed_chunks(str(compressed_path)):
            decoded_batch = compressor.decode(chunk, (frame_count, meta["height"], meta["width"]))
            decoded_batches.append(decoded_batch)

        decoded_matrix = np.concatenate(decoded_batches, axis=0)
        _assert(decoded_matrix.shape == video_matrix.shape, f"decoded shape mismatch: expected={video_matrix.shape}, actual={decoded_matrix.shape}")
        _assert(decoded_matrix.dtype == np.uint8, f"decoded dtype mismatch: expected=uint8, actual={decoded_matrix.dtype}")

        if compressor.is_lossless:
            _assert(np.array_equal(decoded_matrix, video_matrix), "lossless decode does not exactly match the raw video matrix")
            _assert(np.all((decoded_matrix == 0) | (decoded_matrix == 1)), "lossless decoded matrix contains values outside {0,1}")
            return f"{algorithm_id}: exact roundtrip OK"

        _assert(metrics["is_lossless_algorithm"] is False, "metrics says lossless for a lossy algorithm")
        _assert(decoded_matrix.shape[0] == meta["total_frames"], "lossy decoded frame count mismatch")
        return f"{algorithm_id}: lossy decode completed, shape={decoded_matrix.shape}, dtype={decoded_matrix.dtype}"

    run_manifest = shared_state.get("run_manifest")
    if run_manifest and run_manifest.get("algorithms"):
        for algorithm_entry in run_manifest["algorithms"]:
            algorithm_id = algorithm_entry["algorithm_id"]
            runner.run(
                f"compression artifact consistency [{algorithm_id}]",
                lambda entry=algorithm_entry: validate_algorithm_artifact(entry),
            )

    print("\n=== Validation Summary ===")
    print(f"Passed: {runner.passed_checks}")
    print(f"Failed: {runner.failed_checks}")
    print(f"Total:  {runner.total_checks}")

    if runner.failed_checks == 0:
        print("Overall result: PASS")
        return 0

    print("Overall result: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
