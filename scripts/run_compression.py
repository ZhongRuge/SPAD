"""压缩评测入口脚本。

对应原 compress/main_compress.py，改用标准包导入，
使用优化后的单遍评估器。
"""

import os
from pathlib import Path

from spad.compress import build_compressor
from spad.config import (
    load_config,
    resolve_algorithm_params,
    resolve_compression_output_dir,
    resolve_dataset_paths,
    resolve_enabled_algorithms,
    resolve_evaluator_config,
)
from spad.evaluate import (
    CompressorEvaluator,
    build_run_id,
    build_timestamp,
    build_variant_name,
    dataset_name_from_paths,
    prepare_algorithm_output_dir,
    prepare_run_output_root,
    write_json,
)
from spad.io import SpadReader


def _print_config_summary(config, dataset_paths, output_dir, evaluator_config):
    sensor = config.get("sensor", {})
    sim = config.get("simulation", {})
    scene = config.get("scene", {})
    noise = config.get("noise", {})
    algos = resolve_enabled_algorithms(config)

    print("\n=== Compression Run Config Summary ===")
    print(f"Dataset: {Path(dataset_paths['data_path']).resolve()}")
    print(f"Output:  {Path(output_dir).resolve()}")
    print(f"Sensor:  {sensor.get('width')}x{sensor.get('height')} @ {sensor.get('fps')} fps")
    print(f"Scene:   {scene.get('type')} | bg={scene.get('background_cps')} cps")
    print(f"Noise:   dcr={noise.get('dcr_cps')} | xt_orth={noise.get('crosstalk_orthogonal_prob')}")
    print(f"Batch:   {evaluator_config.get('batch_size')} | verify_lossless={evaluator_config.get('verify_lossless')}")
    print(f"Algorithms ({len(algos)}): {', '.join(algos)}")
    print("=" * 40 + "\n")


def main():
    config = load_config()
    dataset_paths = resolve_dataset_paths(config)
    output_dir = resolve_compression_output_dir(config)
    evaluator_config = resolve_evaluator_config(config)
    _print_config_summary(config, dataset_paths, output_dir, evaluator_config)

    meta_path = dataset_paths["meta_path"]
    data_path = dataset_paths["data_path"]

    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        print("Dataset files missing. Run scripts/generate_data.py first.")
        return

    reader = SpadReader(meta_path, data_path)
    dataset_name = dataset_name_from_paths(dataset_paths)
    run_id = build_run_id()
    timestamp = build_timestamp()
    run_dir = prepare_run_output_root(output_dir, dataset_name, run_id)

    algorithm_records = []

    for algorithm_id in resolve_enabled_algorithms(config):
        params = resolve_algorithm_params(config, algorithm_id)
        compressor = build_compressor(algorithm_id, params)
        variant_name = build_variant_name(algorithm_id, params)
        alg_dir = prepare_algorithm_output_dir(run_dir, variant_name)

        output_path = alg_dir / "compressed.bin"
        metrics_path = alg_dir / "metrics.json"
        manifest_path = alg_dir / "manifest.json"

        evaluator = CompressorEvaluator(
            reader, compressor,
            batch_size=evaluator_config["batch_size"],
            verify_lossless=evaluator_config["verify_lossless"],
        )
        metrics = evaluator.run_evaluation(str(output_path))

        manifest = {
            "schema_version": 1,
            "experiment_type": "compression",
            "run_id": run_id,
            "timestamp": timestamp,
            "dataset": {
                "name": dataset_name,
                "data_path": str(Path(data_path).resolve()),
                "meta_path": str(Path(meta_path).resolve()),
            },
            "algorithm": {
                "id": algorithm_id,
                "display_name": compressor.algorithm_name,
                "params": params,
                "variant_name": variant_name,
            },
            "runtime": {
                "batch_size": evaluator_config["batch_size"],
                "verify_lossless": evaluator_config["verify_lossless"],
            },
            "outputs": {
                "artifact_dir": str(alg_dir.resolve()),
                "compressed_file": str(output_path.resolve()),
                "metrics_file": str(metrics_path.resolve()),
            },
        }

        metrics_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "timestamp": timestamp,
            "dataset_name": dataset_name,
            "algorithm_id": algorithm_id,
            "algorithm_display_name": compressor.algorithm_name,
            "algorithm_params": params,
            **metrics,
            "input_meta_path": str(Path(meta_path).resolve()),
            "output_file_path": str(output_path.resolve()),
        }

        write_json(manifest_path, manifest)
        write_json(metrics_path, metrics_payload)

        algorithm_records.append({
            "algorithm_id": algorithm_id,
            "variant_name": variant_name,
            "artifact_dir": str(alg_dir.resolve()),
            "compressed_file": str(output_path.resolve()),
            "metrics_file": str(metrics_path.resolve()),
            "manifest_file": str(manifest_path.resolve()),
        })

    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset": {
            "name": dataset_name,
            "data_path": str(Path(data_path).resolve()),
            "meta_path": str(Path(meta_path).resolve()),
        },
        "compression_output_root": str(Path(output_dir).resolve()),
        "run_dir": str(run_dir.resolve()),
        "algorithms": algorithm_records,
    }
    write_json(run_dir / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()
