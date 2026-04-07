import os
import sys
from pathlib import Path

from algorithm_registry import build_compressor
from algorithm_registry import build_output_filename
from evaluator import CompressorEvaluator
from experiment_output import build_run_id
from experiment_output import build_timestamp
from experiment_output import build_variant_name
from experiment_output import dataset_name_from_paths
from experiment_output import prepare_algorithm_output_dir
from experiment_output import prepare_run_output_root
from experiment_output import write_json
from io_manager import SpadIOManager


CURRENT_DIR = Path(__file__).resolve().parent
DATA_GENERATE_DIR = CURRENT_DIR.parent / "data_generate"
if str(DATA_GENERATE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATE_DIR))

from simulation_io import load_config
from simulation_io import resolve_algorithm_params
from simulation_io import resolve_compression_output_dir
from simulation_io import resolve_dataset_paths
from simulation_io import resolve_enabled_algorithms
from simulation_io import resolve_evaluator_config


def _print_startup_config_summary(config, dataset_paths, compression_output_dir, evaluator_config):
    sensor_cfg = config.get("sensor", {})
    simulation_cfg = config.get("simulation", {})
    scene_cfg = config.get("scene", {})
    noise_cfg = config.get("noise", {})
    runtime_cfg = config.get("runtime", {}).get("batch_size", {})
    enabled_algorithms = resolve_enabled_algorithms(config)

    print("\n=== Compression Run Config Summary ===")
    print(f"Dataset data path        : {Path(dataset_paths['data_path']).resolve()}")
    print(f"Dataset meta path        : {Path(dataset_paths['meta_path']).resolve()}")
    print(f"Output root              : {Path(compression_output_dir).resolve()}")
    print("")
    print("Sensor:")
    print(f"  width x height         : {sensor_cfg.get('width')} x {sensor_cfg.get('height')}")
    print(f"  fps                    : {sensor_cfg.get('fps')}")
    print(f"  pde                    : {sensor_cfg.get('pde')}")
    print(f"  dead_time_ns           : {sensor_cfg.get('dead_time_ns')}")
    print("")
    print("Simulation:")
    print(f"  total_seconds          : {simulation_cfg.get('total_seconds')}")
    print(f"  batch_size(generate)   : {simulation_cfg.get('batch_size')}")
    print(f"  random_seed            : {simulation_cfg.get('random_seed')}")
    print("")
    print("Scene:")
    print(f"  type                   : {scene_cfg.get('type')}")
    print(f"  background_cps         : {scene_cfg.get('background_cps')}")
    print(f"  signal_cps             : {scene_cfg.get('signal_cps')}")
    print(f"  target_radius          : {scene_cfg.get('target_radius')}")
    print(f"  velocity_pps           : {scene_cfg.get('velocity_pps')}")
    print("")
    print("Noise:")
    print(f"  dcr_cps                : {noise_cfg.get('dcr_cps')}")
    print(f"  crosstalk_orthogonal   : {noise_cfg.get('crosstalk_orthogonal_prob')}")
    print(f"  crosstalk_diagonal     : {noise_cfg.get('crosstalk_diagonal_prob')}")
    print(f"  afterpulsing_prob      : {noise_cfg.get('afterpulsing_prob')}")
    print("")
    print("Runtime/Evaluation:")
    print(f"  batch_size(compress)   : {evaluator_config.get('batch_size')}")
    print(f"  batch_size(runtime)    : {runtime_cfg.get('compress')}")
    print(f"  verify_lossless        : {evaluator_config.get('verify_lossless')}")
    print("")
    print(f"Algorithms ({len(enabled_algorithms)}):")
    for algorithm_id in enabled_algorithms:
        algorithm_params = resolve_algorithm_params(config, algorithm_id)
        if algorithm_params:
            print(f"  - {algorithm_id}: {algorithm_params}")
        else:
            print(f"  - {algorithm_id}")
    print("======================================\n")


def main():
    config = load_config(DATA_GENERATE_DIR / "config.yaml")
    dataset_paths = resolve_dataset_paths(config)
    compression_output_dir = resolve_compression_output_dir(config)
    evaluator_config = resolve_evaluator_config(config)
    _print_startup_config_summary(config, dataset_paths, compression_output_dir, evaluator_config)

    meta_path = dataset_paths["meta_path"]
    data_path = dataset_paths["data_path"]

    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        print("Missing dataset files. Please run data_generate/main_datagenerate.py first.")
        return

    io_manager = SpadIOManager(meta_path, data_path)
    dataset_name = dataset_name_from_paths(dataset_paths)
    run_id = build_run_id()
    timestamp = build_timestamp()
    run_dir = prepare_run_output_root(compression_output_dir, dataset_name, run_id)

    algorithm_records = []

    for algorithm_id in resolve_enabled_algorithms(config):
        algorithm_params = resolve_algorithm_params(config, algorithm_id)
        compressor = build_compressor(algorithm_id, algorithm_params)
        variant_name = build_variant_name(algorithm_id, algorithm_params)
        algorithm_dir = prepare_algorithm_output_dir(run_dir, variant_name)

        output_path = algorithm_dir / build_output_filename(compressor)
        metrics_path = algorithm_dir / "metrics.json"
        manifest_path = algorithm_dir / "manifest.json"

        evaluator = CompressorEvaluator(
            io_manager,
            compressor,
            batch_size=evaluator_config["batch_size"],
            verify_lossless=evaluator_config["verify_lossless"],
        )
        metrics_summary = evaluator.run_evaluation(str(output_path))

        manifest_payload = {
            "schema_version": 1,
            "experiment_type": "compression",
            "run_id": run_id,
            "timestamp": timestamp,
            "dataset": {
                "name": dataset_name,
                "data_path": str(Path(data_path).resolve()),
                "meta_path": str(Path(meta_path).resolve()),
                "source_meta_path": str(Path(meta_path).resolve()),
            },
            "algorithm": {
                "id": algorithm_id,
                "display_name": compressor.algorithm_name,
                "params": algorithm_params,
                "variant_name": variant_name,
            },
            "runtime": {
                "batch_size": evaluator_config["batch_size"],
                "verify_lossless": evaluator_config["verify_lossless"],
            },
            "outputs": {
                "artifact_dir": str(algorithm_dir.resolve()),
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
            "algorithm_params": algorithm_params,
            **metrics_summary,
            "input_meta_path": str(Path(meta_path).resolve()),
            "output_file_path": str(output_path.resolve()),
        }

        write_json(manifest_path, manifest_payload)
        write_json(metrics_path, metrics_payload)

        algorithm_records.append(
            {
                "algorithm_id": algorithm_id,
                "variant_name": variant_name,
                "artifact_dir": str(algorithm_dir.resolve()),
                "compressed_file": str(output_path.resolve()),
                "metrics_file": str(metrics_path.resolve()),
                "manifest_file": str(manifest_path.resolve()),
            }
        )

    run_manifest_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset": {
            "name": dataset_name,
            "data_path": str(Path(data_path).resolve()),
            "meta_path": str(Path(meta_path).resolve()),
        },
        "compression_output_root": str(Path(compression_output_dir).resolve()),
        "run_dir": str(run_dir.resolve()),
        "algorithms": algorithm_records,
    }
    write_json(run_dir / "run_manifest.json", run_manifest_payload)


if __name__ == "__main__":
    main()
