import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPRESS_DIR = PROJECT_ROOT / "compress"
if str(COMPRESS_DIR) not in sys.path:
    sys.path.insert(0, str(COMPRESS_DIR))

from algorithm_registry import build_compressor
from algorithms import DeltaSparseVarintZlibCompressor
from algorithms import DeltaSparseZlibCompressor
from evaluator import CompressorEvaluator
from io_manager import SpadIOManager


class TestDeltaSparseVarintZlibCompressor(unittest.TestCase):
    def test_round_trip_sparse_batch(self):
        batch = np.zeros((12, 8, 8), dtype=np.uint8)
        batch[0, 1, 1] = 1
        batch[0, 1, 2] = 1
        batch[4, 1, 2] = 0
        batch[4, 2, 2] = 1
        batch[9, 6, 6] = 1
        batch[10, 6, 6] = 0
        batch[10, 6, 7] = 1

        compressor = DeltaSparseVarintZlibCompressor()
        compressed = compressor.encode(batch)
        decoded = compressor.decode(compressed, batch.shape)

        self.assertTrue(np.array_equal(decoded, batch))

    def test_sparse_empty_frames_compress_better_than_fixed_width_sparse_stream(self):
        batch = np.zeros((64, 16, 16), dtype=np.uint8)
        batch[3, 5, 5] = 1
        batch[3, 5, 6] = 1
        batch[19, 8, 8] = 1
        batch[42, 8, 8] = 0
        batch[42, 8, 9] = 1
        batch[57, 2, 2] = 1

        fixed_width = DeltaSparseZlibCompressor().encode(batch)
        varint_stream = DeltaSparseVarintZlibCompressor().encode(batch)

        self.assertLess(len(varint_stream), len(fixed_width))

    def test_evaluator_runs_via_registry(self):
        frames = np.zeros((10, 8, 8), dtype=np.uint8)
        frames[0, 1, 1] = 1
        frames[5, 2, 3] = 1
        frames[9, 7, 7] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            meta_path = temp_path / "sample.meta.json"
            data_path = temp_path / "sample.bin"
            output_path = temp_path / "compressed.bin"

            with open(meta_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "width": 8,
                        "height": 8,
                        "total_frames": 10,
                        "save_as_bits": True,
                    },
                    file,
                )

            packed = np.packbits(frames.reshape(frames.shape[0], -1), axis=1)
            with open(data_path, "wb") as file:
                file.write(packed.tobytes())

            io_manager = SpadIOManager(str(meta_path), str(data_path))
            compressor = build_compressor("delta_sparse_varint_zlib")
            evaluator = CompressorEvaluator(
                io_manager,
                compressor,
                batch_size=4,
                verify_lossless=True,
            )

            metrics = evaluator.run_evaluation(str(output_path))

            self.assertTrue(output_path.exists())
            self.assertTrue(metrics["is_lossless_algorithm"])
            self.assertTrue(metrics["lossless_check_passed"])
            self.assertGreater(metrics["compressed_size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
