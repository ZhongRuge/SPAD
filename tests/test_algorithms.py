"""压缩算法单元测试。

覆盖：
  - 所有已注册算法的 encode/decode 往返一致性
  - DeltaSparseVarintZlib 的特定优势验证
  - 通过 registry 构造 + 评估器集成测试
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spad.compress import build_compressor, list_algorithms
from spad.compress.delta_sparse_varint_zlib import DeltaSparseVarintZlibCompressor
from spad.compress.delta_sparse_zlib import DeltaSparseZlibCompressor
from spad.evaluate import CompressorEvaluator
from spad.io import SpadReader


def _make_sparse_batch(shape=(12, 8, 8), density=0.01, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < density).astype(np.uint8)


class TestAllAlgorithmsRoundTrip(unittest.TestCase):
    """对所有注册算法做往返一致性检查。"""

    def test_registered_algorithms_exist(self):
        algos = list_algorithms()
        self.assertGreaterEqual(len(algos), 9)

    def test_lossless_round_trip_all(self):
        batch = _make_sparse_batch((20, 16, 16), density=0.02)
        for algo_id in list_algorithms():
            with self.subTest(algo=algo_id):
                compressor = build_compressor(algo_id)
                compressed = compressor.encode(batch)
                decoded = compressor.decode(compressed, batch.shape)
                if compressor.is_lossless:
                    self.assertTrue(
                        np.array_equal(batch, decoded),
                        f"{algo_id} lossless round-trip failed"
                    )
                else:
                    self.assertEqual(decoded.shape, batch.shape)
                    self.assertEqual(decoded.dtype, np.uint8)

    def test_empty_batch(self):
        batch = np.zeros((5, 8, 8), dtype=np.uint8)
        for algo_id in list_algorithms():
            with self.subTest(algo=algo_id):
                compressor = build_compressor(algo_id)
                compressed = compressor.encode(batch)
                decoded = compressor.decode(compressed, batch.shape)
                self.assertEqual(decoded.shape, batch.shape)


class TestDeltaSparseVarintZlib(unittest.TestCase):

    def test_round_trip_sparse_batch(self):
        batch = np.zeros((12, 8, 8), dtype=np.uint8)
        batch[0, 1, 1] = 1
        batch[0, 1, 2] = 1
        batch[4, 2, 2] = 1
        batch[9, 6, 6] = 1
        batch[10, 6, 7] = 1

        compressor = DeltaSparseVarintZlibCompressor()
        compressed = compressor.encode(batch)
        decoded = compressor.decode(compressed, batch.shape)
        self.assertTrue(np.array_equal(decoded, batch))

    def test_compresses_better_than_fixed_width(self):
        batch = np.zeros((64, 16, 16), dtype=np.uint8)
        batch[3, 5, 5] = 1
        batch[3, 5, 6] = 1
        batch[19, 8, 8] = 1
        batch[42, 8, 9] = 1
        batch[57, 2, 2] = 1

        fixed = DeltaSparseZlibCompressor().encode(batch)
        varint = DeltaSparseVarintZlibCompressor().encode(batch)
        self.assertLess(len(varint), len(fixed))


class TestEvaluatorIntegration(unittest.TestCase):

    def test_evaluator_via_registry(self):
        frames = np.zeros((10, 8, 8), dtype=np.uint8)
        frames[0, 1, 1] = 1
        frames[5, 2, 3] = 1
        frames[9, 7, 7] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            meta_path = tmp / "sample.meta.json"
            data_path = tmp / "sample.bin"
            output_path = tmp / "compressed.bin"

            with open(meta_path, "w") as f:
                json.dump({
                    "width": 8, "height": 8,
                    "total_frames": 10, "save_as_bits": True,
                }, f)

            packed = np.packbits(frames.reshape(10, -1), axis=1)
            with open(data_path, "wb") as f:
                f.write(packed.tobytes())

            reader = SpadReader(str(meta_path), str(data_path))
            compressor = build_compressor("delta_sparse_varint_zlib")
            evaluator = CompressorEvaluator(reader, compressor, batch_size=4, verify_lossless=True)
            metrics = evaluator.run_evaluation(str(output_path))

            self.assertTrue(output_path.exists())
            self.assertTrue(metrics["is_lossless_algorithm"])
            self.assertTrue(metrics["lossless_check_passed"])
            self.assertGreater(metrics["compressed_size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
