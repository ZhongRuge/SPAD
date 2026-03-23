import os
from io_manager import SpadIOManager
from evaluator import CompressorEvaluator
from algorithms import AerCompressor, RleCompressor, DeltaRleCompressor, DeltaSparseCompressor, DeltaSparseZlibCompressor, TemporalBinningCompressor, PackBitsZlibCompressor, FrameZeroSuppressionCompressor, RowSparseZlibCompressor, DeltaGapZlibCompressor, GlobalEventStreamCompressor, BlockSparseBitmapCompressor, MortonPackBitsZlibCompressor, H264VideoCompressor, H265VideoCompressor


def build_algorithm_groups():
    return [
        (
            "基础无损位流/游程",
            [
                RleCompressor(),
                PackBitsZlibCompressor(),
                MortonPackBitsZlibCompressor(),
            ],
        ),
        (
            "差分与事件稀疏编码",
            [
                DeltaRleCompressor(),
                DeltaSparseCompressor(),
                DeltaSparseZlibCompressor(),
                DeltaGapZlibCompressor(),
                FrameZeroSuppressionCompressor(),
                GlobalEventStreamCompressor(),
                AerCompressor(use_delta=False),
            ],
        ),
        (
            "空间结构建模",
            [
                RowSparseZlibCompressor(),
                BlockSparseBitmapCompressor(),
            ],
        ),
        (
            "传统视频编码实验",
            [
                H264VideoCompressor(),
                H265VideoCompressor(),
            ],
        ),
        (
            "有损时间聚合",
            [
                TemporalBinningCompressor(),
            ],
        ),
    ]


def print_algorithm_groups(algorithm_groups):
    print("\n========== 算法分类 ==========")
    for category_name, algorithms in algorithm_groups:
        print(f"[{category_name}]")
        for algorithm in algorithms:
            loss_tag = "无损" if algorithm.is_lossless else "有损/实验性"
            print(f"  - {algorithm.algorithm_name} | {loss_tag}")
    print("================================\n")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(current_dir, "../data/spad_dataset.meta.json")
    data_path = os.path.join(current_dir, "../data/spad_dataset.bin")
    output_path = os.path.join(current_dir, "../data/rle_compressed.bin")

    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        print("找不到数据！请先回 data_generate 跑一下 main.py。")
        return

    # 准备数据流
    io_manager = SpadIOManager(meta_path, data_path)

    algorithm_groups = build_algorithm_groups()
    print_algorithm_groups(algorithm_groups)

    # 评估
    for category_name, algorithms in algorithm_groups:
        print(f"\n########## 分类开始: {category_name} ##########")
        for my_algorithm in algorithms:
            file_name = f"{my_algorithm.__class__.__name__}_compressed.bin"
            output_path = os.path.join(current_dir, f"../data/{file_name}")

            evaluator = CompressorEvaluator(io_manager, my_algorithm)
            evaluator.run_evaluation(output_path)
        print(f"########## 分类结束: {category_name} ##########\n")

if __name__ == "__main__":
    main()