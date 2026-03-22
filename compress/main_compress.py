import os
from io_manager import SpadIOManager
from evaluator import CompressorEvaluator
from algorithms import AerCompressor, RleCompressor, DeltaRleCompressor, DeltaSparseCompressor, DeltaSparseZlibCompressor, TemporalBinningCompressor

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
    
    # 挑选算法
    algorithms_to_test = [
        RleCompressor(),
        DeltaRleCompressor(),
        DeltaSparseCompressor(),
        DeltaSparseZlibCompressor(),
        AerCompressor(use_delta=False),  # use_delta=False: 记录所有到来的光子 (标准 SPAD 模式)  use_delta=True:  只记录发生变化的像素 (标准 DVS 仿生视觉模式)
        TemporalBinningCompressor()      
    ]
    
    # 评估
    for my_algorithm in algorithms_to_test:
        file_name = f"{my_algorithm.__class__.__name__}_compressed.bin"
        output_path = os.path.join(current_dir, f"../data/{file_name}")
        
        # 实例化并开始测试
        evaluator = CompressorEvaluator(io_manager, my_algorithm)
        evaluator.run_evaluation(output_path)

if __name__ == "__main__":
    main()