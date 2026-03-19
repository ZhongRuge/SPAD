import os
from io_manager import SpadIOManager
from evaluator import CompressorEvaluator
from algorithms import RleCompressor

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
    my_algorithm = RleCompressor()
    
    # 开始评估
    evaluator = CompressorEvaluator(io_manager, my_algorithm)
    evaluator.run_evaluation(output_path)

if __name__ == "__main__":
    main()