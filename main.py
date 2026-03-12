# & "G:/SPAD/SPAD_Code/.venv/Scripts/python.exe" -m pip install numpy
import os
import yaml
import numpy as np
from generate_signal import SignalGenerator
from add_noise import NoiseInjector

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_simulation():
    # 1. 加载配置
    config = load_config()
    total_frames = config['simulation']['total_frames']
    batch_size = config['simulation']['batch_size']
    
    # 确保输出目录存在
    out_dir = config['io']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, config['io']['filename'])

    # 2. 初始化模块
    generator = SignalGenerator(config)
    noise_injector = NoiseInjector(config)

    print(f"开始生成数据: {config['sensor']['width']}x{config['sensor']['height']}, 总帧数: {total_frames}")

    # 3. 分批生成与追加写入
    with open(out_path, 'wb') as f: # 'wb' 会覆盖旧文件，如果想一直追加可以用 'ab'
        for i in range(0, total_frames, batch_size):
            # 保证最后一批不会超出总帧数
            current_batch_size = min(batch_size, total_frames - i)
            
            # 步骤 A: 生成信号
            clean_data = generator.generate_batch(current_batch_size)
            
            # 步骤 B: 注入噪声 (当前为空)
            noisy_data = noise_injector.apply_noise(clean_data)
            
            # 步骤 C: 位打包 (8个uint8压缩成1个byte，极大幅度减小体积)
            packed_data = np.packbits(noisy_data)
            
            # 步骤 D: 写入二进制文件
            f.write(packed_data.tobytes())
            
            print(f"已处理 {i + current_batch_size} / {total_frames} 帧")

    # 简单验证一下文件大小
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n生成完毕！文件保存在: {out_path}")
    print(f"文件大小: {file_size_mb:.2f} MB")
    
    # 理论大小计算: (10000帧 * 200 * 200) / 8 bit = 50,000,000 bytes ≈ 47.68 MB
    # 你可以对比一下输出结果是否对得上

if __name__ == "__main__":
    run_simulation()
    