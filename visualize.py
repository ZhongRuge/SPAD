import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def visualize_interactive():
    # 1. 读取配置和文件
    config = load_config()
    width = config['sensor']['width']
    height = config['sensor']['height']
    filepath = os.path.join(config['io']['output_dir'], config['io']['filename'])
    
    print(f"正在读取文件: {filepath} ...")
    packed_data = np.fromfile(filepath, dtype=np.uint8)
    unpacked_data = np.unpackbits(packed_data)
    
    pixels_per_frame = width * height
    total_frames = len(unpacked_data) // pixels_per_frame
    print(f"解析成功！共加载 {total_frames} 帧。可以使用左右方向键切换查看。")
    
    # 2. 重塑为 (frames, height, width) 的三维张量
    # 注意：200x200x10000 占用约 400MB 内存，普通电脑完全没问题
    video_matrix = unpacked_data.reshape((total_frames, height, width))
    
    # 3. 设置交互式画布
    fig, ax = plt.subplots(figsize=(6, 6))
    curr_idx = 0  # 当前显示的帧索引
    
    # 初始化第一帧画面
    im = ax.imshow(video_matrix[curr_idx], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    
    # 定义更新标题的函数
    def update_title():
        sparsity = np.mean(video_matrix[curr_idx])
        ax.set_title(f"SPAD Frame: {curr_idx} / {total_frames - 1}\nSparsity: {sparsity:.2%}")
        fig.canvas.draw_idle() # 触发重绘

    update_title()
    ax.axis('off')

    # 4. 定义键盘交互事件
    def on_key_press(event):
        nonlocal curr_idx
        # 按右键：下一帧
        if event.key == 'right':
            curr_idx = (curr_idx + 1) % total_frames
        # 按左键：上一帧
        elif event.key == 'left':
            curr_idx = (curr_idx - 1) % total_frames
        else:
            return  # 按了其他键不作处理
            
        # 更新画面数据和标题
        im.set_data(video_matrix[curr_idx])
        update_title()

    # 绑定键盘事件
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # 提示信息
    plt.figtext(0.5, 0.01, "Press ⬅️ Left or Right ➡️ Arrow Keys to navigate frames", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    # 显示窗口 (阻塞式，直到关闭窗口)
    plt.show()

if __name__ == "__main__":
    visualize_interactive()