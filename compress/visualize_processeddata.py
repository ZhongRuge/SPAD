import numpy as np
import matplotlib.pyplot as plt
import os
from io_manager import SpadIOManager

# 在这里导入查看的算法
from algorithms import (
    AerCompressor,
    RleCompressor,
    DeltaRleCompressor,
    DeltaSparseCompressor,
    DeltaSparseZlibCompressor,
    TemporalBinningCompressor
)

def visualize():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(current_dir, "../data/spad_dataset.meta.json")
    data_path = os.path.join(current_dir, "../data/spad_dataset.bin")

    # ==========================================
    # 切换解压算法
    # ==========================================
    # my_algorithm = TemporalBinningCompressor(bin_size=255)
    # my_algorithm = DeltaSparseZlibCompressor() 
    my_algorithm = DeltaSparseCompressor()

    # 定位该算法对应的压缩文件
    file_name = f"{my_algorithm.__class__.__name__}_compressed.bin"
    compressed_path = os.path.join(current_dir, f"../data/{file_name}")

    if not os.path.exists(compressed_path):
        print(f"找不到文件: {compressed_path}\n")
        return

    # 初始化 IO 管理器并读取元数据
    io_manager = SpadIOManager(meta_path, data_path)
    width = io_manager.width
    height = io_manager.height
    batch_size = 1000

    print(f"正在解码 {my_algorithm.algorithm_name} 的数据")
    
    # 将压缩块全部解压并拼接到内存中
    video_matrix_list = []
    chunk_stream = io_manager.stream_compressed_chunks(compressed_path)
    
    for chunk in chunk_stream:
        # 如果是最后一块，可能不足 1000 帧，但为了简便，decode 内部依靠 chunk 还原
        decoded_batch = my_algorithm.decode(chunk, (batch_size, height, width))
        video_matrix_list.append(decoded_batch)
        
    video_matrix = np.concatenate(video_matrix_list, axis=0)
    total_frames = video_matrix.shape[0]
    
    # 如果是二值图(0/1)最大值为1，如果是累加灰度图最大值为255
    is_grayscale = video_matrix.max() > 1
    vmax = 255 if is_grayscale else 1

    print(f"读取完成: {width}x{height}, 共 {total_frames} 帧")

    state = {
        'curr_idx': 0,
        'playing': False,
        'step': 1
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.2)
    im = ax.imshow(video_matrix[0], cmap='gray', interpolation='nearest', vmin=0, vmax=vmax)
    ax.axis('off')

    def update_display():
        idx = state['curr_idx']
        im.set_data(video_matrix[idx])
        
        # 针对二值图显示稀疏度，针对灰度图显示平均亮度
        if is_grayscale:
            metric = np.mean(video_matrix[idx])
            metric_str = f"Mean Intensity: {metric:.2f}"
        else:
            metric = np.mean(video_matrix[idx])
            metric_str = f"Sparsity: {metric:.2%}"
            
        ax.set_title(f"[{my_algorithm.algorithm_name}]\nFrame: {idx} / {total_frames - 1}\n{metric_str}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            state['curr_idx'] = (state['curr_idx'] + 1) % total_frames
        elif event.key == 'left':
            state['curr_idx'] = (state['curr_idx'] - 1) % total_frames
        elif event.key == 'down':
            state['curr_idx'] = (state['curr_idx'] + 100) % total_frames
        elif event.key == 'up':
            state['curr_idx'] = (state['curr_idx'] - 100) % total_frames
        elif event.key == 'pageup':
            state['curr_idx'] = (state['curr_idx'] + 1000) % total_frames
        elif event.key == 'pagedown':
            state['curr_idx'] = (state['curr_idx'] - 1000) % total_frames
        elif event.key == ' ': # 空格键切换播放/暂停
            state['playing'] = not state['playing']
            print("Auto-play:", "ON" if state['playing'] else "OFF")
        else:
            return
        update_display()

    def on_tick():
        if state['playing']:
            state['curr_idx'] = (state['curr_idx'] + 10) % total_frames
            update_display()

    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(on_tick)
    timer.start()

    fig.canvas.mpl_connect('key_press_event', on_key)

    instructions = (
        "Controls:\n"
        "Left/Right: ±1 frame  |  Up/Down: ±100 frames\n"
        "PgUp/PgDn: ±1000 frames  |  Space: Play/Pause"
    )
    plt.figtext(0.5, 0.05, instructions, ha="center", fontsize=9, 
                bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})

    print(f"Loaded {total_frames} frames. Ready to visualize.")
    plt.show()

if __name__ == "__main__":
    visualize()