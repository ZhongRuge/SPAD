import numpy as np
import matplotlib.pyplot as plt
from simulation_io import load_config
from simulation_io import load_video_matrix
from simulation_io import read_metadata
from simulation_io import resolve_output_paths

def visualize():
    config = load_config()
    paths = resolve_output_paths(config)
    meta = read_metadata(paths["meta_path"])
    video_matrix = load_video_matrix(paths["data_path"], meta)

    width = meta["width"]
    height = meta["height"]
    total_frames = meta["total_frames"]
    save_as_bits = meta.get("save_as_bits", True)

    print(f"读取元数据: {width}x{height}, 共 {total_frames} 帧, PackBits模式: {save_as_bits}")

    state = {
        'curr_idx': 0,
        'playing': False,
        'step': 1
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.2)
    im = ax.imshow(video_matrix[0], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')

    def update_display():
        idx = state['curr_idx']
        im.set_data(video_matrix[idx])
        sparsity = np.mean(video_matrix[idx])
        ax.set_title(f"Frame: {idx} / {total_frames - 1}\nSparsity: {sparsity:.2%}")
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
            state['curr_idx'] = (state['curr_idx'] + 5) % total_frames
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