"""原始数据帧浏览器（交互式）。

用法：
  python scripts/view_raw_data.py

功能：
  - 方向键左右切换帧，上下跳 100 帧，PageUp/Down 跳 1000 帧
  - 空格键自动播放/暂停
  - 显示每帧稀疏度
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from spad.config import load_config, resolve_dataset_paths
from spad.io import read_metadata, load_video_matrix


def main():
    config = load_config()
    paths = resolve_dataset_paths(config)
    meta = read_metadata(paths["meta_path"])
    video = load_video_matrix(paths["data_path"], meta)

    total = meta["total_frames"]
    print(f"Loaded: {meta['width']}x{meta['height']}, {total} frames")

    state = {"idx": 0, "playing": False}

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.15)
    im = ax.imshow(video[0], cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.axis("off")

    def update():
        i = state["idx"]
        im.set_data(video[i])
        ax.set_title(f"Frame {i}/{total-1}  |  sparsity={np.mean(video[i]):.2%}")
        fig.canvas.draw_idle()

    def on_key(event):
        k = event.key
        if k == "right":    state["idx"] = (state["idx"] + 1) % total
        elif k == "left":   state["idx"] = (state["idx"] - 1) % total
        elif k == "down":   state["idx"] = (state["idx"] + 100) % total
        elif k == "up":     state["idx"] = (state["idx"] - 100) % total
        elif k == "pageup": state["idx"] = (state["idx"] + 1000) % total
        elif k == "pagedown": state["idx"] = (state["idx"] - 1000) % total
        elif k == " ":
            state["playing"] = not state["playing"]
            print("Auto-play:", "ON" if state["playing"] else "OFF")
        else:
            return
        update()

    def tick():
        if state["playing"]:
            state["idx"] = (state["idx"] + 10) % total
            update()

    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(tick)
    timer.start()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


if __name__ == "__main__":
    main()
