import numpy as np
from simulation_io import resolve_crosstalk_probabilities


class NoiseInjector:
    ORTHOGONAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    def __init__(self, config, rng):
        self.rng = rng
        self.height = int(config["sensor"]["height"])
        self.width = int(config["sensor"]["width"])
        self.fps = int(config["sensor"]["fps"])

        noise_config = config.get("noise", {})
        self.lam_dcr = float(noise_config.get("dcr_cps", 0.0)) / self.fps
        self.dark_hit_prob = 1.0 - np.exp(-self.lam_dcr)
        crosstalk = resolve_crosstalk_probabilities(noise_config)
        self.crosstalk_orthogonal_prob = crosstalk["orthogonal_prob"]
        self.crosstalk_diagonal_prob = crosstalk["diagonal_prob"]
        self.afterpulsing_prob = float(noise_config.get("afterpulsing_prob", 0.0))
        self.previous_frame = np.zeros((self.height, self.width), dtype=bool)

    def apply_noise(self, clean_batch):
        # 噪声注入顺序会影响统计特性，这里固定为 暗计数 -> 串扰 -> 后脉冲。
        noisy_batch = clean_batch.astype(bool, copy=True)
        noisy_batch = self._apply_dark_counts(noisy_batch)
        noisy_batch = self._apply_crosstalk(noisy_batch)
        noisy_batch = self._apply_afterpulsing(noisy_batch)
        return noisy_batch.astype(np.uint8)

    def _apply_dark_counts(self, frame_batch):
        if self.dark_hit_prob <= 0:
            return frame_batch

        dark_counts = self.rng.random(size=frame_batch.shape, dtype=np.float32) < self.dark_hit_prob
        return frame_batch | dark_counts

    def _apply_crosstalk(self, frame_batch):
        if self.crosstalk_orthogonal_prob <= 0 and self.crosstalk_diagonal_prob <= 0:
            return frame_batch

        # 公开测量通常表明最近正交邻域的瞬时串扰显著高于对角邻域，
        # 因此这里采用 8 邻域两级概率模型，而不是所有方向共用一个概率。
        source_coords = np.argwhere(frame_batch)
        if source_coords.size == 0:
            return frame_batch

        propagated_batch = frame_batch.copy()
        self._propagate_sparse_crosstalk(
            propagated_batch,
            source_coords,
            self.ORTHOGONAL_OFFSETS,
            self.crosstalk_orthogonal_prob,
        )
        self._propagate_sparse_crosstalk(
            propagated_batch,
            source_coords,
            self.DIAGONAL_OFFSETS,
            self.crosstalk_diagonal_prob,
        )

        return propagated_batch

    def _apply_afterpulsing(self, frame_batch):
        if self.afterpulsing_prob <= 0:
            if frame_batch.size:
                self.previous_frame = frame_batch[-1].copy()
            return frame_batch

        # 使用上一帧的 avalanche 结果估计本帧的后脉冲，保留时间相关性。
        result = frame_batch.copy()
        previous_frame = self.previous_frame.copy()
        for frame_idx in range(result.shape[0]):
            previous_hits = np.argwhere(previous_frame)
            if previous_hits.size:
                fired = self.rng.random(previous_hits.shape[0], dtype=np.float32) < self.afterpulsing_prob
                if np.any(fired):
                    selected_hits = previous_hits[fired]
                    result[frame_idx, selected_hits[:, 0], selected_hits[:, 1]] = True
            previous_frame = result[frame_idx].copy()

        self.previous_frame = previous_frame
        return result

    def _propagate_sparse_crosstalk(self, propagated_batch, source_coords, offsets, probability):
        if probability <= 0:
            return

        for shift_y, shift_x in offsets:
            dst_frames = source_coords[:, 0]
            dst_y = source_coords[:, 1] + shift_y
            dst_x = source_coords[:, 2] + shift_x

            valid = (
                (dst_y >= 0)
                & (dst_y < self.height)
                & (dst_x >= 0)
                & (dst_x < self.width)
            )
            if not np.any(valid):
                continue

            valid_count = int(np.count_nonzero(valid))
            fired = self.rng.random(valid_count, dtype=np.float32) < probability
            if not np.any(fired):
                continue

            propagated_batch[
                dst_frames[valid][fired],
                dst_y[valid][fired],
                dst_x[valid][fired],
            ] = True

        return