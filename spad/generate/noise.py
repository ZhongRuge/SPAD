import numpy as np

from spad.config import resolve_crosstalk_probabilities


class NoiseInjector:
    """SPAD 噪声注入器：暗计数 → 串扰 → 后脉冲。"""

    ORTHOGONAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    def __init__(self, config, rng):
        self.rng = rng
        self.height = int(config["sensor"]["height"])
        self.width = int(config["sensor"]["width"])
        self.fps = int(config["sensor"]["fps"])

        noise_cfg = config.get("noise", {})
        self.lam_dcr = float(noise_cfg.get("dcr_cps", 0.0)) / self.fps
        self.dark_hit_prob = 1.0 - np.exp(-self.lam_dcr)
        crosstalk = resolve_crosstalk_probabilities(noise_cfg)
        self.crosstalk_orthogonal_prob = crosstalk["orthogonal_prob"]
        self.crosstalk_diagonal_prob = crosstalk["diagonal_prob"]
        self.afterpulsing_prob = float(noise_cfg.get("afterpulsing_prob", 0.0))
        self.previous_frame = np.zeros((self.height, self.width), dtype=bool)

    def apply_noise(self, clean_batch):
        noisy = clean_batch.astype(bool, copy=True)
        noisy = self._apply_dark_counts(noisy)
        noisy = self._apply_crosstalk(noisy)
        noisy = self._apply_afterpulsing(noisy)
        return noisy.astype(np.uint8)

    def _apply_dark_counts(self, batch):
        if self.dark_hit_prob <= 0:
            return batch
        dark = self.rng.random(size=batch.shape, dtype=np.float32) < self.dark_hit_prob
        return batch | dark

    def _apply_crosstalk(self, batch):
        if self.crosstalk_orthogonal_prob <= 0 and self.crosstalk_diagonal_prob <= 0:
            return batch

        coords = np.argwhere(batch)
        if coords.size == 0:
            return batch

        result = batch.copy()
        self._propagate(result, coords, self.ORTHOGONAL_OFFSETS, self.crosstalk_orthogonal_prob)
        self._propagate(result, coords, self.DIAGONAL_OFFSETS, self.crosstalk_diagonal_prob)
        return result

    def _apply_afterpulsing(self, batch):
        if self.afterpulsing_prob <= 0:
            if batch.size:
                self.previous_frame = batch[-1].copy()
            return batch

        result = batch.copy()
        prev = self.previous_frame.copy()

        for i in range(result.shape[0]):
            hits = np.argwhere(prev)
            if hits.size:
                fired = self.rng.random(hits.shape[0], dtype=np.float32) < self.afterpulsing_prob
                if np.any(fired):
                    sel = hits[fired]
                    result[i, sel[:, 0], sel[:, 1]] = True
            prev = result[i].copy()

        self.previous_frame = prev
        return result

    def _propagate(self, batch, coords, offsets, prob):
        if prob <= 0:
            return
        for dy, dx in offsets:
            dst_f = coords[:, 0]
            dst_y = coords[:, 1] + dy
            dst_x = coords[:, 2] + dx
            valid = (dst_y >= 0) & (dst_y < self.height) & (dst_x >= 0) & (dst_x < self.width)
            if not np.any(valid):
                continue
            n = int(np.count_nonzero(valid))
            fired = self.rng.random(n, dtype=np.float32) < prob
            if not np.any(fired):
                continue
            batch[dst_f[valid][fired], dst_y[valid][fired], dst_x[valid][fired]] = True
