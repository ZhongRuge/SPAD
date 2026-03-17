import numpy as np

class NoiseInjector:
    def __init__(self, config, rng):
        self.rng = rng
        self.height = int(config["sensor"]["height"])
        self.width = int(config["sensor"]["width"])
        self.fps = int(config["sensor"]["fps"])

        noise_config = config.get("noise", {})
        self.lam_dcr = float(noise_config.get("dcr_cps", 0.0)) / self.fps
        self.crosstalk_prob = float(noise_config.get("crosstalk_prob", 0.0))
        self.afterpulsing_prob = float(noise_config.get("afterpulsing_prob", 0.0))
        self.previous_frame = np.zeros((self.height, self.width), dtype=bool)

    def apply_noise(self, clean_batch):
        noisy_batch = clean_batch.astype(bool, copy=True)
        noisy_batch = self._apply_dark_counts(noisy_batch)
        noisy_batch = self._apply_crosstalk(noisy_batch)
        noisy_batch = self._apply_afterpulsing(noisy_batch)
        return noisy_batch.astype(np.uint8)

    def _apply_dark_counts(self, frame_batch):
        if self.lam_dcr <= 0:
            return frame_batch

        dark_counts = self.rng.poisson(self.lam_dcr, size=frame_batch.shape) > 0
        return frame_batch | dark_counts

    def _apply_crosstalk(self, frame_batch):
        if self.crosstalk_prob <= 0:
            return frame_batch

        source_batch = frame_batch.copy()
        propagated_batch = frame_batch.copy()
        for shift_y, shift_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_hits = self._shift_active_pixels(source_batch, shift_y, shift_x)
            random_hits = self.rng.random(frame_batch.shape) < self.crosstalk_prob
            propagated_batch |= neighbor_hits & random_hits

        return propagated_batch

    def _apply_afterpulsing(self, frame_batch):
        if self.afterpulsing_prob <= 0:
            if frame_batch.size:
                self.previous_frame = frame_batch[-1].copy()
            return frame_batch

        result = frame_batch.copy()
        previous_frame = self.previous_frame.copy()
        for frame_idx in range(result.shape[0]):
            afterpulse_hits = previous_frame & (self.rng.random(previous_frame.shape) < self.afterpulsing_prob)
            result[frame_idx] |= afterpulse_hits
            previous_frame = result[frame_idx].copy()

        self.previous_frame = previous_frame
        return result

    def _shift_active_pixels(self, frame_batch, shift_y, shift_x):
        shifted = np.zeros_like(frame_batch, dtype=bool)

        src_y = slice(max(0, -shift_y), frame_batch.shape[1] - max(0, shift_y))
        dst_y = slice(max(0, shift_y), frame_batch.shape[1] - max(0, -shift_y))
        src_x = slice(max(0, -shift_x), frame_batch.shape[2] - max(0, shift_x))
        dst_x = slice(max(0, shift_x), frame_batch.shape[2] - max(0, -shift_x))

        shifted[:, dst_y, dst_x] = frame_batch[:, src_y, src_x]
        return shifted