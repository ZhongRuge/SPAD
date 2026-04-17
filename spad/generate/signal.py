import numpy as np


class SignalGenerator:
    """SPAD 合成信号生成器。"""

    def __init__(self, config, rng):
        self.width = int(config["sensor"]["width"])
        self.height = int(config["sensor"]["height"])
        self.fps = int(config["sensor"]["fps"])
        self.pde = float(config["sensor"].get("pde", 1.0))
        self.rng = rng

        self.scene_type = config["scene"]["type"]
        self.bg_cps = float(config["scene"]["background_cps"])
        self.sig_cps = float(config["scene"]["signal_cps"])
        self.radius = int(config["scene"]["target_radius"])
        self.velocity = float(config["scene"]["velocity_pps"])

        self.lam_bg = self.bg_cps * self.pde / self.fps
        self.lam_sig = self.sig_cps * self.pde / self.fps
        self.hit_prob_bg = 1.0 - np.exp(-self.lam_bg)
        self.hit_prob_sig = 1.0 - np.exp(-self.lam_sig)
        self.current_frame_idx = 0

        self.circle_offsets_y, self.circle_offsets_x = self._build_circle_offsets()
        self.y_center = self.height // 2

    def generate_batch(self, batch_size):
        if self.scene_type == "uniform_poisson":
            result = self._generate_uniform_poisson(batch_size)
        elif self.scene_type == "moving_circle":
            result = self._generate_moving_circle(batch_size)
        else:
            raise ValueError(f"不支持的场景类型: {self.scene_type}")

        self.current_frame_idx += batch_size
        return result

    def _generate_uniform_poisson(self, batch_size):
        data = self._sample_hits(self.hit_prob_bg, (batch_size, self.height, self.width))
        return data, []

    def _generate_moving_circle(self, batch_size):
        data = self._sample_hits(self.hit_prob_bg, (batch_size, self.height, self.width))
        frame_indices = np.arange(
            self.current_frame_idx, self.current_frame_idx + batch_size, dtype=np.int64
        )
        x_centers = ((frame_indices * self.velocity) / self.fps).astype(np.int64) % self.width

        ground_truth = [
            {
                "frame": int(fi),
                "x_center": int(xc),
                "y_center": self.y_center,
                "radius": self.radius,
            }
            for fi, xc in zip(frame_indices, x_centers)
        ]

        if self.hit_prob_sig <= 0 or self.circle_offsets_y.size == 0:
            return data, ground_truth

        for bi, xc in enumerate(x_centers):
            ys = self.circle_offsets_y + self.y_center
            xs = self.circle_offsets_x + int(xc)
            valid = (ys >= 0) & (ys < self.height) & (xs >= 0) & (xs < self.width)
            if not np.any(valid):
                continue
            hits = self._sample_hits(self.hit_prob_sig, int(np.count_nonzero(valid)))
            data[bi, ys[valid], xs[valid]] |= hits

        return data, ground_truth

    def _sample_hits(self, prob, size):
        if prob <= 0:
            return np.zeros(size, dtype=np.uint8)
        return (self.rng.random(size=size, dtype=np.float32) < prob).astype(np.uint8)

    def _build_circle_offsets(self):
        if self.radius <= 0:
            return np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int16)
        oy, ox = np.mgrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
        mask = oy**2 + ox**2 <= self.radius**2
        return oy[mask].astype(np.int16), ox[mask].astype(np.int16)
