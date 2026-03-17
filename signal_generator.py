import numpy as np


class SignalGenerator:
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

        # 对二值 SPAD 而言，单帧只关心是否发生 avalanche。
        # 泊松到达在单帧内转成 hit 的精确概率是 1 - exp(-lambda)，
        # 比先采样泊松再阈值化更省内存，也完全等价。
        self.lam_bg = self.bg_cps * self.pde / self.fps
        self.lam_sig = self.sig_cps * self.pde / self.fps
        self.hit_prob_bg = 1.0 - np.exp(-self.lam_bg)
        self.hit_prob_sig = 1.0 - np.exp(-self.lam_sig)
        self.current_frame_idx = 0

        self.circle_offsets_y, self.circle_offsets_x = self._build_circle_offsets()
        self.y_center = self.height // 2

    def generate_batch(self, batch_size):
        if self.scene_type == "uniform_poisson":
            frame_batch, ground_truth_batch = self._generate_uniform_poisson(batch_size)
        elif self.scene_type == "moving_circle":
            frame_batch, ground_truth_batch = self._generate_moving_circle(batch_size)
        else:
            raise ValueError(f"不支持的场景类型: {self.scene_type}")

        self.current_frame_idx += batch_size
        return frame_batch, ground_truth_batch

    def _generate_uniform_poisson(self, batch_size):
        sensor_data = self._sample_hits(self.hit_prob_bg, (batch_size, self.height, self.width))
        return sensor_data, []

    def _generate_moving_circle(self, batch_size):
        sensor_data = self._sample_hits(self.hit_prob_bg, (batch_size, self.height, self.width))
        frame_indices = np.arange(self.current_frame_idx, self.current_frame_idx + batch_size, dtype=np.int64)
        x_centers = ((frame_indices * self.velocity) / self.fps).astype(np.int64) % self.width

        ground_truth = [
            {
                "frame": int(frame_idx),
                "x_center": int(x_center),
                "y_center": self.y_center,
                "radius": self.radius,
            }
            for frame_idx, x_center in zip(frame_indices, x_centers)
        ]

        if self.hit_prob_sig <= 0 or self.circle_offsets_y.size == 0:
            return sensor_data, ground_truth

        for batch_idx, x_center in enumerate(x_centers):
            ys = self.circle_offsets_y + self.y_center
            xs = self.circle_offsets_x + int(x_center)
            valid = (ys >= 0) & (ys < self.height) & (xs >= 0) & (xs < self.width)
            if not np.any(valid):
                continue

            hit_count = int(np.count_nonzero(valid))
            signal_hits = self._sample_hits(self.hit_prob_sig, hit_count)
            sensor_data[batch_idx, ys[valid], xs[valid]] |= signal_hits

        return sensor_data, ground_truth

    def _sample_hits(self, hit_probability, size):
        if hit_probability <= 0:
            return np.zeros(size, dtype=np.uint8)
        return (self.rng.random(size=size, dtype=np.float32) < hit_probability).astype(np.uint8)

    def _build_circle_offsets(self):
        if self.radius <= 0:
            return np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int16)

        offsets_y, offsets_x = np.mgrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
        mask = offsets_y**2 + offsets_x**2 <= self.radius**2
        return offsets_y[mask].astype(np.int16), offsets_x[mask].astype(np.int16)