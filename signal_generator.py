import numpy as np

class SignalGenerator:
    def __init__(self, config, rng):
        self.width = int(config["sensor"]["width"])
        self.height = int(config["sensor"]["height"])
        self.fps = int(config["sensor"]["fps"])
        self.rng = rng

        self.scene_type = config["scene"]["type"]
        self.bg_cps = float(config["scene"]["background_cps"])
        self.sig_cps = float(config["scene"]["signal_cps"])
        self.radius = int(config["scene"]["target_radius"])
        self.velocity = float(config["scene"]["velocity_pps"])

        self.lam_bg = self.bg_cps / self.fps
        self.lam_sig = self.sig_cps / self.fps
        self.current_frame_idx = 0

        self.Y, self.X = np.ogrid[:self.height, :self.width]

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
        photons = self.rng.poisson(self.lam_bg, size=(batch_size, self.height, self.width))
        sensor_data = (photons > 0).astype(np.uint8)
        return sensor_data, []

    def _generate_moving_circle(self, batch_size):
        photons = self.rng.poisson(self.lam_bg, size=(batch_size, self.height, self.width))
        y_center = self.height // 2

        ground_truth = []

        for i in range(batch_size):
            global_idx = self.current_frame_idx + i
            time_sec = global_idx / self.fps
            x_center = int(time_sec * self.velocity) % self.width

            ground_truth.append({
                "frame": global_idx,
                "x_center": x_center,
                "y_center": y_center,
                "radius": self.radius
            })

            dist_sq = (self.X - x_center)**2 + (self.Y - y_center)**2
            mask = dist_sq <= self.radius**2

            sig_photons = self.rng.poisson(self.lam_sig, size=(self.height, self.width))
            photons[i][mask] = sig_photons[mask]

        sensor_data = (photons > 0).astype(np.uint8)
        return sensor_data, ground_truth