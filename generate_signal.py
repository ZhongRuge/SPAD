import numpy as np

class SignalGenerator:
    def __init__(self, config):
        self.width = config['sensor']['width']
        self.height = config['sensor']['height']
        self.fps = config['sensor']['fps']
        
        self.scene_type = config['scene']['type']
        self.bg_cps = config['scene']['background_cps']
        self.sig_cps = config['scene']['signal_cps']
        self.radius = config['scene']['target_radius']
        self.velocity = config['scene']['velocity_pps']
        
        # 核心物理转换：计算每帧每个像素的泊松分布期望值 (lambda)
        # lambda = 每秒到达的光子数 / 每秒的帧数
        self.lam_bg = self.bg_cps / self.fps
        self.lam_sig = self.sig_cps / self.fps
        
        # 记录全局已生成的帧数，用于计算物体的绝对运动时间
        self.current_frame_idx = 0

    def generate_batch(self, batch_size):
        """路由函数：根据 yaml 里的 scene.type 调用对应的分布生成算法"""
        if self.scene_type == "uniform_poisson":
            batch_data = self._generate_uniform_poisson(batch_size)
        elif self.scene_type == "moving_circle":
            batch_data = self._generate_moving_circle(batch_size)
        else:
            raise ValueError(f"不支持的场景类型: {self.scene_type}")
        
        # 更新全局帧数计数器
        self.current_frame_idx += batch_size
        return batch_data

    # ==========================================
    # 具体分布生成函数区
    # ==========================================

    def _generate_uniform_poisson(self, batch_size):
        """生成纯均匀泊松分布 (只有底噪，无特定目标)"""
        # 1. 对整个 3D 矩阵进行泊松采样，得到每个像素落入的光子个数
        photons = np.random.poisson(self.lam_bg, size=(batch_size, self.height, self.width))
        
        # 2. SPAD 二值化特性：只要光子数 > 0，传感器就输出 1
        return (photons > 0).astype(np.uint8)

    def _generate_moving_circle(self, batch_size):
        """生成具有空间相关性(圆形)和时间相关性(匀速平移)的泊松数据"""
        # 1. 先用背景光强度生成整个矩阵的底噪
        photons = np.random.poisson(self.lam_bg, size=(batch_size, self.height, self.width))
        
        # 设定圆心在 Y 轴居中
        y_center = self.height // 2
        
        # 2. 逐帧绘制移动的目标
        for i in range(batch_size):
            global_idx = self.current_frame_idx + i
            time_sec = global_idx / self.fps
            
            # 计算 X 轴的位移 (取模运算让圆球移出右边界后从左边重新进入)
            x_center = int(time_sec * self.velocity) % self.width
            
            # 创建空间掩码 (Mask)：计算每个像素到圆心的距离
            Y, X = np.ogrid[:self.height, :self.width]
            dist_from_center = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
            mask = dist_from_center <= self.radius
            
            # 3. 在圆形区域内，按高光强度(lam_sig)重新进行泊松采样
            sig_photons = np.random.poisson(self.lam_sig, size=(self.height, self.width))
            
            # 替换该帧圆内区域的数据
            photons[i][mask] = sig_photons[mask]
            
        return (photons > 0).astype(np.uint8)