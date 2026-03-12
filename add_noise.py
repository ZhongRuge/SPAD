class NoiseInjector:
    def __init__(self, config):
        # 预留给未来读取 DCR、串扰等配置
        pass

    def apply_noise(self, clean_batch):
        """目前不做任何处理，直接返回干净的数据"""
        return clean_batch