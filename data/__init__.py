"""
数据模块 (Data Module)

职责：
    对外暴露 Dataset 类，并提供工厂函数，方便被引擎 (Trainer/Inferencer) 调用。
"""

from .dataset import OrthoDataset

def build_dataset(config):
    processed_root = config.DATA.PROCESSED_ROOT
    # 🌟 从配置文件读取时序窗口大小，默认为 1 (兼容老代码)
    window_size = config.MODEL.get('WINDOW_SIZE', 1) 
    dataset = OrthoDataset(processed_root=processed_root, window_size=window_size)
    return dataset
    """
    根据配置构建 Dataset 实例。
    
    Args:
        config: 配置对象，需包含数据集路径等信息
        
    Returns:
        dataset: 实例化后的 Dataset
    """
    # 假设 config.DATA.PROCESSED_ROOT 存储了数据路径
    processed_root = config.DATA.PROCESSED_ROOT
    dataset = OrthoDataset(processed_root=processed_root)
    return dataset

__all__ = ['OrthoDataset', 'build_dataset']