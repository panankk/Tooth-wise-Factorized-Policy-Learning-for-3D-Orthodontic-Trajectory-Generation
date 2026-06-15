from .dataset import OrthoDataset

def build_dataset(config):
    processed_root = config.DATA.PROCESSED_ROOT
    window_size = config.MODEL.get('WINDOW_SIZE', 1) 
    dataset = OrthoDataset(processed_root=processed_root, window_size=window_size)
    return dataset
   
    processed_root = config.DATA.PROCESSED_ROOT
    dataset = OrthoDataset(processed_root=processed_root)
    return dataset

__all__ = ['OrthoDataset', 'build_dataset']
