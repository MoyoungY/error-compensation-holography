from dataclasses import dataclass

@dataclass
class PhysicalConfig:
    # Physical constants
    CM: float = 1e-2
    MM: float = 1e-3
    UM: float = 1e-6
    NM: float = 1e-9
    
    # Wavelengths (RGB)
    WAVELENGTHS = [638e-9, 520e-9, 450e-9]  # R,G,B in meters
    
    # SLM parameters
    PIXEL_PITCH: float = 6.4e-6  # meters
    IMAGE_RESOLUTION = (1072, 1920)

@dataclass 
class TrainingConfig:
    batch_size: int = 1
    num_epochs: int = 20
    learning_rate: float = 1e-3
    scale_output: float = 0.95 
    train_root: str = 'data/DIV2K_train_HR/rgb'
    valid_root: str = 'data/DIV2K_valid_HR/rgb'
    exp_name: str = 'exp'
