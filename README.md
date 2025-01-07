# Error-Compensation Network for Holographic Displays

This repository contains the official implementation of the paper ["Error-compensation network for ringing artifact reduction in holographic displays"](https://opg.optica.org/ol/abstract.cfm?uri=ol-49-11-3210) (Optics Letters 2024).

## Abstract

We propose an error-compensation network for reducing ringing artifacts in computer-generated holograms (CGHs). Our method introduces a specialized network architecture that effectively compensates for errors in the holographic reconstruction process, leading to improved image quality and reduced artifacts.

## Key Features

- Novel error-compensation network architecture
- Angular Spectrum Method (ASM) for accurate wave propagation
- Support for RGB hologram generation
- PyTorch Lightning implementation for efficient training
- Comprehensive evaluation on DIV2K dataset

## Requirements

- Python 3.8+
- PyTorch 1.8.0+
- PyTorch Lightning 2.0.0+
- CUDA capable GPU

For detailed requirements, see [requirements.txt](requirements.txt).

## Installation

```bash
# Clone the repository
git clone https://github.com/MoyoungY/error-compensation-holography.git
cd error-compensation-holography

# Create and activate conda environment
conda create -n errorCGH python=3.8
conda activate errorCGH

# Install dependencies
pip install -r requirements.txt
```

## Dataset

We use the DIV2K dataset for training and evaluation. Please download it from [DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and organize as follows:

```
data/
├── DIV2K_train_HR/
│   └── rgb/
└── DIV2K_valid_HR/
    └── rgb/
```

## Training

To train the model:

```bash
python main.py \
    --channel 1 \
    --exp_name your_experiment_name \
    --batch_size 1 \
    --num_epochs 20 \
    --lr 1e-3
```

Key arguments:
- `channel`: Color channel selection (0:red, 1:green, 2:blue)
- `exp_name`: Name for the experiment
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `lr`: Learning rate

## Testing

To test a trained model:

```bash
python main.py \
    --channel 1 \
    --exp_name test_experiment \
    --test True \
    --ckpt_path path/to/checkpoint.pth
```

## Results

The network achieves significant reduction in ringing artifacts compared to traditional methods. Sample results can be found in the paper.

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{yuan2024error,
  title={Error-compensation network for ringing artifact reduction in holographic displays},
  author={Yuan, Ganzhangqin and Zhou, Mi and Peng, Yifan and Chen, Muku and Geng, Zihan},
  journal={Optics Letters},
  volume={49},
  number={11},
  pages={3210--3213},
  year={2024},
  publisher={Optica Publishing Group}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- [PyTorch Lightning](https://www.pytorchlightning.ai/) for the training framework
- [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) for training data
- [Neural Holography](https://github.com/computational-imaging/neural-holography) for the ASM implementation
- [CCNN-CGH](https://github.com/flyingwolfz/CCNN-CGH) for CCNN network implementation
