import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import imageio.v3 as iio
import utils.utils as utils

from utils.propagation_ASM import propagation_ASM
from utils.losses import PerceptualLoss
from utils.utils import crop_image
from config.default import PhysicalConfig, TrainingConfig
from models.holonet_unet import holonet
from models.ccnn import ccnncgh
from datasets.div2k import DIV2KDataset

class HoloSystem(pl.LightningModule):
    def __init__(self, cfg: TrainingConfig, phys_cfg: PhysicalConfig, channel: int = 1):
        super().__init__()
        self.cfg = cfg
        self.phys_cfg = phys_cfg
        self.channel = channel
        self.batch_size = cfg.batch_size
        
        # Initialize components
        self.setup_physical_params()
        self.setup_propagation()
        self.setup_model()
        self.setup_losses()
        self.setup_datasets()

    def setup_physical_params(self):
        """Setup physical parameters like wavelength, propagation distance etc."""
        self.wavelength = self.phys_cfg.WAVELENGTHS[self.channel]
        self.prop_dist = -20 * self.phys_cfg.CM
        self.pitch = self.phys_cfg.PIXEL_PITCH
        self.pad = True
        self.scale_output = self.cfg.scale_output
        self.homography_res = self.phys_cfg.IMAGE_RESOLUTION
        self.image_res = self.phys_cfg.IMAGE_RESOLUTION

    def setup_propagation(self):
        """Initialize propagation operators"""
        n, m = self.phys_cfg.IMAGE_RESOLUTION
        empty_field = torch.empty(self.batch_size, 1, n, m)
        
        self.Hforward = propagation_ASM(
            empty_field,
            feature_size=[self.pitch, self.pitch],
            wavelength=self.wavelength,
            z=-self.prop_dist,
            linear_conv=self.pad,
            return_H=True
        )
        
        self.Hbackward = propagation_ASM(
            empty_field,
            feature_size=[self.pitch, self.pitch],
            wavelength=self.wavelength,
            z=self.prop_dist,
            linear_conv=self.pad,
            return_H=True
        )

    def setup_model(self):
        """Initialize neural network model"""
        self.phase_generator = holonet()
        self.phase_generator = holonet()

    def setup_losses(self):
        """Setup loss functions"""
        self.pep_loss = PerceptualLoss(lambda_feat=0.025)
        self.mse_loss = nn.MSELoss()

    def setup_datasets(self):
        """Initialize training and validation datasets"""
        self.train_set = DIV2KDataset(
            self.cfg.train_root,
            channel=self.channel,
            split='train',
            image_res=self.image_res,
            homography_res=self.homography_res
        )
        
        self.val_set = DIV2KDataset(
            self.cfg.valid_root,
            channel=self.channel,
            split='val',
            image_res=self.image_res,
            homography_res=self.homography_res
        )

    def forward(self, target_amp):
        """Forward pass through the model"""
        return self.phase_generator(
            target_amp,
            self.Hforward
        )

    def get_reconstruction(self, holo_phase):
        """Calculate reconstruction from hologram phase"""
        slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
        recon_complex = propagation_ASM(
            u_in=slm_complex,
            feature_size=[self.pitch, self.pitch],
            wavelength=self.wavelength,
            z=self.prop_dist,
            linear_conv=self.pad,
            precomped_H=self.Hbackward
        )
        
        recon_amp = recon_complex.abs()
        recon_amp = torch.pow((recon_amp ** 2) * self.scale_output, 0.5)
        return recon_amp

    def compute_losses(self, recon_amp, target_amp, slm_cpx=None):
        """Compute all losses"""
        target_res = self.homography_res
        
        # Crop images to target resolution
        target_amp = crop_image(target_amp, target_res, stacked_complex=False)
        recon_amp = crop_image(recon_amp, target_res, stacked_complex=False)
        
        # Main losses
        loss = self.mse_loss(recon_amp, target_amp)
        loss += self.pep_loss(recon_amp.repeat(1, 3, 1, 1), target_amp.repeat(1, 3, 1, 1))
        
        # Additional reconstruction loss if slm_cpx provided
        if slm_cpx is not None:
            recon_field_2 = propagation_ASM(
                u_in=slm_cpx,
                feature_size=[self.pitch, self.pitch],
                wavelength=self.wavelength,
                z=self.prop_dist,
                linear_conv=self.pad,
                precomped_H=self.Hbackward
            )
            recon_field_2 = crop_image(recon_field_2, target_res, stacked_complex=False)
            recon_field_2 = recon_field_2.abs()
            recon_field_2 = torch.pow((recon_field_2 ** 2) * self.scale_output, 0.5)
            
            loss += 0.1 * self.mse_loss(recon_field_2, target_amp)
            loss += 0.1 * self.pep_loss(recon_field_2.repeat(1, 3, 1, 1), target_amp.repeat(1, 3, 1, 1))
            
        return loss

    def training_step(self, batch, batch_idx):
        target_amp = batch
        holo_phase, slm_cpx = self.forward(target_amp)
        
        recon_amp = self.get_reconstruction(holo_phase)
        loss = self.compute_losses(recon_amp, target_amp, slm_cpx)
        
        self.log('train/loss', loss)
        return {'loss': loss, 'log': {'train/loss': loss}}

    def validation_step(self, batch, batch_idx):
        target_amp = batch
        holo_phase, slm_cpx = self.forward(target_amp)
        
        recon_amp = self.get_reconstruction(holo_phase)
        loss = self.compute_losses(recon_amp, target_amp, slm_cpx)
        
        self.log('val/loss', loss, sync_dist=True)
        
        # Log images periodically
        if batch_idx == 22:
            self.logger.experiment.add_image('phases', (holo_phase[0, ...]+torch.pi)/(2*torch.pi), self.current_epoch)
            self.logger.experiment.add_image('images', recon_amp[0, ...], self.current_epoch)
            self.logger.experiment.add_image('target_img', target_amp[0, ...], self.current_epoch)
            
        return {'loss': loss, 'log': {'val/loss': loss}}

    def test_step(self, batch, batch_idx):
        """Test step for model evaluation"""
        target_amp = batch
        holo_phase, slm_cpx = self.forward(target_amp)
        
        recon_amp = self.get_reconstruction(holo_phase)
        loss = self.compute_losses(recon_amp, target_amp, slm_cpx)
        
        self.log('test/loss', loss, sync_dist=True)
        
        # Save results
        dir = os.path.join('results', self.cfg.exp_name)
        rgb_path = os.path.join(dir, 'rgb')
        phs_path = os.path.join(dir, 'phs')
        target_path = os.path.join(dir, 'target')
        
        # Create directories if they don't exist
        for path in [rgb_path, phs_path, target_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        # Save reconstructed amplitude
        iio.imwrite(
            os.path.join(rgb_path, f'{batch_idx:04d}.png'),
            (torch.clamp(recon_amp[0, 0, ...], min=0, max=1) * 255).round().cpu().numpy().astype(np.uint8)
        )
        
        # Save phase
        iio.imwrite(
            os.path.join(phs_path, f'{batch_idx:04d}.png'),
            utils.phasemap_8bit(holo_phase, inverted=True)
        )
        
        # Save target
        iio.imwrite(
            os.path.join(target_path, f'{batch_idx:04d}.png'),
            (torch.clamp(target_amp[0, 0, ...], min=0, max=1) * 255).round().cpu().numpy().astype(np.uint8)
        )
        
        metrics = {
            'test/loss': loss
        }
        
        self.log_dict(metrics, sync_dist=True)
        
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) 