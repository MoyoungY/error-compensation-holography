import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.system import HoloSystem
from config.default import PhysicalConfig, TrainingConfig
from utils.utils import get_args
import torch

def main():
    # Get command line arguments
    args = get_args()
    
    # Initialize configs
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        scale_output=args.scale_output,
        train_root=args.train_root,
        valid_root=args.valid_root,
        exp_name=args.exp_name
    )
    phys_cfg = PhysicalConfig()
    
    # Initialize system
    system = HoloSystem(train_cfg, phys_cfg, channel=args.channel)
    
    if args.test:
        # Load checkpoint for testing
        if args.ckpt_path:
            print(f"Loading checkpoint from {args.ckpt_path}")
            checkpoint = torch.load(args.ckpt_path)
            system.load_state_dict(checkpoint['state_dict'])
        else:
            print("No checkpoint path provided. Using untrained model.")
        
        # Initialize trainer for testing
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            logger=False
        )
        
        # Run test
        trainer.test(system)
        
    else:
        # Setup training
        logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)
        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            save_top_k=1,
            mode='min',
            save_last=True,
            filename='{epoch}-{val_loss:.4f}'
        )
        
        # Initialize trainer for training
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            logger=logger,
            callbacks=[checkpoint_callback],
            accelerator='gpu',
            devices=1
        )
        
        # Run training
        trainer.fit(system)

if __name__ == '__main__':
    main()
