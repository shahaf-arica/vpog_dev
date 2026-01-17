#!/usr/bin/env python3
"""
VPOG Training Script

Trains the VPOG model for 6D object pose estimation using template-based matching.

Usage:
    # Local training (single GPU)
    python training/train.py machine=local

    # Local training with specific GPU
    CUDA_VISIBLE_DEVICES=0 python training/train.py machine=local

    # Multi-GPU local training
    python training/train.py machine=local machine.trainer.devices=2

    # SLURM cluster training
    sbatch training/scripts/submit_slurm.sh

    # Override config parameters
    python training/train.py machine=local max_epochs=50 machine.batch_size=8

    # Resume from checkpoint
    python training/train.py resume_from_checkpoint=/path/to/checkpoint.ckpt

For more details, see training/TRAINING_README.md
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_logger(name: str):
    """Simple logger."""
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


def setup_callbacks(cfg: DictConfig):
    """Setup training callbacks from config.
    
    Uses Hydra's instantiate to create callbacks from the machine config.
    Automatically selects appropriate checkpoint strategy based on validation.
    """
    from hydra.utils import instantiate
    
    callbacks = []
    has_validation = cfg.get('val_check_interval') is not None
    
    if 'callbacks' in cfg.machine:
        for callback_name, callback_cfg in cfg.machine.callbacks.items():
            if callback_cfg is not None and '_target_' in callback_cfg:
                # Smart checkpoint selection
                if callback_name == 'checkpoint':
                    if has_validation:
                        # Use validation-based checkpoint
                        callback = instantiate(callback_cfg)
                        logger.info(f"Checkpoint: save top-{callback_cfg.save_top_k} by {callback_cfg.monitor}")
                    else:
                        # Skip this one, will use checkpoint_no_val
                        continue
                
                elif callback_name == 'checkpoint_no_val':
                    if not has_validation:
                        # Use epoch-based checkpoint
                        callback = instantiate(callback_cfg)
                        logger.info(f"Checkpoint: save every {callback_cfg.every_n_epochs} epoch(s) (no validation)")
                    else:
                        # Skip this one, using regular checkpoint
                        continue
                
                else:
                    # Other callbacks (lr_monitor, etc.)
                    callback = instantiate(callback_cfg)
                
                callbacks.append(callback)
                logger.info(f"Added callback: {callback_name} -> {callback.__class__.__name__}")
    
    if not callbacks:
        logger.warning("No callbacks configured! Using defaults.")
        callbacks.append(ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            save_last=True,
            every_n_epochs=1,
        ))
    
    return callbacks


def setup_logger_backend(cfg: DictConfig):
    """Setup logging backend (TensorBoard or WandB).
    
    Logger controlled by train.enable_wandb flag (master switch).
    All outputs go to cfg.save_dir/cfg.name_exp/
    """
    enable_wandb = cfg.get('enable_wandb', False)
    
    if enable_wandb:
        # WandB logger
        import wandb
        
        # Login with API key from user config
        if cfg.user.wandb_api_key:
            wandb.login(key=str(cfg.user.wandb_api_key))
        
        # Get project name from user config
        project_name = cfg.user.wandb_project_name
        
        # Set offline mode if dryrun
        if cfg.machine.get('dryrun', False):
            os.environ['WANDB_MODE'] = 'offline'
            logger.info("WandB in offline mode (dryrun=True)")
        
        wandb_logger = WandbLogger(
            name=cfg.name_exp,
            project=project_name,
            save_dir=str(Path(cfg.save_dir) / cfg.name_exp),
            log_model=False,
        )
        logger.info(f"Using WandB logger: project={project_name}, name={cfg.name_exp}")
        logger.info(f"WandB logs: {Path(cfg.save_dir) / cfg.name_exp / 'wandb'}")
        return wandb_logger
    else:
        # TensorBoard logger (default)
        tensorboard_dir = Path(cfg.save_dir) / cfg.name_exp / 'tensorboard'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        tb_logger = TensorBoardLogger(
            save_dir=str(tensorboard_dir.parent),
            name='tensorboard',
            version='',
        )
        logger.info(f"Using TensorBoard logger: {tensorboard_dir}")
        return tb_logger


def setup_trainer(cfg: DictConfig, callbacks, logger_backend):
    """Setup PyTorch Lightning trainer using Hydra instantiation.
    
    The trainer config is defined in machine.trainer (e.g., local.yaml, slurm.yaml).
    This function instantiates the trainer and adds runtime overrides.
    """
    from hydra.utils import instantiate
    
    # Get trainer config from machine settings
    trainer_cfg = OmegaConf.to_container(cfg.machine.trainer, resolve=True)
    
    # Runtime overrides that can't be in YAML
    trainer_cfg['callbacks'] = callbacks
    trainer_cfg['logger'] = logger_backend
    
    # Override progress bar based on dryrun mode
    if cfg.machine.get('dryrun', False):
        trainer_cfg['enable_progress_bar'] = False
    
    # Handle SLURM environment variables (override config if SLURM sets them)
    if cfg.machine.name == 'slurm':
        slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE')
        slurm_nodes = os.environ.get('SLURM_NNODES')
        
        if slurm_gpus:
            trainer_cfg['devices'] = int(slurm_gpus)
            logger.info(f"SLURM override: devices={slurm_gpus}")
        if slurm_nodes:
            trainer_cfg['num_nodes'] = int(slurm_nodes)
            logger.info(f"SLURM override: num_nodes={slurm_nodes}")
        
        logger.info(f"SLURM training: {trainer_cfg['devices']} GPUs × {trainer_cfg['num_nodes']} nodes")
    else:
        logger.info(f"Local training: {trainer_cfg.get('devices', 1)} GPU(s)")
    
    # Add optional trainer parameters from root config
    if cfg.get('limit_train_batches'):
        trainer_cfg['limit_train_batches'] = cfg.limit_train_batches
    if cfg.get('limit_val_batches'):
        trainer_cfg['limit_val_batches'] = cfg.limit_val_batches
    
    # Instantiate trainer
    trainer = instantiate(trainer_cfg)
    
    logger.info(f"Trainer config: accelerator={trainer_cfg.get('accelerator')}, "
                f"devices={trainer_cfg.get('devices')}, "
                f"precision={trainer_cfg.get('precision')}, "
                f"strategy={trainer_cfg.get('strategy', 'auto')}")
    
    return trainer


def setup_dataloaders(cfg: DictConfig):
    """Setup train and validation data loaders following gigapose pattern.
    
    Training:
    - Use train_dataset_id to select dataset(s): 0=gso, 1=shapenet, 2=both
    - Iterate over list and create dataloader for each dataset
    - Skip if validate_only=True to save time
    
    Validation:
    - Use generic BOP config with dataset_name override
    
    Returns:
        train_dataloaders: List of training dataloaders (one or multiple) or None if validate_only
        val_dataloader: Validation dataloader (optional)
    """
    from hydra.utils import instantiate
    # from torch.utils.data import DataLoader
    from src.utils.dataloader import NoneFilteringDataLoader
    
    batch_size = cfg.machine.batch_size
    num_workers = cfg.machine.num_workers
    
    # --- Training Dataloaders (skip if validate_only) ---
    train_dataloaders = []
    
    if not cfg.get('validate_only', False):
        selected_train_dataset_names = cfg.train_dataset_names[cfg.train_dataset_id]
        logger.info(f"Training datasets: {selected_train_dataset_names}")
        
        for dataset_name in selected_train_dataset_names:
            logger.info(f"\n  Loading training dataset: {dataset_name}")
            
            # Set dataset name in the config
            cfg.data_train.dataloader.dataset_name = dataset_name
            dataset = instantiate(cfg.data_train.dataloader)
            
            dataloader = NoneFilteringDataLoader(
                dataset.web_dataloader,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=True,
                drop_last=True,
                persistent_workers=num_workers > 0,
            )
            
            train_dataloaders.append(dataloader)
            # logger.info(f"    Loaded {len(dataset)} samples")
    else:
        logger.info("Skipping training dataset loading (validate_only=True)")
        train_dataloaders = None
    
    # --- Validation Dataloader (BOP dataset) ---
    val_dataloader = None
    
    # Load validation if: val_check_interval is set OR validate_only mode
    if cfg.get('val_check_interval') or cfg.get('validate_only', False):
        val_batch_size = cfg.machine.get('val_batch_size', 1)
        logger.info(f"\nLoading validation dataset: {cfg.val_dataset_name}")
        # logger.info(f"  Validation batch size: {val_batch_size}")
        
        # Set validation dataset name
        cfg.data_val.dataloader.dataset_name = cfg.val_dataset_name
        val_dataset = instantiate(cfg.data_val.dataloader)
        
        val_dataloader = NoneFilteringDataLoader(
            val_dataset.web_dataloader,
            batch_size=val_batch_size,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
        
        # logger.info(f"  Loaded {len(val_dataset)} validation samples")

    return train_dataloaders, val_dataloader


def build_vpog_model_from_config(cfg: DictConfig):
    """Build VPOG model from config by constructing all components.
    
    VPOGModel requires pre-built components:
    - encoder: CroCo ViT encoder
    - aa_module: Alternating Attention module
    - classification_head: Patch classification head
    - flow_head: Dense flow prediction head
    
    Args:
        cfg: Model configuration
        
    Returns:
        VPOGModel instance
    """
    from hydra.utils import instantiate
    from vpog.models.vpog_model import VPOGModel
    
    logger.info("Building VPOG model components...")
    
    # All components should be instantiated from config
    # Each component's config should have _target_ pointing to the class
    
    # Build encoder from config
    logger.info("  Building encoder...")
    encoder = instantiate(cfg.model.encoder)
    
    # Build AA module from config
    logger.info("  Building AA module...")
    aa_module = instantiate(cfg.model.aa_module)
    
    # Build classification head from config
    logger.info("  Building classification head...")
    classification_head = instantiate(cfg.model.classification_head)
    
    # Build flow head from config
    logger.info("  Building flow heads...")
    
    # center_flow_head = instantiate(cfg.model.center_flow_head) # currently unused
    dense_flow_head = instantiate(cfg.model.dense_flow_head) 
    
    # Assemble VPOGModel
    logger.info("  Assembling VPOG model...")
    vpog_model = VPOGModel(
        encoder=encoder,
        aa_module=aa_module,
        classification_head=classification_head,
        # center_flow_head=center_flow_head,
        dense_flow_head=dense_flow_head,
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
    )
    
    return vpog_model

def model_param_counts(model: torch.nn.Module):
    """Utility to count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def create_vpog_model(cfg: DictConfig):
    """Setup VPOG model and lightning module using config-based building."""
    from hydra.utils import instantiate
    from training.lightning_module import VPOGLightningModule, LossWeights
    
    logger.info("Initializing VPOG model...")
    
    # Build VPOG model from config
    vpog_model = build_vpog_model_from_config(cfg)

    # total_params, trainable_params = model_param_counts(vpog_model)
    # logger.info(f"  Loaded model parameters: {total_params / 1e6:.2f}M")
    # logger.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Loss configurations from YAML
    loss_weights = LossWeights(
        cls=cfg.loss.weights.cls,
        dense=cfg.loss.weights.dense,
        invis_reg=cfg.loss.weights.invis_reg,
        center=cfg.loss.weights.center,
    )
    
    dense_loss_cfg = instantiate(cfg.loss.dense_flow)
    center_loss_cfg = instantiate(cfg.loss.center_flow)
    cls_loss_cfg = instantiate(cfg.loss.classification)
    
    # Get optimizer config
    optimizer_cfg = cfg.get('optimizer', {})

    # create trainiglightning module losses
    dense_flow_loss = instantiate(cfg.loss.dense_flow)
    # center_flow_loss = instantiate(cfg.loss.center_flow)
    classification_loss = instantiate(cfg.loss.classification)
    
    # Create Lightning module with optimizer parameters
    lightning_module = VPOGLightningModule(
        model=vpog_model,
        cls_loss=classification_loss,
        dense_flow_loss=dense_flow_loss,
        # center_flow_loss=center_flow_loss,
        lr=optimizer_cfg.get('lr', 1e-4),
        weight_decay=optimizer_cfg.get('weight_decay', 1e-2),
        betas=optimizer_cfg.get('betas', (0.9, 0.999)),
        loss_weights=loss_weights,
        log_every_n_steps=cfg.log_every_n_steps,
        # Parameter groups for different learning rates
        use_param_groups=optimizer_cfg.get('use_param_groups', False),
        backbone_lr_multiplier=optimizer_cfg.get('param_groups', {}).get('backbone', {}).get('lr_multiplier', 0.1),
        new_components_lr_multiplier=optimizer_cfg.get('param_groups', {}).get('new_components', {}).get('lr_multiplier', 1.0),
        # BOP evaluation parameters (uses val_dataset_name from train.yaml)
        enable_pose_eval=cfg.model.get('enable_pose_eval', True),
        dataset_name=cfg.val_dataset_name,
    )
    
    logger.info(f"  Model parameters: {sum(p.numel() for p in vpog_model.parameters()) / 1e6:.2f}M")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in vpog_model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Log optimizer settings
    if optimizer_cfg.get('use_param_groups'):
        logger.info(f"  Using parameter groups:")
        logger.info(f"    Backbone LR multiplier: {optimizer_cfg.get('param_groups', {}).get('backbone', {}).get('lr_multiplier', 0.1)}")
        logger.info(f"    New components LR multiplier: {optimizer_cfg.get('param_groups', {}).get('new_components', {}).get('lr_multiplier', 1.0)}")
    
    return lightning_module


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """Main training function."""
    
    # Set struct to allow adding new fields
    OmegaConf.set_struct(cfg, False)
    
    # Print configuration
    logger.info("="*80)
    logger.info("VPOG Training")
    logger.info("="*80)
    logger.info(f"\nExperiment: {cfg.name_exp}")
    logger.info(f"Save directory: {cfg.save_dir}")
    logger.info(f"Checkpoint directory: {cfg.checkpoint_dir}")
    logger.info(f"\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    pl.seed_everything(cfg.get('seed', cfg.seed), workers=True)
    logger.info(f"Random seed: {cfg.get('seed', cfg.seed)}")
    
    # Create directories
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if cfg.get('vis_dir'):
        Path(cfg.vis_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup components
    logger.info("\n" + "="*80)
    logger.info("Setup")
    logger.info("="*80)
    
    callbacks = setup_callbacks(cfg)
    logger_backend = setup_logger_backend(cfg)
    trainer = setup_trainer(cfg, callbacks, logger_backend)
    
    # Check if validation-only mode BEFORE loading datasets
    if cfg.get('validate_only', False):
        logger.info("\n" + "="*80)
        logger.info("Validation-Only Mode - Loading only validation dataset")
        logger.info("="*80)
        
        # Only load validation dataloader
        train_dataloaders = None
        _, val_dataloader = setup_dataloaders(cfg)
        
        # Setup model
        model = create_vpog_model(cfg)
        if val_dataloader is None:
            logger.error("Cannot run validation-only mode without validation dataloader!")
            logger.error("Set val_check_interval to enable validation.")
            return
        
        ckpt_path = cfg.get('resume_from_checkpoint')
        if ckpt_path:
            logger.info(f"Loading checkpoint: {ckpt_path}")
        else:
            logger.warning("No checkpoint specified - validating with random initialized weights!")
            logger.warning("Use: resume_from_checkpoint=/path/to/checkpoint.ckpt")
        
        trainer.validate(model, dataloaders=val_dataloader, ckpt_path=ckpt_path)
        
        logger.info("\n" + "="*80)
        logger.info("Validation completed!")
        logger.info("="*80)
        return
    
    # Normal training mode - load all datasets
    logger.info("\n" + "="*80)
    logger.info("Training Mode - Loading training datasets")
    logger.info("="*80)
    
    # Setup dataloaders (supports multi-dataset training)
    train_dataloaders, val_dataloader = setup_dataloaders(cfg)
    
    # Setup model
    model = create_vpog_model(cfg)
    
    # Start training
    logger.info("\n" + "="*80)
    logger.info("Training")
    logger.info("="*80)
    
    # Run full validation before training starts (if enabled)
    if val_dataloader is not None:
        logger.info("\n" + "="*80)
        logger.info("Running initial validation...")
        logger.info("="*80)
        trainer.validate(model, dataloaders=val_dataloader)
    
    try:
        trainer.fit(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.get('resume_from_checkpoint'),
        )
        
        logger.info("\n" + "="*80)
        logger.info("Training completed successfully!")
        logger.info("="*80)
        logger.info(f"Checkpoints saved to: {cfg.checkpoint_dir}")
        logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        
    except KeyboardInterrupt:
        logger.info("\n" + "="*80)
        logger.info("Training interrupted by user")
        logger.info("="*80)
        logger.info(f"Last checkpoint: {cfg.checkpoint_dir}/last.ckpt")
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("Training failed!")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
