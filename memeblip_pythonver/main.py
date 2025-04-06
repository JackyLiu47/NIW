import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse

# Import from memeblip_pythonver folder
from memeBlip import MemeBLIP, initialize_weights
from Config import Config
from CachedDataset import CachedDataset

def main():
    # Load configuration
    cfg = Config()
    print(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--validate", action="store_true", help="Validate the model")
    parser.add_argument("--ckpt-path", type=str, help="Path to a checkpoint file")
    parser.add_argument("--no-adapter", action="store_true", help="Disable adapter")
    parser.add_argument("--no-text-proj", action="store_true", help="Disable text projection")
    parser.add_argument("--no-image-proj", action="store_true", help="Disable image projection")
    parser.add_argument("--no-pre-output-layer", action="store_true", help="Disable pre-output layer")
    parser.add_argument("--no-cosine-classifier", action="store_true", help="Disable cosine classifier")

    args = parser.parse_args()
    
    # Setup data loaders
    train_loader = DataLoader(CachedDataset(cfg.trainData_path), batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(CachedDataset(cfg.valData_path), batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_path,
        filename="memeBLIP-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        verbose=True,
        mode="min"
    )

    # Initialize model
    if args.ckpt_path:
        print("Loading model from checkpoint:", args.ckpt_path)
        model = MemeBLIP.load_from_checkpoint(args.ckpt_path, cfg=cfg, map_location="cuda:0", use_adapter=not args.no_adapter,
            use_text_project=not args.no_text_proj, use_image_project=not args.no_image_proj, use_pre_output_layer =not args.no_pre_output_layer ,
            use_cosine_classifier=not args.no_cosine_classifier, strict=False
        )
    else:
        print("Initializing new model")
        model = MemeBLIP(cfg, use_adapter=not args.no_adapter,
            use_text_project=not args.no_text_proj, use_image_project=not args.no_image_proj, use_pre_output_layer=not args.no_pre_output_layer,
            use_cosine_classifier=not args.no_cosine_classifier)
        model.register_hooks()
        model.apply(initialize_weights)
        
    model.to(cfg.device)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        precision=16,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        devices=len(cfg.gpus),
        logger=pl.loggers.TensorBoardLogger("logs/"),
        callbacks=[early_stop_callback]
    )

    # Setup tensorboard writer for gradient logging
    writer = SummaryWriter(log_dir='logs/gradient_logs')
    
    # Train and validate model
    if args.train:
        print("Training model")
        trainer.fit(model, train_loader, val_loader)

    if args.validate:
        print("Validating model")
        validation_metrics = trainer.validate(model, val_loader, verbose=True)
        print("Validation Metrics:", validation_metrics)
        print("Validation Accuracy:", float(trainer.callback_metrics["val_acc"]))
        print("Validation AUROC:", float(trainer.callback_metrics["val_auroc"]))
        print("Validation F1 Score:", float(trainer.callback_metrics["val_f1"]))

if __name__ == "__main__":
    main()