import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import segmentation_models_pytorch as smp
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import wandb

Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
PATIENCE = 5
BATCH_SIZE = 4
LR = 1e-4

PATCH_DIR = Path("prime_patches")
VISUALS_DIR = Path("visual_samples")
MODEL_SAVE_PATH_TEMPLATE = "training_models/best_{model_name}.pth"

# Set seed
def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# Dataset
class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, size_filter=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_files = list(self.root_dir.glob("*.png"))
        if size_filter:
            self.image_files = [f for f in self.image_files if size_filter in f.name]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, img.clone()

# Model
def create_unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation="sigmoid"
    ).to(DEVICE)
    return model

# Mask generator
def generate_mask(batch_size, channels, height, width, mask_ratio=0.2):
    return (torch.rand(batch_size, channels, height, width) > (1 - mask_ratio)).float().to(DEVICE)

# Soft MCA Loss (training)
class SoftMeanClassAccuracyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets):
        preds = preds.clamp(1e-7, 1 - 1e-7)
        targets = targets.float()
        tp = (preds * targets).sum()
        tn = ((1 - preds) * (1 - targets)).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        acc_class1 = tp / (tp + fn + 1e-7)
        acc_class0 = tn / (tn + fp + 1e-7)
        mca = (acc_class1 + acc_class0) / 2.0
        return 1.0 - mca

# MCA Loss (eval only)
class MeanClassAccuracyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets):
        preds_bin = (preds > 0.5).float()
        targets = (targets > 0.5).float()
        tp = ((preds_bin == 1) & (targets == 1)).sum().float()
        tn = ((preds_bin == 0) & (targets == 0)).sum().float()
        fp = ((preds_bin == 1) & (targets == 0)).sum().float()
        fn = ((preds_bin == 0) & (targets == 1)).sum().float()
        acc_class1 = tp / (tp + fn + 1e-7)
        acc_class0 = tn / (tn + fp + 1e-7)
        return 1.0 - ((acc_class1 + acc_class0) / 2.0)

# Combined Loss (Soft MCA + BCE)
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.soft_mca = SoftMeanClassAccuracyLoss()
        self.bce = nn.BCELoss()
        self.alpha = alpha
        self.beta = beta
    def forward(self, preds, targets):
        targets = (targets > 0.5).float()
        return self.alpha * self.soft_mca(preds, targets) + self.beta * self.bce(preds, targets)

# Metrics
def calculate_pixel_accuracy(output, target):
    return ((output > 0.5).float() == target).float().mean()

# Training
def train_epoch(dataloader, model, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data, target in dataloader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        target = (target > 0.5).float()
        optimizer.zero_grad()
        mask = generate_mask(*data.shape)
        masked_input = data * mask
        output = model(masked_input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation
def validate_model(dataloader, model, model_name, epoch, loss_fn, eval_loss_fn):
    model.eval()
    total_loss = total_acc = total_ssim = total_eval_loss = 0
    total_mcacc = 0
    save_dir = VISUALS_DIR / f"{model_name}_val_samples"
    os.makedirs(save_dir, exist_ok=True)
    wandb_image = None
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = (target > 0.5).float()
            mask = generate_mask(*data.shape)
            masked_input = data * mask
            output = model(masked_input)

            loss = loss_fn(output, target)
            eval_loss = eval_loss_fn(output, target)

            total_loss += loss.item()
            total_eval_loss += eval_loss.item()
            total_acc += calculate_pixel_accuracy(output, target).item() * data.size(0)

            preds_bin = (output > 0.5).float()
            tp = ((preds_bin == 1) & (target == 1)).sum().float()
            tn = ((preds_bin == 0) & (target == 0)).sum().float()
            fp = ((preds_bin == 1) & (target == 0)).sum().float()
            fn = ((preds_bin == 0) & (target == 1)).sum().float()
            acc_class1 = tp / (tp + fn + 1e-7)
            acc_class0 = tn / (tn + fp + 1e-7)
            total_mcacc += ((acc_class1 + acc_class0) / 2.0).item() * data.size(0)

            for j in range(data.size(0)):
                total_ssim += ssim(
                    (preds_bin[j,0]).cpu().numpy(), 
                    target[j,0].cpu().numpy(), 
                    data_range=1.0
                )

            if i == 0:
                img_path = save_dir / f"sample_epoch_{epoch+1}.png"
                save_image(
                    torch.cat([target[0], masked_input[0], preds_bin[0]], dim=2),
                    img_path
                )
                wandb_image = wandb.Image(str(img_path), caption=f"Epoch {epoch+1}")

    avg_loss = total_loss / len(dataloader)
    avg_eval_loss = total_eval_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader.dataset)
    avg_mcacc = total_mcacc / len(dataloader.dataset)
    avg_ssim = total_ssim / len(dataloader.dataset)
    return avg_loss, avg_eval_loss, avg_acc, avg_mcacc, avg_ssim, wandb_image

# Training Loop
def run_training_pipeline(model, train_loader, val_loader, optimizer, scheduler, model_name, loss_fn, eval_loss_fn):
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_eval_loss, val_acc, val_mcacc, val_ssim, val_image = validate_model(val_loader, model, model_name, epoch, loss_fn, eval_loss_fn)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Eval Loss: {val_eval_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"mCAcc: {val_mcacc*100:.2f}% | "
              f"Val SSIM: {val_ssim:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_eval_loss": val_eval_loss,
            "val_accuracy": val_acc,
            "val_mCAcc": val_mcacc * 100,
            "val_ssim": val_ssim,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "validation_sample": val_image
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = MODEL_SAVE_PATH_TEMPLATE.format(model_name=model_name)
            torch.save(model.state_dict(), save_path)
            print(f"   -> Best model saved in '{save_path}'")

            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(local_path=save_path)
            wandb.log_artifact(artifact, aliases=["best"])
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   -> Early stopping in epoch {epoch+1}")
                break  # Comment to stop training early
    print(f"\nTraining completed for '{model_name}'. Best model has been saved.")

# Main
if __name__ == "__main__":
    set_seeds(42)
    print(f"Using device: {DEVICE}")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Change this to 25,000,000; 50,000,000; 100,000,000; 200,000,000
    size_str = "200,000,000"
    
    config = {
        "dataset_size": size_str,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "architecture": "U-Net with ResNet34",
        "loss_function": "CombinedLoss (SoftMCA + BCE, alpha=1.0, beta=0.5)"
    }

    model_name = f"model_{size_str.replace(',', '')}"
    
    wandb.init(
        project="prime-spiral-inpainting",
        config=config,
        name=f"train_{model_name}"
    )
    
    print(f"\n{'='*20} Starting training for {size_str} {'='*20}")
    model = create_unet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    loss_fn = CombinedLoss(alpha=1.0, beta=0.5)
    eval_loss_fn = MeanClassAccuracyLoss()
    train_data = SimpleImageDataset(PATCH_DIR / "train", transform, size_str)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data = SimpleImageDataset(PATCH_DIR / "test", transform, size_str)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    if len(train_data) == 0 or len(val_data) == 0:
        print(f"Data can't be found for size {size_str}. Skipping")
    else:
        wandb.watch(model, log="all", log_freq=100)

        run_training_pipeline(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model_name=model_name,
            loss_fn=loss_fn,
            eval_loss_fn=eval_loss_fn
        )
    print("\nSingle-size training completed")

    wandb.finish()