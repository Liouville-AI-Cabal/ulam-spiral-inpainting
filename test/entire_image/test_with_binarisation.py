import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None

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
        original_name = img_path.name
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, img.clone(), original_name

def create_unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation="sigmoid"
    ).to(DEVICE)
    return model

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
    
def binarize_top_k(tensor, k_percent=6.0):
    total_pixels = tensor.numel() 
    k = int(total_pixels * (k_percent / 100.0))

    if k == 0:
        return torch.zeros_like(tensor)

    flat_tensor = tensor.flatten()

    threshold_value = torch.topk(flat_tensor, k).values[-1]

    return (tensor >= threshold_value).float()
    
def calculate_metrics_from_counts_list(counts_list):
    total_tp = sum(c['tp'] for c in counts_list)
    total_tn = sum(c['tn'] for c in counts_list)
    total_fp = sum(c['fp'] for c in counts_list)
    total_fn = sum(c['fn'] for c in counts_list)
    total_pixels = total_tp + total_tn + total_fp + total_fn

    accuracy = (total_tp + total_tn) / total_pixels if total_pixels > 0 else 0
    
    precision_1 = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall_1 = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    precision_0 = total_tn / (total_tn + total_fn) if (total_tn + total_fn) > 0 else 0
    recall_0 = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    balanced_acc = (recall_0 + recall_1) / 2.0
    
    class_0_percent = ((total_tn + total_fp) / total_pixels) * 100 if total_pixels > 0 else 0
    class_1_percent = ((total_tp + total_fn) / total_pixels) * 100 if total_pixels > 0 else 0

    macro_f1 = (f1_0 + f1_1) / 2.0

    return {
        "Overall Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Class 1 Precision": precision_1, "Class 1 Recall": recall_1, "Class 1 F1": f1_1,
        "Class 0 Precision": precision_0, "Class 0 Recall": recall_0, "Class 0 F1": f1_0,
        "Macro F1": macro_f1,
        "Confusion Matrix": {"tp": total_tp, "tn": total_tn, "fp": total_fp, "fn": total_fn},
        "Class 0 Percent": class_0_percent, "Class 1 Percent": class_1_percent
    }

def calculate_detailed_metrics_per_image(pred_binary, target):
    pred_binary = pred_binary.bool()
    target = target.bool()

    tp = (pred_binary & target).sum().item()
    tn = (~pred_binary & ~target).sum().item()
    fp = (pred_binary & ~target).sum().item()
    fn = (~pred_binary & target).sum().item()

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

def calculate_detailed_metrics(pred_binary, target):
    pred_binary = pred_binary.bool()
    target = target.bool()

    tp = (pred_binary & target).sum().item()
    tn = (~pred_binary & ~target).sum().item()
    fp = (pred_binary & ~target).sum().item()
    fn = (~pred_binary & target).sum().item()

    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    balanced_acc = (recall_0 + recall_1) / 2.0 if (recall_0 + recall_1) > 0 else 0
    
    total_pixels = tp + tn + fp + fn
    class_0_percent = ((tn + fp) / total_pixels) * 100 if total_pixels > 0 else 0
    class_1_percent = ((tp + fn) / total_pixels) * 100 if total_pixels > 0 else 0

    metrics = {
        "accuracy": accuracy * 100,
        "balanced_acc": balanced_acc * 100,
        "conf_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "class_1": {
            "precision": precision_1 * 100,
            "recall": recall_1 * 100,
            "f1": f1_1 * 100,
            "percent": class_1_percent
        },
        "class_0": {
            "precision": precision_0 * 100,
            "recall": recall_0 * 100,
            "f1": f1_0 * 100,
            "percent": class_0_percent
        },
    }
    return {"conf_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}} 

def create_error_map(pred_binary, target):
    pred_np = pred_binary.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    h, w = target_np.shape
    error_map_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    error_map_rgb[(pred_np == 0) & (target_np == 0)] = [20, 20, 20]
    error_map_rgb[(pred_np == 1) & (target_np == 1)] = [255, 255, 255]
    error_map_rgb[(pred_np == 1) & (target_np == 0)] = [255, 0, 0]
    error_map_rgb[(pred_np == 0) & (target_np == 1)] = [0, 0, 255]
    
    return Image.fromarray(error_map_rgb)

def save_comparison_image(images, metrics, save_path):
    titles = ["Masked Input", "Inpainting Result", "Error Map", "Original Image"]
    width, height = images[0].size
    
    panel_width = width
    panels_total_width = panel_width * len(images)
    sidebar_width = 350
    title_height = 30

    try:
        font_size = 8
        font = ImageFont.truetype("arial.ttf", font_size)
        font_bold = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        font_bold = font

    estimated_lines = 38
    line_height = font_size + 4
    needed_sidebar_height = estimated_lines * line_height + 40

    total_height = max(height + title_height, needed_sidebar_height)

    canvas = Image.new('RGB', (panels_total_width + sidebar_width, total_height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for i, img in enumerate(images):
        if img.mode == 'L':
            img = img.convert('RGB')
        canvas.paste(img, (i * panel_width, title_height))
        text_bbox = draw.textbbox((0,0), titles[i], font=font_bold)
        text_w = text_bbox[2] - text_bbox[0]
        draw.text((i * panel_width + (panel_width - text_w) // 2, 5), titles[i], font=font_bold, fill=(255, 255, 255))

    sidebar_x0 = panels_total_width
    sidebar_x1 = panels_total_width + sidebar_width
    draw.rectangle([sidebar_x0, 0, sidebar_x1, total_height], fill=(15, 15, 15))

    x_pos, y_pos = panels_total_width + 15, title_height
    draw.text((x_pos, y_pos), "Inpainting Performance Analysis", font=font_bold, fill=(255,255,0))
    y_pos += 25
    draw.line([(x_pos-5, y_pos-5), (sidebar_x1-10, y_pos-5)], fill=(80,80,80), width=1)

    draw.text((x_pos, y_pos), "Pixel Class Distribution:", font=font_bold, fill=(255,255,255))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  Black (0): {metrics['class_0']['percent']:.1f}%", font=font, fill=(200,200,200))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  White (1): {metrics['class_1']['percent']:.1f}%", font=font, fill=(200,200,200))
    y_pos += 25

    draw.text((x_pos, y_pos), f"Overall Accuracy: {metrics['accuracy']:.2f}%", font=font_bold, fill=(255,255,255))
    y_pos += 18
    draw.text((x_pos, y_pos), f"Balanced Accuracy (mCAcc): {metrics['balanced_acc']:.2f}%", font=font_bold, fill=(255,255,255))
    y_pos += 25

    draw.text((x_pos, y_pos), "Black Pixels (0)", font=font_bold, fill=(255,255,255))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  Precision: {metrics['class_0']['precision']:.2f}%", font=font, fill=(200,200,200))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  Recall: {metrics['class_0']['recall']:.2f}%", font=font, fill=(200,200,200))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  F1-Score: {metrics['class_0']['f1']:.2f}%", font=font, fill=(200,200,200))
    y_pos += 25

    draw.text((x_pos, y_pos), "White Pixels (1)", font=font_bold, fill=(255,255,255))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  Precision: {metrics['class_1']['precision']:.2f}%", font=font, fill=(200,200,200))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  Recall: {metrics['class_1']['recall']:.2f}%", font=font, fill=(200,200,200))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  F1-Score: {metrics['class_1']['f1']:.2f}%", font=font, fill=(200,200,200))
    y_pos += 25

    cm = metrics['conf_matrix']
    draw.text((x_pos, y_pos), "Confusion Matrix", font=font_bold, fill=(255,255,255))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  TN: {cm['tn']:,}", font=font, fill=(200,200,200))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  FP: {cm['fp']:,}", font=font, fill=(200,200,200))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  FN: {cm['fn']:,}", font=font, fill=(200,200,200))
    y_pos += 18
    draw.text((x_pos, y_pos), f"  TP: {cm['tp']:,}", font=font, fill=(200,200,200))

    y_pos += 25
    draw.text((x_pos, y_pos), "Error Map Legend:", font=font_bold, fill=(255,255,255))
    y_pos += 18
    draw.rectangle([(x_pos, y_pos), (x_pos+12, y_pos+12)], fill=(20,20,20))
    draw.text((x_pos+18, y_pos), "TN (Correct Black)", font=font, fill=(200,200,200))
    y_pos += 18
    draw.rectangle([(x_pos, y_pos), (x_pos+12, y_pos+12)], fill=(255,255,255))
    draw.text((x_pos+18, y_pos), "TP (Correct White)", font=font, fill=(200,200,200))
    y_pos += 18
    draw.rectangle([(x_pos, y_pos), (x_pos+12, y_pos+12)], fill=(255,0,0))
    draw.text((x_pos+18, y_pos), "FP (Wrong White)", font=font, fill=(200,200,200))
    y_pos += 18
    draw.rectangle([(x_pos, y_pos), (x_pos+12, y_pos+12)], fill=(0,0,255))
    draw.text((x_pos+18, y_pos), "FN (Wrong Black)", font=font, fill=(200,200,200))

    canvas.save(save_path)

def generate_mask(batch_size, channels, height, width, mask_ratio=0.2, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    mask = (torch.rand(batch_size, channels, height, width) > (1 - mask_ratio)).float().to(DEVICE)
    if seed is not None:
        torch.seed()
    return mask

def test_model(model, dataloader, model_name="", mask_ratio=0.2, R=1000, ci=95):
    model.eval()
    
    save_dir = Path("visual_samples") / f"{model_name}_TEST_RESULTS"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Visualisation result will be saved in: {save_dir}")

    # --- Step 1: Run prediction once & collect statistics per image ---
    per_image_counts = []
    print("\nStep 1: Running predictions for all test images...")
    with torch.no_grad():
        for i, (data, target, original_names) in enumerate(tqdm(dataloader, desc="Prediksi")):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            mask = generate_mask(*data.shape, mask_ratio=mask_ratio, seed=i)
            masked_input = data * mask
            
            output = model(masked_input)
            # output_binary = (output > 0.5).float()

            binarized_images_list = []

            for single_image_output in output:
                binarized_image = binarize_top_k(single_image_output, k_percent=6.0)
                binarized_images_list.append(binarized_image)

            output_binary = torch.stack(binarized_images_list)

            for j in range(data.size(0)):
                # Count statistics for this single image and save to list
                stats = calculate_detailed_metrics_per_image(output_binary[j], target[j])
                per_image_counts.append(stats)

    # --- Step 2: Calculate original metrics (point estimate) from all data ---
    print("\nStep 2: Calculating original metrics (point estimate)...")
    point_estimates = calculate_metrics_from_counts_list(per_image_counts)

    # --- Step 3: Run bootstrapping simulation ---
    print(f"\nStep 3: Running {R} bootstrapping simulations...")
    bootstrap_metrics_results = []
    n_images = len(per_image_counts)
    
    for _ in tqdm(range(R), desc="Bootstrapping"):
        indices = np.random.choice(n_images, size=n_images, replace=True)
        resampled_counts = [per_image_counts[i] for i in indices]
        sample_metrics = calculate_metrics_from_counts_list(resampled_counts)
        bootstrap_metrics_results.append(sample_metrics)

    # --- Step 4: Calculate confidence intervals and prepare report ---
    print("\nStep 4: Calculating confidence intervals (Confidence Intervals)...")
    results_summary = []
    lo, hi = (100 - ci) / 2, 100 - ((100 - ci) / 2)

    for metric_name in point_estimates:
        if metric_name == "Confusion Matrix" or "Percent" in metric_name:
            continue
        
        point_estimate = point_estimates[metric_name]
        bootstrap_distribution = [res[metric_name] for res in bootstrap_metrics_results]
        ci_low = np.percentile(bootstrap_distribution, lo)
        ci_high = np.percentile(bootstrap_distribution, hi)
        
        results_summary.append({
            "Metric": metric_name, 
            "Point Estimate": point_estimate,
            f"CI {ci}% Low": ci_low, 
            f"CI {ci}% High": ci_high
        })

    # --- Step 5: Print and save final report ---
    final_report_path = save_dir / f"report_{model_name}.txt"
    pe = point_estimates
    cm = pe["Confusion Matrix"]

    micro_f1_res = next((item for item in results_summary if item["Metric"] == "Overall Accuracy"), None)
    macro_f1_res = next((item for item in results_summary if item["Metric"] == "Macro F1"), None)

    report_content = []
    report_content.append("="*60)
    report_content.append(f"FINAL RESULTS for {model_name}")
    report_content.append("="*60)
    report_content.append(f"Total images evaluated: {n_images}")
    report_content.append(f"Number of simulations: {R}")
    report_content.append(f"Confidence Level: {ci}%")
    # This section still uses percent because it's more intuitive
    report_content.append("\n--- Class Distribution (from entire dataset) ---")
    report_content.append(f"  - Class 0 (Black): {pe['Class 0 Percent']:.2f}%")
    report_content.append(f"  - Class 1 (White): {pe['Class 1 Percent']:.2f}%")

    report_content.append("\n--- Class Performance Metrics ---")
    for res in results_summary:
        if "Class" in res['Metric']:
            line = (f"  {res['Metric']:<20}: {res['Point Estimate']} "
                    f"(CI {ci}%: [{res[f'CI {ci}% Low']} - {res[f'CI {ci}% High']}])")
            report_content.append(line)

    report_content.append("\n--- Summary Metrics (Micro & Macro) ---")
    if micro_f1_res:
        line = (f"  {'Overall Accuracy/Micro F1':<20}: {micro_f1_res['Point Estimate']}")
        report_content.append(line)
        ci_line = (f"  {'Micro CI':<20}: [{micro_f1_res[f'CI {ci}% Low']} - {micro_f1_res[f'CI {ci}% High']}]")
        report_content.append(ci_line)
    if macro_f1_res:
        line = (f"  {'Macro F1':<20}: {macro_f1_res['Point Estimate']}")
        report_content.append(line)
        ci_line = (f"  {'Macro CI':<20}: [{macro_f1_res[f'CI {ci}% Low']} - {macro_f1_res[f'CI {ci}% High']}]")
        report_content.append(ci_line)

    report_content.append("\n--- Confusion matrix (from entire dataset) ---")
    report_content.append(f"  - True Negatives (TN) : {cm['tn']:,}")
    report_content.append(f"  - False Positives (FP): {cm['fp']:,}")
    report_content.append(f"  - False Negatives (FN): {cm['fn']:,}")
    report_content.append(f"  - True Positives (TP) : {cm['tp']:,}")
    report_content.append("="*60)

    # Print to console
    print("\n" + "\n".join(report_content))

    # Save to file
    with open(final_report_path, 'w') as f:
        f.write("\n".join(report_content))
    print(f"\nFinal report saved at: {final_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to test inpainting model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--test-dir", type=str, default="prime_patches/test", help="Directory containing test images")
    parser.add_argument("--size-filter", type=str, required=True, help="Filter for image size (e.g., '25,000,000')")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--mask-ratio", type=float, default=0.2, help="Ratio of pixels to mask (0-1)")
    parser.add_argument("--R", type=int, default=10000, help="Number of bootstrap replicates")
    parser.add_argument("--ci", type=int, default=95, help="Confidence interval level (e.g., 95 for 95%%)")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Model: {args.model_path}")
    print(f"Mask ratio: {args.mask_ratio}")

    model = create_unet()
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = SimpleImageDataset(
        root_dir=args.test_dir,
        transform=transform,
        size_filter=args.size_filter
    )
    
    if len(test_dataset) == 0:
        print(f"❌ ERROR: No test data found in '{args.test_dir}' with filter '{args.size_filter}'.")
        print("Make sure path and filter is correct.")
    else:
        print(f"✅ Found {len(test_dataset)} test images with filter '{args.size_filter}'.")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model_name_from_path = Path(args.model_path).stem
        
        test_model(
            model=model, 
            dataloader=test_loader, 
            model_name=f"{model_name_from_path}_on_{args.size_filter}_data",
            mask_ratio=args.mask_ratio,
            R=args.R,
            ci=args.ci
        )
        print("Test completed.")