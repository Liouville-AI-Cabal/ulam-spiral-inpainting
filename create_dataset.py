import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SPIRAL_DIR = Path("prime_spirals")
PATCH_DIR = Path("prime_patches")
PATCH_SIZE = 256
NUM_TEST_IMAGES = 10  # Number of patches to set aside for the test set
AUGMENT_PATCHES = True # Apply random flips/rotations to patches

Image.MAX_IMAGE_PIXELS = None


def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def create_grid_patches(img, patch_size):
    width, height = img.size
    patches = []
    for j in range(0, height // patch_size):
        for i in range(0, width // patch_size):
            x0 = i * patch_size
            y0 = j * patch_size
            # Crop the patch from the grid cell
            patch = img.crop((x0, y0, x0 + patch_size, y0 + patch_size))
            patches.append(patch)
    return patches


def apply_augmentation(patch):
    if random.random() < 0.5:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
    # Only 90, 180, 270 degree rotations are applied
    rotations = random.choice([0, 90, 180, 270])
    if rotations:
        patch = patch.rotate(rotations)
    return patch


def create_dataset_from_image(
    source_image, output_root_dir, patch_size, num_test, augment
):
    train_dir = Path(output_root_dir) / "train"
    test_dir = Path(output_root_dir) / "test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    try:
        img = Image.open(source_image).convert("L")
        if img.size[0] < patch_size or img.size[1] < patch_size:
            print(f"Warning: Image too small for patches. Skipping {source_image}")
            return

        # 1. Create a grid of unique, non-overlapping patches
        unique_patches = create_grid_patches(img, patch_size)

        # 2. Prepare all patches with augmentation and filenames
        all_patches = []
        for i, patch in enumerate(unique_patches):
            if augment:
                patch = apply_augmentation(patch)
            
            fname = f"patch_{Path(source_image).stem}_{i:04d}.png"
            all_patches.append((patch, fname))

        # 3. Shuffle and split the unique patches into train and test sets
        random.shuffle(all_patches)
        
        if len(all_patches) <= num_test:
            print(f"Warning: Not enough patches to create a test set for {source_image}. All patches will be used for training.")
            train_patches = all_patches
            test_patches = []
        else:
            train_patches, test_patches = all_patches[:-num_test], all_patches[-num_test:]

        # 4. Save the files
        for patch, fname in train_patches:
            patch.save(train_dir / fname)
        for patch, fname in test_patches:
            patch.save(test_dir / fname)

        print(
            f"Created {len(train_patches)} train + {len(test_patches)} test patches from {source_image}"
        )
    except Exception as e:
        print(f"Error processing {source_image}: {str(e)}")


def generate_patches_from_all_images():
    set_seeds(42)
    print("Starting data preparation from existing images...")

    if not os.path.exists(SPIRAL_DIR):
        print(f"Error: Source directory '{SPIRAL_DIR}' not found.")
        return

    spiral_images = list(Path(SPIRAL_DIR).glob("*.png"))
    if not spiral_images:
        print(f"No .png images found in '{SPIRAL_DIR}'.")
        return

    print(f"Found {len(spiral_images)} images to process.")

    for img_path in spiral_images:
        print(f"\nProcessing image: {img_path.name}...")
        create_dataset_from_image(
            source_image=img_path,
            output_root_dir=PATCH_DIR,
            patch_size=PATCH_SIZE,
            num_test=NUM_TEST_IMAGES,
            augment=AUGMENT_PATCHES,
        )

    print("\nDataset preparation completed! ✨")


if __name__ == "__main__":
    generate_patches_from_all_images()