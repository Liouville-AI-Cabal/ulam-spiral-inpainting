import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SPIRAL_DIR = Path("/home/ulam-spiral-inpainting/prime_spirals")
PATCH_DIR = Path("/home/ulam-spiral-inpainting/prime_patches")
PATCH_SIZE = 256
NUM_TEST_IMAGES = 50  # Number of patches to set aside for the test set
NUM_TRAIN_IMAGES = 300  # Number of patches for the training set
AUGMENT_PATCHES = False # Apply random flips/rotations to patches

Image.MAX_IMAGE_PIXELS = None


def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def create_spiral_patches(img, patch_size):
    width, height = img.size
    cols = width // patch_size
    rows = height // patch_size

    patches = []
    left, right = 0, cols - 1
    top, bottom = 0, rows - 1

    while left <= right and top <= bottom:
        # top row: left -> right
        for c in range(left, right + 1):
            x0 = c * patch_size
            y0 = top * patch_size
            patches.append(img.crop((x0, y0, x0 + patch_size, y0 + patch_size)))

        # right col: top+1 -> bottom
        for r in range(top + 1, bottom + 1):
            x0 = right * patch_size
            y0 = r * patch_size
            patches.append(img.crop((x0, y0, x0 + patch_size, y0 + patch_size)))

        if top < bottom:
            # bottom row: right-1 -> left
            for c in range(right - 1, left - 1, -1):
                x0 = c * patch_size
                y0 = bottom * patch_size
                patches.append(img.crop((x0, y0, x0 + patch_size, y0 + patch_size)))

        if left < right:
            # left col: bottom-1 -> top+1
            for r in range(bottom - 1, top, -1):
                x0 = left * patch_size
                y0 = r * patch_size
                patches.append(img.crop((x0, y0, x0 + patch_size, y0 + patch_size)))

        left += 1
        right -= 1
        top += 1
        bottom -= 1

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
    source_image, output_root_dir, patch_size, num_train, num_test, augment
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
        unique_patches = create_spiral_patches(img, patch_size)

        # 2. Prepare all patches with augmentation and filenames
        all_patches = []
        for i, patch in enumerate(unique_patches):
            if augment:
                patch = apply_augmentation(patch)
            
            fname = f"patch_{Path(source_image).stem}_{i:04d}.png"
            all_patches.append((patch, fname))

        # 3. Shuffle and split the unique patches into train and test sets
        random.shuffle(all_patches)
        
        actual_test_patches_count = min(num_test, len(all_patches))
        test_patches = all_patches[:actual_test_patches_count]
        
        remaining_patches = all_patches[actual_test_patches_count:]
        actual_train_patches_count = min(num_train, len(remaining_patches))
        train_patches = remaining_patches[:actual_train_patches_count]

        # Add warnings if actual counts are less than desired
        if actual_test_patches_count < num_test:
            print(f"Warning: Only {actual_test_patches_count} test patches created for {source_image}, {num_test} were requested.")
        if actual_train_patches_count < num_train:
            print(f"Warning: Only {actual_train_patches_count} train patches created for {source_image}, {num_train} were requested.")

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
            num_train=NUM_TRAIN_IMAGES, # New parameter
            num_test=NUM_TEST_IMAGES,
            augment=AUGMENT_PATCHES,
        )

    print("\nDataset preparation completed! ✨")


if __name__ == "__main__":
    generate_patches_from_all_images()