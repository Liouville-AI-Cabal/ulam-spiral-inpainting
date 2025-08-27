import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SPIRAL_DIR = Path("prime_spirals")
PATCH_DIR = Path("prime_patches")
PATCH_SIZE = 256
NUM_PATCHES_PER_IMAGE = 200
NUM_TEST_IMAGES = 10
AUGMENT_PATCHES = True

Image.MAX_IMAGE_PIXELS = None


def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def create_patch_dataset(
    source_image, output_root_dir, patch_size, num_patches, num_test, augment
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

        all_patches = [
            (
                random_crop_with_aug(img, patch_size, augment),
                f"patch_{Path(source_image).stem}_{i:04d}.png",
            )
            for i in range(num_patches)
        ]

        random.shuffle(all_patches)
        train_patches, test_patches = all_patches[:-num_test], all_patches[-num_test:]

        for patch, fname in train_patches:
            patch.save(train_dir / fname)
        for patch, fname in test_patches:
            patch.save(test_dir / fname)

        print(
            f"Created {len(train_patches)} train + {len(test_patches)} test patches from {source_image}"
        )
    except Exception as e:
        print(f"Error processing {source_image}: {str(e)}")


def random_crop_with_aug(img, crop_size, augment):
    width, height = img.size
    x0 = random.randint(0, width - crop_size)
    y0 = random.randint(0, height - crop_size)
    patch = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))

    if augment:
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
        rotations = random.choice([0, 1, 2, 3])
        if rotations:
            patch = patch.rotate(90 * rotations, expand=True)
            patch = patch.crop((0, 0, crop_size, crop_size))
    return patch


def generate_patches_from_existing_images():
    set_seeds(42)
    print("Starting data preparation for patch dataset from existing images...")

    if not os.path.exists(SPIRAL_DIR):
        print(f"❌ Error: Source directory '{SPIRAL_DIR}' not found.")
        print("Please make sure to place your spiral images inside this folder.")
        return

    spiral_images = list(Path(SPIRAL_DIR).glob("*.png"))

    if not spiral_images:
        print(f"❌ No .png images found in '{SPIRAL_DIR}'.")
        return

    print(f"✅ Found {len(spiral_images)} images to process.")

    for img_path in spiral_images:
        print(f"\nProcessing image: {img_path.name}...")
        create_patch_dataset(
            source_image=img_path,
            output_root_dir=PATCH_DIR,
            patch_size=PATCH_SIZE,
            num_patches=NUM_PATCHES_PER_IMAGE,
            num_test=NUM_TEST_IMAGES,
            augment=AUGMENT_PATCHES,
        )

    print("\nDataset preparations completed!")


if __name__ == "__main__":
    generate_patches_from_existing_images()
