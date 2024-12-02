import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
from utils.metrics import calculate_fid, calculate_inception_score
import torch

# Directories containing generated images for each GAN
dcgan_dir = "dcgan/output/images/"
wgan_dir = "wgan/output/images/"
acgan_dir = "acgan/output/images/"

# Calculate and save metrics for each GAN
def compare_gans():
    for gan_dir in [dcgan_dir, wgan_dir, acgan_dir]:
        images = load_images_from_directory(gan_dir)  # Placeholder
        fid_score = calculate_fid(images, images, model=None)  # Placeholder
        is_score = calculate_inception_score(images)
        with open("evaluation/output/comparison_metrics.txt", "a") as f:
            f.write(f"GAN: {gan_dir}, FID: {fid_score}, Inception Score: {is_score}\n")

# Placeholder for loading images
def load_images_from_directory(directory):
    return torch.randn(10, 3, 64, 64)

compare_gans()