import os
import torchvision.utils as vutils

def save_best_images(images, epoch, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vutils.save_image(images, os.path.join(output_dir, f"best_epoch_{epoch}.png"), normalize=True)
