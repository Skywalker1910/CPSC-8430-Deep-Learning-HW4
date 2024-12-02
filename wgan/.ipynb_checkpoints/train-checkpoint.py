import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torchvision.utils as vutils
from wgan.model import WGANGenerator, WGANDiscriminator
from utils.dataset import get_cifar10_loader
from utils.image_utils import save_best_images

# Hyperparameters
nz = 100
epochs = 25
batch_size = 128
lr = 0.00005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen = WGANGenerator(nz).to(device)
disc = WGANDiscriminator().to(device)

optimizer_gen = optim.RMSprop(gen.parameters(), lr=lr)
optimizer_disc = optim.RMSprop(disc.parameters(), lr=lr)

dataloader = get_cifar10_loader(batch_size)

# Tracking Losses
generator_losses = []
discriminator_losses = []

for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # Train Discriminator
        disc.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        output = disc(real_data).view(-1)
        loss_real = -output.mean()
        loss_real.backward()
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_data = gen(noise)
        output = disc(fake_data.detach()).view(-1)
        loss_fake = output.mean()
        loss_fake.backward()
        optimizer_disc.step()
        
        # Weight Clipping
        for p in disc.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        # Calculate total discriminator loss
        loss_disc = loss_real + loss_fake
        discriminator_losses.append(loss_disc.item())

        # Train Generator every n_critic steps
        if i % 5 == 0:
            gen.zero_grad()
            output = disc(fake_data).view(-1)
            loss_gen = -output.mean()
            loss_gen.backward()
            optimizer_gen.step()
            generator_losses.append(loss_gen.item())
        
        # Print training progress
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")
        
        # Save generated images periodically
        if i % 100 == 0:
            vutils.save_image(fake_data, f"wgan/output/images/epoch_{epoch}_batch_{i}.png", normalize=True)

    # Save the best images from each epoch
    save_best_images(fake_data, epoch, "wgan/output/best_images/")

# Save Loss Data
import matplotlib.pyplot as plt

# Plot Discriminator and Generator Loss
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses, label="Generator")
plt.plot(discriminator_losses, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("wgan/output/loss_plot.png")
plt.show()