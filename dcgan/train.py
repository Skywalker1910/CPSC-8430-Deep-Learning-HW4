import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torchvision.utils as vutils
from dcgan.model import DCGANGenerator, DCGANDiscriminator
from utils.dataset import get_cifar10_loader
from utils.image_utils import save_best_images

# Hyperparameters
nz = 100
epochs = 50  # Increased to 50 for better training
batch_size = 64  # Reduced to match notebooks for stability
lr_gen = 2e-4  # Learning rate for generator
lr_disc = 1e-4  # Reduced learning rate for discriminator
beta1 = 0.5
features_g = 64  # Feature size for Generator
features_d = 64  # Feature size for Discriminator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen = DCGANGenerator(nz, features_g).to(device)
disc = DCGANDiscriminator(features_d).to(device)

optimizer_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(beta1, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(beta1, 0.999))

criterion = torch.nn.BCELoss()
dataloader = get_cifar10_loader(batch_size)

# Tracking Losses
generator_losses = []
discriminator_losses = []

for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # Train Discriminator every alternate step
        if i % 2 == 0:
            disc.zero_grad()
            real_data = data[0].to(device)
            batch_size = real_data.size(0)
            labels_real = torch.full((batch_size,), 0.9, dtype=torch.float, device=device)  # Label smoothing

            # Forward pass for real data
            output = disc(real_data)  # Output shape is now [batch_size]
            loss_real = criterion(output, labels_real)
            loss_real.backward()

            # Generate fake data
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_data = gen(noise)
            labels_fake = torch.full((batch_size,), 0, dtype=torch.float, device=device)

            # Forward pass for fake data
            output = disc(fake_data.detach())  # Output shape is now [batch_size]
            loss_fake = criterion(output, labels_fake)
            loss_fake.backward()
            optimizer_disc.step()

            # Calculate total discriminator loss
            loss_disc = loss_real + loss_fake
            discriminator_losses.append(loss_disc.item())

        # Train Generator
        gen.zero_grad()
        labels_gen = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_data = gen(noise)
        output = disc(fake_data)  # Output shape is now [batch_size]
        loss_gen = criterion(output, labels_gen)
        loss_gen.backward()
        optimizer_gen.step()

        generator_losses.append(loss_gen.item())

        # Print training progress
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")

        # Save generated images periodically
        if i % 100 == 0:
            vutils.save_image(fake_data, f"dcgan/output/images/epoch_{epoch}_batch_{i}.png", normalize=True)

    # Save the best images from each epoch
    save_best_images(fake_data, epoch, "dcgan/output/best_images/")

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
plt.savefig("dcgan/output/loss_plot.png")
plt.show()