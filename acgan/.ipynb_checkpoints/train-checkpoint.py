import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torchvision.utils as vutils
from acgan.model import ACGANGenerator, ACGANDiscriminator
from utils.dataset import get_cifar10_loader
from utils.image_utils import save_best_images

# Hyperparameters
nz = 100
epochs = 50
batch_size = 64
lr_gen = 1.5e-4
lr_disc = 1.5e-4
beta1 = 0.5
features_g = 64
features_d = 64
n_classes = 10
embed_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen = ACGANGenerator(nz, n_classes, embed_size, features_g).to(device)
disc = ACGANDiscriminator(n_classes, features_d).to(device)

optimizer_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(beta1, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(beta1, 0.999))

criterion_adv = torch.nn.BCELoss()
criterion_aux = torch.nn.CrossEntropyLoss()
dataloader = get_cifar10_loader(batch_size)

# Tracking Losses
generator_losses = []
discriminator_losses = []

for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.size(0)
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # Adversarial and ground truths
        valid = torch.full((batch_size, 1), 0.9, device=device)
        fake = torch.full((batch_size, 1), 0.0, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        gen.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        gen_labels = torch.randint(0, n_classes, (batch_size,), device=device)

        # Generate a batch of images
        gen_imgs = gen(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = disc(gen_imgs, gen_labels)
        g_loss = 0.5 * (criterion_adv(validity, valid) + criterion_aux(pred_label, gen_labels))

        g_loss.backward()
        optimizer_gen.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        disc.zero_grad()

        # Loss for real images
        real_pred, real_aux = disc(real_imgs, labels)
        d_real_loss = 0.5 * (criterion_adv(real_pred, valid) + criterion_aux(real_aux, labels))

        # Loss for fake images
        fake_pred, fake_aux = disc(gen_imgs.detach(), gen_labels)
        d_fake_loss = 0.5 * (criterion_adv(fake_pred, fake) + criterion_aux(fake_aux, gen_labels))

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        optimizer_disc.step()

        # Track losses
        generator_losses.append(g_loss.item())
        discriminator_losses.append(d_loss.item())

        # Print training progress
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

        # Save generated images periodically
        if i % 100 == 0:
            vutils.save_image(gen_imgs, f"acgan/output/images/epoch_{epoch}_batch_{i}.png", normalize=True)

    # Save the best images from each epoch
    save_best_images(gen_imgs, epoch, "acgan/output/best_images/")

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
plt.savefig("acgan/output/loss_plot.png")
plt.show()

