import os
import subprocess

# Train DCGAN
subprocess.call(['python', 'dcgan/train.py'])

# Train WGAN
subprocess.call(['python', 'wgan/train.py'])

# Train ACGAN
subprocess.call(['python', 'acgan/train.py'])

# Compare GANs
subprocess.call(['python', 'evaluation/compare_gans.py'])