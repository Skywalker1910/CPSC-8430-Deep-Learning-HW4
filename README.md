# CPSC-8430 Deep Learning - Homework 4

## Overview
This repository contains the implementation of several **Generative Adversarial Networks (GANs)** as part of the Homework 4 assignment for **CPSC-8430 Deep Learning**. The models implemented in this assignment are:
- **DCGAN (Deep Convolutional GAN)**
- **WGAN (Wasserstein GAN)**
- **ACGAN (Auxiliary Classifier GAN)**

The goal of this homework is to implement, train, and evaluate these models, comparing their performances using key metrics such as **Frechet Inception Distance (FID)** and **Inception Score (IS)**.


## Prerequisites
- **Python 3.8** or later.
- Install the required dependencies with the command:

```bash
pip install -r requirements.txt
```

## Metrics for Evaluation
The two main metrics used for evaluating the performance of GANs are:

- Frechet Inception Distance (FID): Measures how similar the generated images are to real images.
- Inception Score (IS): Evaluates the quality and diversity of generated images by computing the Kullback-Leibler divergence of the class probabilities.

## Running scripts
Example: Training DCGAN on CIFAR-10
Make sure CIFAR-10 is downloaded in /data/ or use the default PyTorch loader in the training script.
Run the DCGAN script to train the model: 

```python
python train.py
```

## Notes and Recommendations
- GPU Acceleration: For faster training, use a GPU. Ensure torch.cuda.is_available() returns True.
- Parameter Tuning: Adjust batch size and learning rate in the respective training script for better performance.
- Training Time: Training GANs can be time-consuming. Expect the training to take hours to complete, depending on the dataset and computational power.
