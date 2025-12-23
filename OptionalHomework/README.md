# Optional Homework – Advanced Topics in Neural Networks

## Summary
I trained a small CNN that transforms CIFAR-10 images from RGB 3x32x32
to grayscale 1x28x28 images with horizontal and vertical flipping.

## Model
A lightweight 2-layer convolutional network was used for fast inference.

## Loss
MSELoss was used as this is an image-to-image regression task.

## Early Stopping
Training stops if validation loss does not improve for 5 epochs.

## Inference
The model is benchmarked against sequential torchvision transforms.
On GPU/MPS and sufficient batch sizes, the model is significantly faster.
