# Reinforcing Canine Behavior in Real-Time with Machine Learning on an Embedded Device

In this study we outline the development methodology for an automatic dog treat dispenser which combines machine learning and embedded hardware to identify and reward dog behaviors in real-time. Using machine learning techniques for training an image classification model we identify three behaviors of our canine companions: "sit", "stand", and "lie down" with up to 92% test accuracy and 39 frames per second. We evaluate a variety of neural network architectures, interpretability methods, model quantization and optimization techniques to develop a model specifically for an NVIDIA Jetson Nano. We detect the aforementioned behaviors in real-time and reinforce positive actions by making inference on the Jetson Nano and transmitting a signal to a servo motor to release rewards from a treat delivery apparatus.

## Install Dependencies
1. Flash SD Card Image for [Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetpack)
2. Ensure JetPack is installed to latest version. [Upgrade/install JetPack](https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html)
3. Install TensorFlow for Jetson Platform. [Installation 1](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html) [Installation 2](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770)

## Dataset

Our curated dataset can be found on HuggingFace:

https://huggingface.co/datasets/stockeh/dog-pose-cv
