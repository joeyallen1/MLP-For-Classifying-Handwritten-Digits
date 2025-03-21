# MLP-For-Classifying-Handwritten-Digits

A neural network for classifying handwritten digits from the MNIST dataset, implemented from scratch using NumPy for matrix operations. The purpose of this
project is to understand the inner workings of neural networks and apply it to a real-world classification problem.

This project was inspired by and contains portions of code from the *Introduction to Machine Learning* course on MIT OpenCourseWare.

**MNIST Dataset (CSV format):** [Kaggle - MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data)

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Future Work](#futurework)

## Features
- Modular design with linear layers, ReLU and Softmax activation functions, and an NLL loss module  
- Flexible architecture with customizable network design
- Stochastic Gradient Descent (SGD) implementation  
- Achieves ~85% accuracy on the test set 

## Installation
To get started, follow these steps:

1. Clone the repository:
   ```bash
   git clone git@github.com:joeyallen1/MLP-For-Classifying-Handwritten-Digits.git
2. Install Dependencies
   ```bash
   pip install numpy pandas matplotlib
4. Run the model:
   ```bash
   python3 train.py

## Usage
To modify the number of training iterations, learning rate, or network architecture, update the corresponding lines at the bottom of train.py.

- The final activation should always be Softmax.
- Ensure that consecutive linear layers have matching dimensions.
- The first linear layer should have an input dimension of 784 since the MNIST images are 28x28 pixels.

After training, a plot of accuracy over time will be displayed.

## Future Work
- Implementing optimizers like Adam for improved convergence
- Adding regularization techniques such as dropout to prevent overfitting
- Hyperparameter tuning for better performance