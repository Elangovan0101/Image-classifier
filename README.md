# Image Classifier using Convolutional Neural Networks (CNN) 🖼️🧠

Welcome to the GitHub repository for the Image Classifier using Convolutional Neural Networks (CNN) project! 🌟 In this project, we build a powerful image classifier using deep learning techniques with Python and TensorFlow. The classifier is trained on the CIFAR-10 dataset, consisting of 60,000 32x32 color images across 10 different classes.

## What's Inside:

### 1. Data Loading and Preprocessing:
- Load the CIFAR-10 dataset and preprocess the images.
- Normalize pixel values to the range [0, 1] by dividing by 255.0.

### 2. Model Architecture:
- Define a CNN model architecture using TensorFlow's Keras API.
- Implement convolutional layers with ReLU activation and max-pooling layers for feature extraction.
- Include fully connected (dense) layers with ReLU activation for classification.
- Use softmax activation in the final dense layer for multi-class classification.

### 3. Model Compilation:
- Compile the model with the Adam optimizer, Sparse Categorical Crossentropy loss function, and accuracy metric for evaluation.

### 4. Model Training:
- Train the model on the CIFAR-10 training data for 10 epochs, with validation on the test data.

### 5. Model Evaluation:
- Evaluate the trained model on the test data and print the test loss and accuracy.

## Get Started:
1. **Clone the Repository:**
```bash
git clone https://github.com/Elangovan0101/Image-classifier.git
cd image-classifier

2. **Install Dependencies:**

pip install tensorflow

3.**Run the code**

python image_classifier.py

