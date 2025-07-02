MNIST Handwritten Digit Classification with Keras
This project demonstrates a simple deep learning model built using Tensorflow and Keras to classify digits from the dataset.The model is trained on 28x28 images of digits(0-9) and uses a basic neural network architecture to recognize the digits with good accuracy.
##Dataset
The project uses the "MNIST dataset",which is actually loaded using:
from tensorflow.keras.datasets import mnist

Training samples:60,000
Testing samples:10,000
image size:28x28 pixels
Classes:Digits from 0-9

## Model Architecture
The neural network is built using Keras Sequential API with the following layers:
Input Layer: Accepts 28x28 grayscale images.
Flatten Layer: Converts 2D image into 1D vector.
Dense Layer: 5 neurons with ReLU activation.
Output Layer: 10 neurons with Softmax activation (for classification of digits 0-9).

## Model Compilation
Loss function:categorical_crossentropy
Optimizer:Adam
Metric:Accuracy

## Training and Evaluation
The model is trained for 5 epochs with a batch size of 32:

Evaluation on the test set:

## Saving the model
The trained model is saved in the Keras format:

## Sample Visualization
Displays a sample input image and its corresponding label:

## License
This project is licensed under the MIT License.

## Requirements
Python 3.7+
TensorFlow
Numpy
Matplotlib
