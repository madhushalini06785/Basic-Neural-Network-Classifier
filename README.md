MNIST Handwritten Digit Classification with Keras
This project demonstrates a simple deep learning model built using Tensorflow and Keras to classify digits from the dataset.The model is trained on 28x28 images of digits(0-9) and uses a basic neural network architecture to recognize the digits with good accuracy.

## Dataset
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

## Requirements
Python 3.7+
TensorFlow
Numpy
Matplotlib

## Notes
The dataset is normalized by dividing pixel values by 255.
Labels are one-hot encoded using to_categorical().
The model uses a very small hidden layer (5 neurons) for demonstration purposes. You can increase it to improve performance.

## output
sample output
Epoch 1/5
1875/1875 - 5s - 3ms/step - accuracy: 0.7444 - loss: 0.8175
Epoch 2/5
1875/1875 - 3s - 2ms/step - accuracy: 0.8489 - loss: 0.5202
Epoch 3/5
1875/1875 - 3s - 2ms/step - accuracy: 0.8643 - loss: 0.4709
Epoch 4/5
1875/1875 - 4s - 2ms/step - accuracy: 0.8742 - loss: 0.4406
Epoch 5/5
1875/1875 - 5s - 3ms/step - accuracy: 0.8783 - loss: 0.4228

## License
This project is licensed under the MIT License.
