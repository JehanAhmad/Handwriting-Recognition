# Handwriting-Recognition
Python code uses TensorFlow and OpenCV to implement a CNN for recognizing handwritten digits. It loads the MNIST dataset, preprocesses the images, defines and trains the model, and evaluates its performance, it predicts a digit from a sample handwriting image fetched from a URL.
# https://colab.research.google.com/drive/1-9hMf7hbZyni411N-7N6npSzKHnHBIRe#scrollTo=DDVzB7z3XsL6&line=7&uniqifier=1
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import cv2
import urllib.request

# Download the MNIST dataset
mnist = tf.keras.datasets.mnist
 (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the input images and normalize the pixel values between 0and 1
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

# Define the neural network model
model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),
                                                    activation='relu', input_shape=(28,28,1)),
                             tf.keras.layers.MaxPooling2D((2,2)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10)])

# Compile the model with appropriate loss and metric functions
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model on the MNIST dataset
model.fit(train_images, train_labels, epochs=10,
validation_data=(test_images, test_labels))

# Test the model on a sample handwriting image
url ='https://github.com/opencv/opencv/raw/master/samples/data/digits.png'
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)[1]
img = img.reshape((1, 28, 28, 1)) / 255.0

# Predict the digit in the handwriting image
prediction = model.predict(img)
digit = np.argmax(prediction)

# Evaluate the performance of the trained model on the test dataset
test_predictions = np.argmax(model.predict(test_images), axis=1)
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions,
                            average='macro')
recall = recall_score(test_labels, test_predictions, average='macro')
f1 = f1_score(test_labels, test_predictions, average='macro')

# Print the predicted digit and the performance metrics of the trained model
print("Predicted Digit: {}".format(digit))
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
