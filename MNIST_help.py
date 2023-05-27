from matplotlib import pyplot as plt  
from tensorflow.keras.datasets import mnist
import numpy as np

# Einlesen des MNIST_Datensatzes
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normierung der Inputpixel auf [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# Plotten 10 zufälliger grey-scale digits
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.axis("off")
    plt.imshow(1 - x_train[i], cmap=plt.cm.binary)
    plt.xlabel(f"Label: {y_train[i]}")
plt.show()

x_train_vectorized = np.reshape(x_train, (x_train.shape[0], 784)).T
print(x_train_vectorized.shape)
print(x_train)