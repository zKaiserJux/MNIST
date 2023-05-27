import numpy as np
from tensorflow.keras.datasets import mnist

# Einlesen des MNIST-Datensatzes
# x_train / x_test sind die Inputdaten => hier: 28 x 28 pxl handgeschriebene grey-scale digits
# y_train / y_test sind die gewünschten Outputs der grey-scale digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normierung der Pixelwerte auf [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# Umwandlung des Arrays von Input_Matrizen in eine Matrix mit den jeweiligen Input-Vektoren als Spalten
x_train_vectorized = np.reshape(x_train, (x_train.shape[0], 784)).T
x_test_vectorized = np.reshape(x_test, (x_test.shape[0], 784)).T

# Initilisierung der benötigten Gewichtematrizen und Biases
def init_paramters():
    W_1 = np.random.rand(10, 784) - 0.5
    b_1 = np.random.rand(10, 1) - 0.5
    W_2 = np.random.rand(10, 10) - 0.5
    b_2 = np.random.rand(10, 1) - 0.5
    return W_1, b_1, W_2, b_2

# Aktivierungsfunktion: x, wenn x > 0 , sonst 0
def ReLU(Z):
    return np.maximum(Z, 0)

# Ableitung der ReLU-Funktion
def derivative_ReLU(Z):
    return Z > 0
    
# Wahrscheinlichkeitsverteilung => gibt einen Vektor wieder, wie sicher sich das CNN bei der Schätzung ist
def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

# Kann ebenfalls als Aktivierungsfunktion des Output-Layers verwendet werden und ist zudem auch als Wahrscheinlichkeitsverteilung zu interpretieren
def sigmoid_function(Z):
    return 1 / (1 + np.exp(-Z))

# forward propagation 
def forward_propagation(W_1, b_1, W_2, b_2, X):
    Z_1 = W_1.dot(X) + b_1
    A_1 = ReLU(Z_1)
    Z_2 = W_2.dot(A_1) + b_2
    A_2 = softmax(Z_2)
    return Z_1, A_1, Z_2, A_2

# Erstellt eine Null-Matrix und setzt anschließend den Index des korrekten Labels aus dem Datensatz y_train des i-ten Labels auf 1
def one_hot(Y):
    # Erstellt eine Matrix der Dimension (Y.size x Y.Ma)
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    # Setzt den Wert des korrekten Labels auf 1
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

# backwards propagation
def backwards_propagation(Z_1, A_1, A_2, W_2, X, Y):
    # Bestimmt die Dimension der Matrizen
    m = Y.size
    # Stellt den gewünschten Output-Vektor dar
    desired_output = one_hot(Y)
    # Berechnung der Kosten durch Differenzbildung mit dem geschätzten Wert des KNN und des eigentlich gewünschten Outputs
    dZ_2 = A_2 - desired_output 
    dW_2 = 1 / m * dZ_2.dot(A_1.T)
    db_2 = 1 / m * np.sum(dZ_2)
    dZ_1 = W_2.T.dot(dZ_2) * derivative_ReLU(Z_1)
    dW_1 = 1 / m * dZ_1.dot(X.T)
    db_1 = 1 / m * np.sum(dZ_1)
    return dW_1, db_1, dW_2, db_2

# Updaten der neuen Gewichtematrizen und Biases
def update_parameters(W_1, b_1, W_2, b_2, dW_1, db_1, dW_2, db_2, alpha):
    W_1 = W_1 - alpha * dW_1
    b_1 = b_1 - alpha * db_1
    W_2 = W_2 - alpha * dW_2
    b_2 = b_2 - alpha * db_2
    return W_1, b_1, W_2, b_2

# Gibt die Vorhersage des Netzes zurück
def get_predictions(A_2):
    # Erstellt ein Array der Indizes der größten Werte jeder Spalte
    return np.argmax(A_2, 0)

# Gibt die Genauigkeit der Vorhersage zurück
def get_accuracy(predictions, Y):
    # predictions == Y: Es wird Elementweise geprüft, ob diese gleich sind und es wird ein boolsches-Array erstellt
    # np.sum(...): Summiert über das boolesche-Array und gibt die Anzahl der korrekten Vorhersagen zurück
    return np.sum(predictions == Y) / Y.size

# Gradient-descent, um den Fehler zu minimieren
def gradient_descent(X, Y, epochs, alpha):
    W_1, b_1, W_2, b_2 = init_paramters()
    for i in range(epochs):
        Z_1, A_1, Z_2, A_2 = forward_propagation(W_1, b_1, W_2, b_2, X)
        dW_1, db_1, dW_2, db_2 = backwards_propagation(Z_1, A_1, A_2, W_2, X, Y)
        W_1, b_1, W_2, b_2 = update_parameters(W_1, b_1, W_2, b_2, dW_1, db_1, dW_2, db_2, alpha)
        if i % 50 == 0:
            print(f"Iteration: {i}")
            print("Accuracy of the network: %s " % (get_accuracy(get_predictions(A_2), Y)))
    return W_1, b_1, W_2, b_2

W_1, b_1, W_2, b_2 = gradient_descent(x_train_vectorized, y_train, 1000, 0.15)