"""
Practica 6. Clasificadores de distancia minima
Hernández Jiménez Erick Yael
Patiño Flores Samuel
Robert Garayzar Arturo

Descripción: Es un programa donde se desarrollaron los algoritmos de clasificacion de distancia
minima y el algoritmo 1NN. Se aplican sobre 3 datasets (Iris Wine y Breast Cancer), mostrando 
como resultado el accuracy y su matriz de confusion, despues de aplicarlo con los metodos de 
validacion Hold-Out, 10 Fold Croos Validation y Leave One Out
"""

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_matrix, title="Matriz de Confusión"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


class MinimumDistanceClassifier:
    def __init__(self):
        self.centroids = None
        self.classes = None

    def fit(self, X, y):
        # Define el total de clases en el dataset
        self.classes = np.unique(y)
        # Calcula los centroides de cada clase c, con el uso de mean(axis=0) se calcula el promedio,
        # lo que genera el centroide
        self.centroids = np.array([X[y == c].mean(axis=0) for c in self.classes])

    def predict(self, X):
        # Calcula la distancia del punto a cada centroide
        distances = cdist(X, self.centroids)
        # Regresa la clase a la que pertenece el punto tomando la menor distancia
        return self.classes[np.argmin(distances, axis=1)]

class OneNearestNeighborClassifier:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = cdist(X, self.X_train)
        nearest_neighbors = np.argmin(distances, axis=1)
        return self.y_train[nearest_neighbors]


def evaluate_model(X, y, tipo, method="holdout"):
    classifier = MinimumDistanceClassifier() if tipo == 0  else OneNearestNeighborClassifier()

    if method == "holdout":
        # Hold-Out 70/30
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Hold-Out (70/30):")
        print(f"Accuracy: {accuracy:.4f}")

        # Graficar la matriz de confusión
        plot_confusion_matrix(conf_matrix, title="Hold-Out (70/30) Confusion Matrix")

    elif method == "cross_val":
        # 10-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=10)
        accuracies = []
        confusion_matrices = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            confusion_matrices.append(confusion_matrix(y_test, y_pred))

        # Calcular la precisión media y la matriz de confusión acumulada
        accuracy = np.mean(accuracies)
        conf_matrix = np.sum(confusion_matrices, axis=0)

        print(f"10-Fold Cross-Validation:")
        print(f"Accuracy: {accuracy:.4f}")

        # Graficar la matriz de confusión
        plot_confusion_matrix(conf_matrix, title="10-Fold Cross-Validation Confusion Matrix")

    elif method == "leave_one_out":
        # Leave-One-Out
        loo = LeaveOneOut()
        accuracies = []
        confusion_matrices = []

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            confusion_matrices.append(confusion_matrix(y_test, y_pred, labels=np.unique(y)))

        # Calcular la precisión media y la matriz de confusión acumulada
        accuracy = np.mean(accuracies)
        conf_matrix = np.sum(confusion_matrices, axis=0)

        print(f"Leave-One-Out:")
        print(f"Accuracy: {accuracy:.4f}")

        # Graficar la matriz de confusión
        plot_confusion_matrix(conf_matrix, title="Leave-One-Out Confusion Matrix")


datasets = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "Breast Cancer": load_breast_cancer()
}

for name, data in datasets.items():
    print(f"Dataset: {name}")
    X, y = data.data, data.target
    evaluate_model(X, y, 0,method="holdout")
    evaluate_model(X, y, 0,method="cross_val")
    evaluate_model(X, y, 0,method="leave_one_out")
    print("-" * 30)

"""# 1NN"""

for name, data in datasets.items():
    print(f"Dataset: {name}")
    X, y = data.data, data.target
    evaluate_model(X, y, 1,method="holdout")
    evaluate_model(X, y, 1,method="cross_val")
    evaluate_model(X, y, 1,method="leave_one_out")
    print("-" * 30)
