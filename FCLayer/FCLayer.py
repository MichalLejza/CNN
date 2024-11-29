import numpy as np


def sigmoid(weights: np.ndarray) -> np.ndarray:
    """
    Metoda oblicza funkcję aktywacji sigmoid dla podanych wag.
    :param weights: Tablica zawierające wagi
    :return: Tablica zawierające wagi po zastosowaniu funkcji aktywacyjnej "Sigmoid"
    """
    return 1 / (1 + np.exp(-weights))


def sigmoidDerivative(weights: np.ndarray) -> np.ndarray:
    """
    Metoda oblicza pochodną funkcji aktywacji sigmoid dla podanych wag.
    :param weights: Tablica zawierająca wagi dla których policzymy pochodną funkcji aktywacyjnej "Sigmoid"
    :return: Tablica zawierające wagi po zastosowaniu funkcji pochodnej "Sigmoid"
    """
    sigmoidWeights = sigmoid(weights)
    return sigmoidWeights * (1 - sigmoidWeights)


def softMax(weights: np.ndarray) -> np.ndarray:
    """
    Metoda oblicza funkcję aktywacji softmax dla podanych wag. Stosuje się ją dla ostatniej warstwy
    :param weights: Tablica zawierające wagi dla którój policzymy funkcje Softmax. Zazwyczaj jest dwuwwymiarowa.
    :return: Tablica zawierająca wagi po zastosowaniu funkcji Softmax
    """
    exponents = np.exp(weights - np.max(weights, axis=1, keepdims=True))
    return exponents / np.sum(exponents, axis=1, keepdims=True)


def forwardPropagation(input: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple:
    """
    Metoda propagacji wprzód dla danej warstwy sieci. Polega na obliczeniu wartości iloczynu skalarnego wejścia i wag
    danej warstwy sieci neuronowej oraz na zastosowaniu funkcji aktywacji (sigmoid) dla tego iloczynu
    :param input: Tablica zawierające wejście, dla pierwszej warsty jest to spłaszczony obraz, a dla dlaszych iloczyn
    skalarny po zastosowaniu Sigmoida
    :param weights: Tablica zawierająca wagi dla danej warstwy sieci
    :param biases: Tablica zawierające Biasy dla danej warstwy sieci
    :return: Iloczyn skalarny wejścia i wag oraz ten sam iloczyn po zastosowaniu Sigmoida
    """
    dotProduct = np.dot(input, weights) + biases
    return dotProduct, sigmoid(dotProduct)


def backwardPropagation(prevDerivatives: np.ndarray, weights: np.ndarray, dotProduct: np.ndarray, size: int) -> tuple:
    """
    Metoda propagacji wstecznej dla danej warstwy sieci. Polega na obliczaniu gradientów wag i biasów na podstawie
    pochodnych z poprzedniej warstwy oraz wartości iloczynu skalarnego wag i poprzednich gradientów.
    :param prevDerivatives: Tablica zawierająca pochodne błedu z poprzedniej warstwy
    :param weights: Tablica zawierająca wagi dla danej warstwy sieci
    :param dotProduct: Tablica zawierające Biasy dla danej warstwy sieci
    :param size: Rozmiar batcha
    :return: Pochodne błędu dla wyjść bieżącej warstwy, Gradienty wag bieżącej warstwy,
    Gradienty biasów bieżącej warstwy.
    """
    dWeights = (1 / size) * np.dot(prevDerivatives, dotProduct)
    dBias = (1 / size) * np.sum(prevDerivatives, axis=1, keepdims=True)
    dOutput = np.dot(weights, prevDerivatives)
    return dOutput, dWeights, dBias
