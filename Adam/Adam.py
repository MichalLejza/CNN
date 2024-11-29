import numpy as np


class Adam:
    """
      Klasa implementująca algorytm optymalizacji Adam (Adaptive Moment Estimation).
      Adam to metoda optymalizacji, która utrzymuje średnie wartości gradientów
      pierwszego i drugiego rzędu oraz odpowiednio je normalizuje.
      """
    def __init__(self, learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Inicjalizacja parametrów dla optymalizatora Adam.
        :param learningRate: Współczynnik uczenia.
        :param beta1: Współczynnik dla pierwszego momentu.
        :param beta2: Współczynnik dla drugiego momentu.
        :param epsilon: Mała stała zapobiegająca dzieleniu przez zero.
        """
        self.learning_rate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = {}
        self.firstMoment = {}
        self.secondMoment = {}

    def initialize(self, shape: tuple, index: int) -> None:
        """
        Metoda inicjalizuje wektory momentów pierwszego (m) i drugiego (v) rzędu dla danej wagi.
        :param shape: Kształt tablicy wagi, który ma być zainicjalizowany.
        :param index: Indeks identyfikujący daną wagę w sieci.
        :return: None
        """
        self.firstMoment[index] = np.zeros(shape)
        self.secondMoment[index] = np.zeros(shape)
        self.iterations[index] = 0

    def update(self, index: int, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Metoda aktualizuje wagi sieci neuronowej na podstawie obliczonych gradientów.
        :param index: Indeks dla danej wagi w sieci.
        :param weights: Aktualne wartości wag, które mają zostać zaktualizowane.
        :param gradients: Obliczone gradienty dla danego zestawu wag.
        :return: Zaktualizowane wagi
        """
        self.iterations[index] += 1
        self.firstMoment[index] = self.beta1 * self.firstMoment[index] + (1 - self.beta1) * gradients
        self.secondMoment[index] = self.beta2 * self.secondMoment[index] + (1 - self.beta2) * gradients * gradients

        firstMomentCorrected = self.firstMoment[index] / (1 - self.beta1 ** self.iterations[index])
        secondMomentCorrected = self.secondMoment[index] / (1 - self.beta2 ** self.iterations[index])
        updateWeights = self.learning_rate * firstMomentCorrected / (np.sqrt(secondMomentCorrected) + self.epsilon)
        weights -= updateWeights
        return weights
