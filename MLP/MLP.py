import pickle
import matplotlib.pyplot as plt

from Adam.Adam import Adam
from FCLayer.FCLayer import *
from Utility.Utility import *
from ConvLayer.functions import *


class Model:
    def __init__(self, sizes: tuple, epochs: int, batchSize: int, learningRate: float):
        """
        Konstruktor Sieci Perceptronów. W tym przypadku 5 warstwowej. W niej inicjujemy wagi kazdej warstwy, biasy
        pobieramy dane do uczenia/testowania i inicjalizujemy optymalizator Adam do zmieniania wag podczas propagacji
        wstecznej.
        :param sizes: Rozmiary warstw
        :param epochs: Liczba Epochów (iteracji po całym zbiorze treningowym)
        :param batchSize: Rozmiar batcha (kawałka zbioru treningowego dla którego przeprowadzamy ucznie)
        :param learningRate: Tępo uczenia
        """
        self.kernels = np.random.normal(0, np.sqrt(2.0 / (1 * 3 * 3)), size=(4, 1, 3, 3))
        self.inputW = np.random.normal(0, np.sqrt(2 / 784), size=(676, sizes[0]))
        self.hiddenOneW = np.random.normal(0, np.sqrt(2 / 784), size=(sizes[0], sizes[1]))
        self.hiddenTwoW = np.random.normal(0, np.sqrt(2 / 784), size=(sizes[1], sizes[2]))

        self.inputB = np.zeros((1, sizes[0]))
        self.hiddenOneB = np.zeros((1, sizes[1]))
        self.hiddenTwoB = np.zeros((1, sizes[2]))

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.accuracyHistory = []
        self.lossHistory = []

        self.adam = Adam(learningRate=self.learningRate)
        self.adam.initialize(shape=self.inputW.shape, index=0)
        self.adam.initialize(shape=self.hiddenOneW.shape, index=1)
        self.adam.initialize(shape=self.hiddenTwoW.shape, index=2)
        self.adam.initialize(shape=self.inputB.shape, index=3)
        self.adam.initialize(shape=self.hiddenOneB.shape, index=4)
        self.adam.initialize(shape=self.hiddenTwoB.shape, index=5)
        self.adam.initialize(shape=self.kernels.shape, index=6)

        self.trainImages = loadImages(filePath="Data/Numbers/train-images.idx3-ubyte")
        self.trainLabels = loadLabels(filePath="Data/Numbers/train-labels.idx1-ubyte")
        self.testImages = loadImages(filePath="Data/Numbers/t10k-images.idx3-ubyte")
        self.testLabels = loadLabels(filePath="Data/Numbers/t10k-labels.idx1-ubyte")
        # self.trainImages, self.trainLabels = prepareTrain(images=self.trainImages, labels=self.trainLabels)
        # self.testImages, self.testLabels = prepareTrain(images=self.testImages, labels=self.testLabels)

    def __str__(self) -> str:
        """
        Metoda zwraca reprezentację tekstową modelu perceptronu wielowarstwowego.
        :return: Opis zawierający szczegółowy opis modelu MLP.
        """
        return ("Multi Layer Perceptron Model For PPY Project\n" + "Train Set Size: " + str(self.trainImages.shape[0]) +
                "   Test Set Size: " + str(self.testImages.shape[0]) +
                "\nInput-HiddenOne Weights: " + str(self.inputW.shape) +
                "\nHiddenOne-HiddenTwo Weights: " + str(self.hiddenOneW.shape) +
                "\nHiddenTwo-Output Weights: " + str(self.hiddenTwoW.shape) +
                "\nEpochs: " + str(self.epochs) + "\nBatch Size: " + str(self.batchSize) +
                "\nUpdating algorithm: Adam" + "\nLearning Rate: " + str(self.learningRate))

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Metoda przewiduje etykietę dla podanego obrazu.
        :param image: Tablicę zawierającą obraz 28x28
        :return: Przewidywana etykieta
        """
        appliedKernels = applyKernels(self.kernels, image, 1)
        appliedRelu = applyRelu(appliedKernels)
        apppliedPool = applyMaxPool(appliedRelu, 2)
        # spłaszczenie obrazów
        flattened = apppliedPool.reshape(apppliedPool.shape[0], -1)
        # propagacja wprzód
        dot1, sigmoidDot1 = forwardPropagation(input=flattened, weights=self.inputW, biases=self.inputB)
        dot2, sigmoidDot2 = forwardPropagation(input=sigmoidDot1, weights=self.hiddenOneW, biases=self.hiddenOneB)
        output, _ = forwardPropagation(input=sigmoidDot2, weights=self.hiddenTwoW, biases=self.hiddenTwoB)
        return np.argmax(output, axis=1)[0]

    def trainStep(self, images: np.ndarray, labels: np.ndarray) -> float:
        """
        Metoda Wykonuje pojedynczy krok treningowy dla modelu.
        :param images: Batch zawierający batchSize obrazów.
        :param labels: Etykiety dla każdego obrazu
        :return: Koszt dla tego batcha
        """
        appliedKernels = applyKernels(self.kernels, images, 1)
        appliedRelu = applyRelu(appliedKernels)
        apppliedPool = applyMaxPool(appliedRelu, 2)
        # spłaszczenie obrazów
        flattened = apppliedPool.reshape(apppliedPool.shape[0], -1)
        # propagacja wprzód
        dot1, sigmoidDot1 = forwardPropagation(input=flattened, weights=self.inputW, biases=self.inputB)
        dot2, sigmoidDot2 = forwardPropagation(input=sigmoidDot1, weights=self.hiddenOneW, biases=self.hiddenOneB)
        output, _ = forwardPropagation(input=sigmoidDot2, weights=self.hiddenTwoW, biases=self.hiddenTwoB)
        dOutput = softMax(output)
        # obliczenie kosztu
        loss = calculateLoss(dOutput, labels)
        # zastosowanie one hot encoding dla etykiet
        dZ5 = dOutput - oneHotEncoding(labels, 10)
        # propagacja wstecz
        dA2, dHiddenTwoW, dHiddenTwoB = backwardPropagation(dZ5.T, self.hiddenTwoW, sigmoidDot2, flattened.shape[0])
        dZ2 = dA2 * sigmoidDerivative(dot2.T)
        dA1, dHiddenOneW, dHiddenOneB = backwardPropagation(dZ2, self.hiddenOneW, sigmoidDot1, flattened.shape[0])
        dZ1 = dA1 * sigmoidDerivative(dot1.T)
        dA0, dInputW, dInputB = backwardPropagation(dZ1, self.inputW, flattened, flattened.shape[0])

        gradients = dA0.reshape(apppliedPool.shape)
        gradientsPool = backwardMaxPool(gradients, appliedRelu, 2)
        backwardRelu(gradientsPool, appliedKernels)
        gradientsKernels = backwardKernelGradients(gradientsPool, images, self.kernels)

        # zupdatowanie wag i biasów
        self.inputW = self.adam.update(index=0, weights=self.inputW, gradients=dInputW.T)
        self.hiddenOneW = self.adam.update(index=1, weights=self.hiddenOneW, gradients=dHiddenOneW.T)
        self.hiddenTwoW = self.adam.update(index=2, weights=self.hiddenTwoW, gradients=dHiddenTwoW.T)
        self.inputB = self.adam.update(index=3, weights=self.inputB, gradients=dInputB.T)
        self.hiddenOneB = self.adam.update(index=4, weights=self.hiddenOneB, gradients=dHiddenOneB.T)
        self.hiddenTwoB = self.adam.update(index=5, weights=self.hiddenTwoB, gradients=dHiddenTwoB.T)
        self.kernels = self.adam.update(index=6, weights=self.kernels, gradients=gradientsKernels)

        return loss

    def train(self) -> None:
        """
        Proces treningowy dla sieci.
        :return: None
        """
        for epoch in range(self.epochs):
            print("EPOCH " + str(epoch))
            count = 0
            # tablica do zbierania najmniejszego kosztu w danym obszarze batchów
            losses = []
            for b in range(0, 25000, self.batchSize):
                loss = self.trainStep(self.trainImages[b:b+self.batchSize], self.trainLabels[b:b+self.batchSize])
                losses.append(loss)
                if count % 50 == 0:
                    print(count)
                if count % 200 == 0:
                    self.testAccuracy()
                count += 1
        # zapisujemy model do pliku pickle
        self.saveModel()

    def testAccuracy(self) -> None:
        """
        Metoda sprawdza dokładnośc modelu na podstawie całego zbioru testowego.
        :return: None
        """
        count = 0
        for i in range(3000):
            predict = self.predict(self.testImages[i:i+1])
            if predict == self.testLabels[i]:
                count += 1
        print("Dokładność : " + str(count * 100 / len(self.testLabels)) + "%")
        # linijka poniżej tylko do procesu uczenia aby zbierać na bieżąco dokładność
        # self.accuracyHistory.append((count * 100) / len(self.testLabels))

    def showAccuracyHistory(self) -> None:
        """
        Metoda wyświetla graf za pomocą biblioteki matplotlib przedstawiającą historię dokładności modelu perceptronu
        :return: None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0, len(self.accuracyHistory)), self.accuracyHistory, marker='o',
                 linestyle='-', color='b', label='Perceptron Accuracy')
        plt.xlabel('After 925 batches')
        plt.ylabel('Accuracy')
        plt.title('Perceptron Model Accuracy over Batches')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def showLossHistory(self) -> None:
        """
        Metoda wyświetla graf za pomocą biblioteki matplotlib przedstawiającą historię kosztu modelu perceptronu
        :return: None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0, len(self.lossHistory)), self.lossHistory, marker='o',
                 linestyle='-', color='r', label='Loss')
        plt.xlabel('After 666 batches')
        plt.ylabel('Minimal Loss')
        plt.title('Perceptron Model Loss over Batches')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def loadModel(self) -> None:
        """
        Metoda pobiera wartości wag, biasów wytrenowanej sieci a także historię dokładności i kosztu z pliku.
        :return: None
        """
        weights = pickle.load(open('Data/arrays.pkl', 'rb'))
        self.inputW = weights[0]
        self.hiddenOneW = weights[1]
        self.hiddenTwoW = weights[2]
        self.inputB = weights[3]
        self.hiddenOneB = weights[4]
        self.hiddenTwoB = weights[5]
        self.accuracyHistory = weights[6]
        self.lossHistory = weights[7]
        print("Loaded Model From File: Data/arrays.pkl")

    def saveModel(self) -> None:
        """
        Metoda zapisuje do pliku pickle wszystkie warstwy wagi i biasów wytrenowanej sieci a takżę historię dokładności
        i kosztu
        :return: None
        """
        weights = [self.inputW, self.hiddenOneW, self.hiddenTwoW, self.inputB, self.hiddenOneB, self.hiddenTwoB,
                   self.accuracyHistory, self.lossHistory]
        with open('Data/arrays.pkl', 'wb') as f:
            pickle.dump(weights, f)
        print("Model Saved From file: Data/arrays.pkl")
