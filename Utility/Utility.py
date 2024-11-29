import struct
import numpy as np


def loadImages(filePath: str) -> np.ndarray:
    """
    Metoda otwiera wskazany plik i pobiera wszystkie liczby które są z zakresu 0-256 i tworzy z nich obrazy
    o wymiarach 28x28 w skali szarości 0-1 gdzie 0 to kolor biały a 1 to kolor czarny
    :param filePath: Ścieżka do pliku zawierającego obrazy w postaci binarnej
    :return: Talica z obrazami 28X28 o wymiarach Ilość Obrazów X 28 X 28
    """
    try:
        with open(filePath, 'rb') as f:
            _, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
            images = images / 256.0
            return images
    except FileNotFoundError:
        print("File with images not found")
    except struct.error as e:
        print(e)
    except IOError:
        print("I/O Error during reading Image file")


def loadLabels(filePath: str) -> np.ndarray:
    """
    Metoda otwiera wskazany plik i pobiera wszystkie etykirty
    :param filePath: Ścieżka do pliku zawierające etykiety poszczególnych obrazów
    :return: Tablica z etykietami
    """
    try:
        with open(filePath, 'rb') as f:
            _, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    except FileNotFoundError:
        print("File with Labels not found")
    except struct.error as e:
        print(e)
    except IOError:
        print("I/O Error during reading Labels file")


def prepareTrain(images: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Metoda która pozostawia tylko obrazy reprezentujęca liczby 0-9 oraz duże litery od A do Z. Ponieważ w pliku
    są również obrazy reprezentują małe litery od a do z, musimy je wyrzucić wykorzystując maskę.
    :param images: Tablica o wymiarach Ilość Obrazów X 28 X 28 zawierające obrazy reprezentujące Liczcby oraz Litery
    :param labels: Tablica o wymiarach Ilość Obrazów X 1 zawierające etykiety obrazów
    :return: Krotka przefiltrowanych obrazów i etykiet
    """
    mask = labels <= 35
    images = images[mask]
    labels = labels[mask]
    return images, labels


def returnLabel(number: np.ndarray):
    """
    Metoda zwraca char reprezentujący daną etykietę
    :param number: Wartośc etykiety: od 0 do 9 - Liczby. Od 10 do 35 - Duże litery, od 36 małe litery
    :return: Char odpowiadający etykiecie
    """
    if number < 10:
        return str(number)
    elif number < 36:
        return chr(ord('A') + number - 10)
    else:
        return chr(ord('a') + number - 36)


def printLabel(number: int) -> None:
    """
    Metoda która wypisuje charową reprentacje etykiety
    :param number: Wartośc etykiety
    :return: None
    """
    if number < 10:
        print("Number: " + str(number))
    elif number < 36:
        print("Big letter: " + chr(ord('A') + number - 10))
    else:
        print("Letter: " + chr(ord('a') + number - 36))


def printImage(image: np.ndarray) -> None:
    """
    Metoda która wypisuje na output obraz. Z jakiegoś powodu wszystkie obrazy w pliku są po transpozycji, więc musimy
    wypisać obraz transponowny
    :param image: Tablica reprezentująca obraz
    :return: None
    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 0:
                print("\033[31m", end="")
            print(format(image[i][j], ".1f"), end='')
            print("\033[0m ", end="")
        print()
    print()


def oneHotEncoding(labels: np.ndarray, classes: int) -> np.ndarray:
    """
    Metoda która tworzy tablice "one hoe encoding" która ma same zera poza indeksami wskazanymi, tam są jedynkai
    :param labels: Tablica zawierjące etykiety
    :param classes: Liczba klas albo rozmiar pojedynczej tablicy które tworzymy
    :return: Tablica zawierająca tablice one hot encoding
    """
    oneHot = np.zeros((len(labels), classes), dtype=int)
    oneHot[np.arange(len(labels)), labels] = 1
    return oneHot


def calculateLoss(outputLayer: np.ndarray, labels: np.ndarray) -> float:
    """
    Metoda która oblicza nam koszt dla sieci neuronowej
    :param outputLayer: Tablica zawierające przewidywane wartości
    :param labels: Tablica zawierające faktyczne wartości. Wynik metody oneHotEncoding
    :return: Wartość reprentująca wartość kosztu
    """
    shiftedOutput = outputLayer - np.max(outputLayer, axis=1, keepdims=True)
    expSum = np.sum(np.exp(shiftedOutput), axis=1, keepdims=True)
    logOfProbabilites = shiftedOutput - np.log(expSum)
    loss = -np.sum(logOfProbabilites[np.arange(outputLayer.shape[0]), labels]) / outputLayer.shape[0]
    return loss
