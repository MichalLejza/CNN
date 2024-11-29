import struct
import numpy as np
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self, dataDir: str):
        self.dataDir = dataDir
        self.trainImages = None
        self.testImages = None
        self.trainLabels = None
        self.testLabels = None
        self._loadTrainImages()
        self._loadTestImages()
        self._loadTrainLabels()
        self._loadTestLabels()

    def hotEncoding(self):
        pass

    def _loadTrainImages(self) -> None:
        try:
            with open(self.dataDir + '/trainImages', 'rb') as f:
                _, num, rows, cols = struct.unpack('>IIII', f.read(16))
                self.trainImages = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
                self.trainImages = self.trainImages / 256.0
        except FileNotFoundError:
            print('File not found')

    def _loadTestImages(self) -> None:
        try:
            with open(self.dataDir + '/testImages', 'rb') as f:
                _, num, rows, cols = struct.unpack('>IIII', f.read(16))
                self.testImages = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
                self.testImages = self.testImages / 256.0
        except FileNotFoundError:
            print('File not found')

    def _loadTrainLabels(self) -> None:
        try:
            with open(self.dataDir + '/trainLabels', 'rb') as f:
                _, num = struct.unpack('>II', f.read(8))
                self.trainLabels = np.frombuffer(f.read(), dtype=np.uint8)
        except FileNotFoundError:
            print("File with Labels not found")
        except struct.error as e:
            print(e)
        except IOError:
            print("I/O Error during reading Labels file")

    def _loadTestLabels(self) -> None:
        try:
            with open(self.dataDir + '/testLabels', 'rb') as f:
                _, num = struct.unpack('>II', f.read(8))
                self.testLabels = np.frombuffer(f.read(), dtype=np.uint8)
        except FileNotFoundError:
            print("File with Labels not found")
        except struct.error as e:
            print(e)
        except IOError:
            print("I/O Error during reading Labels file")

    def printImage(self, index: int) -> None:
        print("Value: " + str(self.trainLabels[index]))
        for i in range(28):
            for j in range(28):
                if self.trainImages[index][0][i][j] > 0:
                    print("\033[31m", end="")
                print(format(self.trainImages[index][0][i][j], ".1f"), end='')
                print("\033[0m", end="")
            print()
        print()

    def showImage(self, index: int) -> None:
        image = self.trainImages[index, 0, :, :]
        plt.imshow(image, cmap='gray')
        plt.title('Image at index [' + str(index) + '] = ' + str(self.trainLabels[index]))
        plt.axis('off')
        plt.show()
