# from GUI import *
from MLP.MLP import Model
from Utility import *

################################################################################################
# Michał Lejza s26690 projekt zaliczeniowy - Rozpoznwawanie Liczb
# Model 5 warstwowego Perceptronu do rozpoznawania Liczb i dużych liter.
# Baza zawiera blisko 814,255 obrazów ręcznie napisanych Liter i Liczb.
# Baza pochodzi ze strony: https://www.nist.gov/itl/products-and-services/emnist-dataset
# Nazwa Bazy: EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
################################################################################################
# Zastosowane bibiloteki:
# numpy - do obliczeń
# tkinter - do aplikacji GUI
# matplotlib - do przedstawienia historii dokładności i kosztu
# cv2 i subprocces - do pobrania narysowanego obrazu i zamienienie plike .png na tablicę 28x28
# pickle - do zapisu modelu do pliku pickle
################################################################################################

if __name__ == '__main__':
    # tworzenie obiektu Modelu
    model = Model((256, 128, 10), 3, 16, 0.001)
    model.train()
    # wypisanie informacji o modelu
