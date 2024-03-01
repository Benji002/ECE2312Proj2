import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
import librosa.display

y, sr = librosa.load('sound.wav', mono= False)
y2, sr2 = librosa.load('Overlay.wav')
y3, sr3 = librosa.load('sound.wav')

print(y)
print(y.size)
print(y2.size)
print(y3.size)
y[1] = y2.copy()
y[0] = y3.copy()
print(y)

wavfile.write('Right_Whine.wav', sr, y.T)