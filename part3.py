import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
import librosa.display

y, sr = librosa.load('sound.wav', mono= False)
y2, sr2 = librosa.load('speechsine.wav')
y3, sr3 = librosa.load('sound.wav')

print(y)
print(y.size)
print(y2.size)
print(y3.size)
y[1] = y2.copy()
y[0] = y3.copy()
print(y)

fig, ax = plt.subplots(nrows=1, sharex=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y[0])), ref=np.max)
img = librosa.display.specshow(D, y_axis='log', x_axis='time',sr=sr)

fig, ax = plt.subplots(nrows=1, sharex=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y[1])), ref=np.max)
img = librosa.display.specshow(D, y_axis='log', x_axis='time',sr=sr)
plt.show()

wavfile.write('stereospeachsine.wav', sr, y.T)