import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
import librosa.display

# Loads the same file in mono and stereo so that I have a y of the correct dimension
# For whatever reason, this is the only way (I have found) to do this without converting to int16
# Oddly, y[0] and y[1] are not the same even though the original sound.wav file is mono... strange

y, sr = librosa.load('sound.wav', mono= False)
y2, sr2 = librosa.load('speechsine.wav')
y3, sr3 = librosa.load('sound.wav')
#y =np.zeros(y.shape) this doesn't work! The audio file refuses to write like this

print(y)
print(y.shape)
print(y2.shape)
print(y3.shape)
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