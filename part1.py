import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
import librosa.display

sampleRate = 22050
frequency = 8000
length = 5

t = np.arange(0, length, 1/(sampleRate))
t2 = np.linspace(0, frequency/2, sampleRate * length)
t = np.multiply(t,t2) # do frequency instead of t2 for constant frequency
print(type(t))
y = 0.1*np.sin(2 * np.pi * t)  #  Has frequency of 440Hz
    
m = np.max(np.abs(y))
print("m", m)

maxint16 = np.iinfo(np.int16).max  # == 2**15-1
print("maxint16", maxint16)

# You have to Normalize the audio. The * 100 is just so I can listen to this without hurting my ears
y = maxint16 * y / (m * 100)

# You have to convert to int16, else doesn't work
y = y.astype(np.int16) 

wavfile.write('chirp.wav', sampleRate, y)

print(y)

plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

plt.plot(t,y, 'o')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

y, sr = librosa.load('chirp.wav')
fig, ax = plt.subplots(nrows=1, sharex=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sr)
plt.show()