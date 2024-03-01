import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
import librosa.display

sampleRate = 22050
frequency = [392, 440, 349, 174, 261]
length = 5

# Makes a time vector and then 5 seperate frequency vectors of with size (length * sr / 5)
# Adds these 5 frequency vectors together and then multiplies
t = np.arange(0, length, 1/(sampleRate))
t2 = np.ones(sampleRate) * frequency[0]
t2 = np.append(t2, np.ones(sampleRate) * frequency[1])
t2 = np.append(t2, np.ones(sampleRate) * frequency[2])
t2 = np.append(t2, np.ones(sampleRate) * frequency[3])
t2 = np.append(t2, np.ones(sampleRate) * frequency[4])

t = np.multiply(t,t2) # do frequency instead of t2 for constant frequency
print(type(t))
y = 0.1*np.sin(2 * np.pi * t)  #  0.1 is the magnitude in this case
    
m = np.max(np.abs(y))
print("m", m)

maxint16 = np.iinfo(np.int16).max  # == 2**15-1
print("maxint16", maxint16)

# You have to Normalize the audio. The * 100 is just so I can listen to this without hurting my ears
# I later found out this normalization wasn't necessary and it's fine to just do y = maxint16 * y
# I left this in because its what I used for fig 1, 2, and 3 and it also doesn't really affect the figure
y = maxint16 * y / (m * 100)

# You have to convert to int16, else doesn't work
y = y.astype(np.int16) 

wavfile.write('cetk.wav', sampleRate, y)

print(y)

plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

plt.plot(t,y, 'o')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

y, sr = librosa.load('cetk.wav')
fig, ax = plt.subplots(nrows=1, sharex=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
img = librosa.display.specshow(D, y_axis='fft_note', x_axis='time',sr=sr, )
plt.show()