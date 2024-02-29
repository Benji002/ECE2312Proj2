import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
import librosa.display
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter, freqz
from scipy.signal import butter,filtfilt

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs * 2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

y, sr = librosa.load('sound.wav')

frequency = 5000
length = 5

t = np.arange(0, length, 1/(sr))
t2 = np.linspace(0, frequency/2, sr * length)
t = np.multiply(t,frequency) # do frequency instead of t2 for constant frequency
print(type(t))
y2 = 0.1*np.sin(2 * np.pi * t)  #  Has frequency of 440Hz

y = np.add(y, y2)

fig, ax = plt.subplots(nrows=1, sharex=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sr)
plt.show()

y = butter_lowpass_filter(y, 4000, sr, 3)

fig, ax = plt.subplots(nrows=1, sharex=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sr)
plt.show()


maxint16 = np.iinfo(np.int16).max  # == 2**15-1
print("maxint16", maxint16)

m = np.max(np.abs(y))
print("m", m)

# You have to Normalize the audio. The * 100 is just so I can listen to this without hurting my ears
y = maxint16 * y / (m * 100)

# You have to convert to int16, else doesn't work
y = y.astype(np.int16)
wavfile.write('Overlay_and_Filter.wav', sr, y)