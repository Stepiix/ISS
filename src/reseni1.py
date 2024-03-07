import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
#24 42 97
#32.70 92.50 2217.46    
#Fs = 48 000
toneone = 24
freqone = 32.70
tonetwo = 42
freqtwo = 92.50
tonethree = 97
freqthree = 2217.46
orig, Fs = sf.read("klavir.wav")
midiLow = 24
toneSamples = 2 * Fs
plt.figure()
#kresleni prvniho
plt.subplot(311)
tone = toneone
start = toneSamples * (tone - midiLow) + 12000
period = (1/freqone)*Fs
threeperiods = int(period * 3)
toneWave = orig[start : start + threeperiods]
t_tone = np.arange(toneWave.size)/Fs * 1000
plt.plot(t_tone,toneWave)
plt.xlabel("Čas [ms]")
plt.ylabel("Amplituda")
#kresleni druheho
plt.subplot(312)
tone = tonetwo
start = toneSamples * (tone - midiLow) + 12000
period = (1/freqtwo)*Fs
threeperiods = int(period * 3)
toneWave = orig[start : start + threeperiods]
t_tone = np.arange(toneWave.size)/Fs * 1000
plt.plot(t_tone,toneWave)
plt.xlabel("Čas [ms]")
plt.ylabel("Amplituda")
#kresleni tretiho
plt.subplot(313)
tone = tonethree
start = toneSamples * (tone - midiLow) + 12000
period = (1/freqthree)*Fs
threeperiods = int(period * 3)
toneWave = orig[start : start + threeperiods]
t_tone = np.arange(toneWave.size)/Fs * 1000
plt.plot(t_tone,toneWave)
plt.xlabel("Čas [ms]")
plt.ylabel("Amplituda")
plt.show()
