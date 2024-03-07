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
toneSamples = 2 * Fs #96 000
plt.figure()
#kresleni prvniho
plt.subplot(311)
tone = toneone
start = toneSamples * (tone - midiLow) + 12000
halfsecond = int(Fs * 0.5)
toneWave = orig[start : start + halfsecond]
sf.write("../audio/a_orig.wav",toneWave,48000)
toneDFT = np.fft.fft(toneWave)
module = np.abs(toneDFT)
moduleHalf = module[:module.size // 2]
moduleHalf = np.log(moduleHalf**2+10**-5)
bigger = np.arange(moduleHalf.size) * (Fs/toneWave.size)
plt.plot(bigger,moduleHalf)
plt.xlabel("frekvence [Hz]")
plt.ylabel("logPSD[dB]")
#kresleni druheho
plt.subplot(312)
tone = tonetwo
start = toneSamples * (tone - midiLow) + 12000
toneWave = orig[start : start + halfsecond]
sf.write("../audio/b_orig.wav",toneWave,48000)
toneDFT = np.fft.fft(toneWave)
module = np.abs(toneDFT)
moduleHalf = module[:module.size // 2]
moduleHalf = np.log(moduleHalf**2+10**-5)
bigger = np.arange(moduleHalf.size) * (Fs/toneWave.size)
plt.plot(bigger,moduleHalf)
plt.xlabel("frekvence [Hz]")
plt.ylabel("logPSD[dB]")
#kresleni tretiho
plt.subplot(313)
tone = tonethree
start = toneSamples * (tone - midiLow) + 12000
toneWave = orig[start : start + halfsecond]
sf.write("../audio/c_orig.wav",toneWave,48000)
toneDFT = np.fft.fft(toneWave)
module = np.abs(toneDFT)
moduleHalf = module[:module.size // 2]
moduleHalf = np.log(moduleHalf**2+10**-5)
bigger = np.arange(moduleHalf.size) * (Fs/toneWave.size)
plt.plot(bigger,moduleHalf)
plt.xlabel("frekvence [Hz]")
plt.ylabel("logPSD[dB]")
plt.show()
 