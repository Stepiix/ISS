import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as ss
orig, Fs = sf.read("klavir.wav")
midiLow = 24
midiHigh = 108
toneSamples = 2 * Fs #96 000
halfsecond = int(Fs * 0.5)
tones = np.arange(midiLow,midiHigh+1)
for tone in tones[:32]:
    tone = tone
    start = toneSamples * (tone - midiLow) + 12000
    toneWave = orig[start : start + halfsecond]
    toneWave = np.correlate(toneWave,toneWave,'full')
    max_value = np.argmax(toneWave)
    toneWave = toneWave[max_value:]
    min_value = np.argmin(toneWave)
    toneWave = toneWave[min_value:]
    max_value = np.argmax(toneWave)
    vysledek = Fs/(max_value+min_value+2)
    print(tone,'	',vysledek)
for tone in tones[32:]:
    tone = tone
    start = toneSamples * (tone - midiLow) + 12000
    toneWave = orig[start : start + halfsecond]
    toneDFT = np.fft.fft(toneWave)
    module = np.abs(toneDFT)
    moduleHalf = module[:module.size // 2]
    max_value = np.argmax(moduleHalf) * (Fs / toneWave.size)
    print(tone,'	',max_value)
 