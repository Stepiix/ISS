import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as ss
import math
def dtft(toneWave, max_valuefirst):
    L=[float(i)/100 for i in range(0,100)]
    number = np.zeros(100,dtype=complex)
    for n in range(number.size):
        for m in range(toneWave.size):
            number[n] += toneWave[m] * np.exp(-1j * 2*math.pi*m*(max_valuefirst+L[n])/toneWave.size)
    return number
#24 42 97
#32.70 92.50 2217.46    
#Fs = 48 000
orig, Fs = sf.read("klavir.wav")
midiLow = 24
midiHigh = 108
toneSamples = 2 * Fs #96 000
halfsecond = int(Fs * 0.5)
tones = np.arange(midiLow,midiHigh+1)
for tone in tones:
    tone = tone
    start = toneSamples * (tone - midiLow) + 12000
    toneWave = orig[start : start + halfsecond]
    toneDFT = np.fft.fft(toneWave)
    module = np.abs(toneDFT)
    moduleHalf = module[:module.size // 2]
    max_valuefirst = np.argmax(moduleHalf) - 0.5
    number = dtft(toneWave,max_valuefirst)
    module = np.abs(number)
    max_valuesecond = np.argmax(module)
    if tone < 38:
        max_valuesecond = (max_valuesecond/100+max_valuefirst)
    elif tone < 41:
        max_valuesecond = (max_valuesecond/100+max_valuefirst) * 2
        max_valuesecond = max_valuesecond/3
    elif 53<=tone and tone<=55:
        max_valuesecond = (max_valuesecond/100+max_valuefirst)
    else:
        max_valuesecond = (max_valuesecond/100+max_valuefirst) * 2
    print(tone,'	',max_valuesecond)
