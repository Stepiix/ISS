import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as ss
import math
import copy
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
G=[0,0,0,0,0]
F=[0,0,0,0,0]
midiLow = 24
midiHigh = 108
toneSamples = 2 * Fs #96 000
halfsecond = int(Fs * 0.5)
tones = np.arange(midiLow,midiHigh+1)
tones = np.arange(midiLow,midiHigh+1)
for tone in tones:
    tone = tone
    start = toneSamples * (tone - midiLow) + 12000
    toneWave = orig[start : start + halfsecond ]
    nules = np.zeros(toneWave.size)
    toneWave = np.concatenate((toneWave,nules))
    toneDFT = np.fft.fft(toneWave)
    module = np.abs(toneDFT)
    moduleHalf = module[:module.size // 2]
    if tone < 38:
        max_valuefirst = np.argmax(moduleHalf)/2 - 0.5
    elif tone < 41:
        max_valuefirst = np.argmax(moduleHalf)/3 -0.5
    elif 53<=tone and tone<=55:
        max_valuefirst = np.argmax(moduleHalf)/2 - 0.5
    else:
        max_valuefirst = np.argmax(moduleHalf) - 0.5
    number = dtft(toneWave,max_valuefirst)
    module = np.abs(number)
    max_valuesecond = np.argmax(module)
    max_valuesecond = (max_valuesecond/100+max_valuefirst)
    ymax_value = np.max(module)
    F[0] = max_valuesecond
    G[0] = ymax_value

    moduleHalf1 = moduleHalf.copy()
    od = max_valuesecond + F[0]/2
    do = max_valuesecond + F[0]/2 + F[0]
    for i in range(moduleHalf1.size):
        if i < od:
            moduleHalf1[i] = 0  
        if i > do:
            moduleHalf1[i] = 0
    max_valuefirst = np.argmax(moduleHalf1) - 0.5
    number = dtft(toneWave,max_valuefirst)
    module = np.abs(number)
    max_valuesecond = np.argmax(module)
    max_valuesecond = (max_valuesecond/100+max_valuefirst)
    ymax_value = np.max(module)
    F[1] = max_valuesecond
    G[1] = ymax_value

    moduleHalf2 = moduleHalf.copy()
    od = max_valuesecond + F[0]/2
    do = max_valuesecond + F[0]/2 + F[0]
    for i in range(moduleHalf2.size):
        if i < od:
            moduleHalf2[i] = 0  
        if i > do:
            moduleHalf2[i] = 0
    max_valuefirst = np.argmax(moduleHalf2) - 0.5
    number = dtft(toneWave,max_valuefirst)
    module = np.abs(number)
    max_valuesecond = np.argmax(module)
    max_valuesecond = (max_valuesecond/100+max_valuefirst)
    ymax_value = np.max(module)
    F[2] = max_valuesecond
    G[2] = ymax_value

    moduleHalf3 = moduleHalf.copy()
    od = max_valuesecond + F[0]/2
    do = max_valuesecond + F[0]/2 + F[0]
    for i in range(moduleHalf3.size):
        if i < od:
            moduleHalf3[i] = 0  
        if i > do:
            moduleHalf3[i] = 0
    max_valuefirst = np.argmax(moduleHalf3) - 0.5
    number = dtft(toneWave,max_valuefirst)
    module = np.abs(number)
    max_valuesecond = np.argmax(module)
    max_valuesecond = (max_valuesecond/100+max_valuefirst)
    ymax_value = np.max(module)
    F[3] = max_valuesecond
    G[3] = ymax_value

    moduleHalf4 = moduleHalf.copy()
    od = max_valuesecond + F[0]/2
    do = max_valuesecond + F[0]/2 + F[0]
    for i in range(moduleHalf4.size):
        if i < od:
            moduleHalf4[i] = 0  
        if i > do:
            moduleHalf4[i] = 0
    max_valuefirst = np.argmax(moduleHalf4) - 0.5
    number = dtft(toneWave,max_valuefirst)
    module = np.abs(number)
    max_valuesecond = np.argmax(module)
    max_valuesecond = (max_valuesecond/100+max_valuefirst)
    ymax_value = np.max(module)
    F[4] = max_valuesecond
    G[4] = ymax_value
    G[0] = np.log(G[0]**2+10**-5)
    G[1] = np.log(G[1]**2+10**-5)
    G[2] = np.log(G[2]**2+10**-5)
    G[3] = np.log(G[3]**2+10**-5)
    G[4] = np.log(G[4]**2+10**-5)
    print(tone,F,G)

