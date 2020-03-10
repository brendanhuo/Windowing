import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set() # styling
from scipy.signal import get_window

m = 513
t = np.arange(m)

# Number of samplepoints 

N = 513

# sample spacing 

T = 1.0 / 1000

x = np.linspace(0.0, N*T, N) 

y = np.sin(10.1 * 2.0*np.pi*x) + 0.9* np.sin(25.8 * 2.0*np.pi*x) + 1.8*np.sin(35.1 * 2.0*np.pi*x)+0.1*np.sin(350.4 * 2.0*np.pi*x)
z1 = np.sin(80 * 2.0*np.pi*x)
z2 = np.sin(90 * 2.0*np.pi*x) + np.sin(91.5 * 2.0*np.pi*x)
z3 = np.sin(40 * 2.0*np.pi*x) + np.sin(120 * 2.0*np.pi*x)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(x,z1)
plt.xlabel("sample (time)")
plt.ylabel("amplitude")
plt.title("Sine at 80Hz (time)")

plt.subplot(122)

plt.xlabel("sample (freq)")
plt.ylabel("amplitude")
plt.title("Sine at 80Hz (Freq)")

number = 4600
w_fft = np.fft.rfft(z1, number)
freqs = np.fft.rfftfreq(number, d=1/(2*N))
plt.xlim(0, 160)
plt.ylim(-100, 1)
plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()))

plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(x,z2)
plt.xlabel("sample (time)")
plt.ylabel("amplitude")
plt.title("Sine at 90Hz and 91.5Hz(time)")

plt.subplot(122)

plt.xlabel("sample (freq)")
plt.ylabel("amplitude")
plt.title("Sine at 90Hz and 91.5Hz (Freq)")

number = 4600
w_fft = np.fft.rfft(z2, number)
freqs = np.fft.rfftfreq(number, d=1/(2*N))
plt.xlim(0, 160)
plt.ylim(-60, 1)
plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()))

plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(x,z3)
plt.xlabel("sample (time)")
plt.ylabel("amplitude")
plt.title("Sine at 40Hz and 120Hz(time)")

plt.subplot(122)

plt.xlabel("sample (freq)")
plt.ylabel("amplitude")
plt.title("Sine at 40Hz and 120Hz (Freq)")

number = 4600
w_fft = np.fft.rfft(z3, number)
freqs = np.fft.rfftfreq(number, d=1/(2*N))
plt.xlim(0, 160)
plt.ylim(-60, 1)
plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()))

plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(x,y)
plt.xlabel("sample (time)")
plt.ylabel("amplitude")
plt.title("Sine with noise (time)")

plt.subplot(122)

plt.xlabel("sample (freq)")
plt.ylabel("amplitude")
plt.title("Sine with noise (Freq)")

number = 4600
w_fft = np.fft.rfft(y, number)
freqs = np.fft.rfftfreq(number, d=1/(2*N))
plt.xlim(0, 500)
plt.ylim(-60, 1)
plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()))

plt.show()

def compute_mainlobe_width(spectrum):
    """
    computes mainlobe width from spectrum
    
    assumes the mainlobe starts at 0, that spectrum size is odd, and that 
    the spectrum is real-valued (half of the frequencies)
    
    returns the number of samples of full mainlobe (not just half)
    """
    abs_spectrum = np.abs(spectrum)
    current_value = abs_spectrum[0]
    for ind, next_value in enumerate(abs_spectrum):
        if next_value > current_value:
            break
        else:
            current_value = next_value        
    return 2 * ind - 1

def compute_sidelobe_level(spectrum):
    """
    computes sidelobe level from spectrum

    assumes the mainlobe starts at 0, that spectrum size is odd, and that 
    the spectrum is real-valued (half of the frequencies)
    
    returns the level of sidelobes in dB 
    """
    mainlobe_width = compute_mainlobe_width(spectrum)
    
    ind = (mainlobe_width - 1) / 2
    
    abs_spectrum = np.abs(spectrum)
    
    return 20 * np.log10(abs_spectrum[int(ind):].max() / abs_spectrum.max())
num = 3
for window in ['boxcar', 'triang', 'hanning', 'hamming', 'blackman', 'blackmanharris']:
    plt.figure(num)
    num += 1
    m = 513
    w = get_window(window, m)
    n = 4096
    w_fft = np.fft.rfft(w, n)
    freqs = np.fft.rfftfreq(n, d=1/m)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(t, w)
    plt.xlabel("sample / s")
    plt.ylabel("amplitude")
    plt.title("{} window".format(window))
    plt.xlim(0, t.size)
    plt.ylim(-0.025, 1.025)
    plt.subplot(122)
    plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()))
    plt.xlim(0, 25)
    plt.ylim(-120, 1)
    width = compute_mainlobe_width(w_fft)
    width_bins = width * m / n
    level = compute_sidelobe_level(w_fft)
    ylim_range = plt.ylim()
    plt.vlines((width - 1) / 2 * m / n, ylim_range[0], ylim_range[1], lw=3)
    xlim_range = plt.xlim()
    plt.hlines(level, xlim_range[0], xlim_range[1], lw=3)
    plt.title("{} window\nmainlobe width = {:.0f} Hz, sidelobe level = {:.0f} dB".format(window,
                                                                       width_bins, 
                                                                       level))
    plt.xlabel('frequency / Hz')
    plt.ylabel("amplitude / dB")

    plt.figure(1, figsize=(8, 6))
    w_fft = np.fft.rfft(z1*w, number)
    freqs = np.fft.rfftfreq(number, d=1/(2*N))
    plt.xlim(0, 160)
    plt.ylim(-100, 1)
    plt.title("Sine at 80Hz (Freq), windowed")
    plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()), label = "{} window".format(window))
    plt.legend()

    plt.figure(2, figsize=(8, 6))
    w_fft = np.fft.rfft(z2*w, number)
    freqs = np.fft.rfftfreq(number, d=1/(2*N))
    plt.xlim(0, 160)
    plt.ylim(-100, 1)
    plt.title("Sine at 90Hz and 91.5Hz (Freq), windowed")
    plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()), label = "{} window".format(window))
    plt.legend()

    plt.figure(11, figsize=(8, 6))
    w_fft = np.fft.rfft(z3*w, number)
    freqs = np.fft.rfftfreq(number, d=1/(2*N))
    plt.xlim(0, 160)
    plt.ylim(-120, 1)
    plt.title("Sine at 40Hz and 120Hz (Freq), windowed")
    plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()), label = "{} window".format(window))
    plt.legend()

    plt.figure(12, figsize=(8, 6))
    w_fft = np.fft.rfft(y*w, number)
    freqs = np.fft.rfftfreq(number, d=1/(2*N))
    plt.xlim(0, 400)
    plt.ylim(-120, 1)
    plt.title("Sines with noise, windowed")
    plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max()), label = "{} window".format(window))
    plt.legend()

plt.show()