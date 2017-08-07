import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __name__ == "__main__":
    # load data
    wave, flux = np.loadtxt('sunspec_500nm.txt', unpack=True, dtype=np.float64)
    flux /= 1e4
    wi, wf, weight = np.loadtxt('G2.mas',unpack=True,dtype=np.float64)
    
    # setup figure
    c2 = '#003399' # blue
    plt.rcParams["font.sans-serif"] = "Helvetica"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(wave, flux, color='black') # plot the Sun
    
    # plot mask lines
    mask = (wi > 4995.) & (wf < 5005.)
    for wwi,wwf,wweight in zip(wi[mask], wf[mask], weight[mask]):
        ws = np.arange(wwi,wwf,0.001)
        fs1 = np.zeros_like(ws)
        fs2 = np.zeros_like(ws) + np.sqrt(wweight)
        #fs1 = np.ones_like(ws) - np.sqrt(wweight)
        #fs2 = np.ones_like(ws)
        ax.fill_between(ws, fs1, fs2, facecolor=c2, alpha=0.5)
    
    # tweak & save    
    ax.set_xlabel(r'Wavelength $(\AA)$')
    ax.set_ylabel('Normalized Flux')
    ax.set_xlim([4995., 5005.])   
    fig.savefig('binarymask.pdf')