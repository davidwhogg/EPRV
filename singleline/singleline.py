"""
This code is part of the EPRV project.
Copyright 2017 David W Hogg (NYU).
"""

import numpy as np

c = 299792458. # m/s

def xcorr(data, model, ivar):
    """
    `data`: `[M]` array of pixels
    `model`: `[M]` array of predicted pixels
    `ivar`: `[M]` array of inverse variance values
    (presumes `ivar` is a list of diagonal entries)
    """
    dd = data - np.mean(data)
    mm = model - np.mean(model)
    return np.sum(dd * ivar * mm) / np.sqrt(np.sum(dd * ivar * dd) * np.sum(mm * ivar * mm))

def chisq(data, model, ivar):
    """
    `data`: `[M]` array of pixels
    `model`: `[M]` array of predicted pixels
    `ivar`: `[M]` array of inverse variance values
    (presumes `ivar` is a list of diagonal entries)
    """
    # presumes ivar is a list of diagonal entries
    return np.sum((data - model) * ivar * (data - model))

def mask_model(xs, mm, w1, w2):
    """
    `xs`: [M] array of wavelength values
    `mm`: location of the center of the binary mask tophat
    `w1`: half-width of a wavelength pixel
    `w2`: half-width of the binary mask tophat
    """
    assert w1 > 0
    assert w2 > 0
    ddd = np.abs(d1 - d2)
    dd = d1 + d2
    xmms = xs - mm
    model = np.zeros_like(xs)
    model[xmms > -dd] = (dd + xmms) / (dd - ddd)
    model[xmms > -ddd] = 1.
    model[xmms > ddd] = (dd - xmms) / (dd - ddd)
    model[xmms > dd] = 0.
    return model

def oned_gaussian(xs, mm, sig):
    return np.exp(-0.5 * (xs - mm) ** 2 / sig ** 2) / np.sqrt(2. * np.pi * sig)

def doppler(rv):
    beta = rv / c
    return np.sqrt((1. - beta) / (1. + beta))

def make_synthetic_spectrum(xs, rv, mm, sig):
    """
    `xs`: `[M]` array of wavelength values
    `rv`: radial velocity in m/s (or same units as `c` above
    `mm`: location of the line center in rest wavelength
    `sig`: sigma of pixel-convolved LSF
    """
    synths = np.ones_like(xs)
    synths -= oned_gaussian(xs * doppler(rv), mm, sig) / 8.
    return synths

def dsynthetic_dv(xs, rv, mm, sig):
    dv = 100. # m/s
    f2 = make_synthetic_spectrum(xs, rv + dv, mm, sig)
    f1 = make_synthetic_spectrum(xs, rv - dv, mm, sig)
    return (f2 - f1) / (2. * dv)

def make_data(N, xs, mm, sig):
    """
    `N`: number of spectra to make
    `xs`: `[M]` array of wavelength values
    `mm`: location of the line center in rest wavelength
    `sig`: sigma of pixel-convolved LSF
    """
    M = len(xs)
    data = np.zeros((N, M))
    ivar = np.zeros((N, M))
    for n in range(N):
        ivar[n, :] = 10000.
        data[n, :] = make_synthetic_spectrum(xs, 0., mm, sig)
        data[n, :] += np.random.normal(size=M) / np.sqrt(ivar[n, :])
    return data, ivar

if __name__ == "__main__":
    import pylab as plt

    # set parameters
    np.random.seed(42)
    mm = 5000.0 # A
    sig = 0.05 # A
    dx = 0.01 # A
    xs = np.arange(4999.5 + 0.5 * dx, 5000.5, dx)

    # make and plot fake data
    N = 8192
    data, ivar = make_data(N, xs, mm, sig)
    plt.clf()
    for n in range(8):
        plt.step(xs, data[n, :] + 0.25 * n, color="k")
    plt.title("examples of (fake) data")
    plt.savefig("data.png")

    # make and plot model
    rv = 0.
    model = make_synthetic_spectrum(xs, rv, mm, sig)
    dmodel_dv = dsynthetic_dv(xs, rv, mm, sig)
    plt.clf()
    plt.plot(xs, model, "k-")
    plt.plot(xs, 4096. * dmodel_dv, "r-")
    plt.title("model and velocity derivative (times 4096)")
    plt.savefig("synth.png")

    # compute CRLBs
    crlbs = np.zeros(N)
    for n in range(N):
        crlbs[n] = 1. / np.sqrt(np.sum(dmodel_dv * ivar[n, :] * dmodel_dv))
    print("CRLB:", np.mean(crlbs), "m/s")
