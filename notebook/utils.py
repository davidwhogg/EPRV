"""
This code is part of the EPRV project.
Copyright 2017,2018 David W Hogg (NYU) & Megan Bedell (Flatiron).
"""

import numpy as np

c = 299792458. # m/s
np.random.seed(42)
sqrttwopi = np.sqrt(2. * np.pi)

def doppler(rv):
    beta = rv / c
    return np.sqrt((1. - beta) / (1. + beta))

def oned_gaussian(xs, mm, sig):
    return np.exp(-0.5 * (xs - mm) ** 2 / sig ** 2) / (sqrttwopi * sig)

def make_synth(rv, xs, ds, ms, sigs):
    """
    Generate a noiseless synthetic spectrum at given rv shift.
    `rv`: radial velocity in m/s (or same units as `c` above)
    `xs`: `[M]` array of wavelength values
    `ds`: depths at line centers
    `ms`: locations of the line centers in rest wavelength
    `sigs`: Gaussian sigmas of lines
    --
    We take the view that lines multiply into the spectrum.
    """
    lnsynths = np.zeros_like(xs)
    for d, m, sig in zip(ds, ms, sigs):
        lnsynths += (d * oned_gaussian(xs * doppler(rv), m, sig))
    return np.exp(lnsynths)

def make_data(N, xs, ds, ms, sigs, snr):
    """
    Generate a set of N synthetic spectra.
    `N`: number of spectra to make
    `xs`: `[M]` array of wavelength values
    `ds`: depth-like parameters for lines
    `ms`: locations of the line centers in rest wavelength
    `sigs`: Gaussian sigmas of lines
    `snr`: desired SNR per pixel
    """
    M = len(xs)
    data = np.zeros((N, M))
    ivars = np.zeros((N, M))
    rvs = 30000. * np.random.uniform(-1., 1., size=N) # 30 km/s bc Earth ; MAGIC
    for n, rv in enumerate(rvs):
        ivars[n, :] = snr**2.
        data[n, :] = make_synth(rv, xs, ds, ms, sigs)
        data[n, :] += np.random.normal(size=M) / np.sqrt(ivars[n, :])
    return data, ivars, rvs
    
def dsynth_dv(rv, xs, ds, ms, sigs):
    """
    Derivative of synthetic spectra w.r.t. rv
    """
    dv = 10. # m/s
    f2 = make_synth(rv + dv, xs, ds, ms, sigs)
    f1 = make_synth(rv - dv, xs, ds, ms, sigs)
    return (f2 - f1) / (2. * dv)

def calc_crlb(xs, ds, ms, sigs, N, ivars):
    dmodel_dv = dsynth_dv(0., xs, ds, ms, sigs)
    crlbs = np.zeros(N)
    for n in range(N): # average CRLB; averaging over true RV
        crlbs[n] = np.sum(dmodel_dv * ivars[n, :] * dmodel_dv)
    crlb = 1. / np.sqrt(np.mean(crlbs))
    return crlb