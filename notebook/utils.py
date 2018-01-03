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
    
def xcorr(data, model, ivars):
    """
    `data`: `[M]` array of pixels
    `model`: `[M]` array of predicted pixels
    `ivars`: `[M]` array of inverse variance values

    Note: Presumes `ivars` is a list of diagonal entries; no
      capacity for a dense inverse-variance matrix.
    """
    dd = data - np.mean(data)
    mm = model - np.mean(model)
    return np.sum(dd * ivars * mm) / np.sqrt(np.sum(dd * ivars * dd) * np.sum(mm * ivars * mm))

def get_objective_on_grid(data, ivars, template, args, method, guess, halfwidth):
    """
    `data`: `[M]` array of pixel values
    `ivars`: matched array of inverse variance values
    `args`: list of inputs to go into `template` function after `rv`
    `template`: function that makes the spectral template or mask
    `method`: objective-function generator; currently `xcorr` or `chisq`
    `guess`: where (in wavelength) to center the grid
    `halfwidth`: half-width of the grid to make
    """
    drv = 25. # m/s grid size
    rvs = np.arange(guess - halfwidth + 0.5 * drv, guess + halfwidth, drv) # m/s grid values
    objs = np.zeros_like(rvs)
    for i, rv in enumerate(rvs):
        model = template(rv, *args)
        objs[i] = method(data, model, ivars)
    return rvs, objs

def quadratic_max(xs, ys):
    """
    Find the maximum from a list, using a quadratic interpolation.
    Note: REQUIRES (and `assert`s) that the xs grid is uniform.
    """
    delta = xs[1] - xs[0]
    assert np.allclose(xs[1:] - xs[:-1], delta), "quadratic_max: not uniform grid!"
    ii = np.nanargmax(ys)
    if ii == 0:
        print("quadratic_max: warning: grid edge")
        return xs[ii]
    if (ii + 1) == len(ys):
        print("quadratic_max: warning: grid edge")
        return xs[ii]
    return xs[ii] + 0.5 * delta * (ys[ii-1] - ys[ii+1]) / (ys[ii-1] - 2. * ys[ii] + ys[ii+1])

def make_mask(rv, xs, ws, ms, w1, w2s):
    """
    `rv`: radial velocity in m/s (or same units as `c` above
    `xs`: [M] array of wavelength values
    `ws`: weights to apply to the binary mask tophats
    `ms`: locations of the centers of the binary mask tophats in rest wavelength
    `w1`: half-width of a wavelength pixel
    `w2s`: half-widths of the binary mask tophat in rest wavelength

    Notes: The doppler "shift" dilates the input wavelengths,
      and also the w1 half-width.
    
    Bugs: Super-inefficient code.
    """
    assert w1 > 0
    #assert w2 > 0
    drv = doppler(rv)
    model = np.zeros_like(xs)
    for ww, mm, w2 in zip(ws, ms, w2s):
        ddd = np.abs(w1 * drv - w2)
        dd = w1 * drv + w2
        xmms = xs * drv - mm
        dmodel = np.zeros_like(xs)
        dmodel[xmms > -dd] = (dd + xmms[xmms > -dd]) / (dd - ddd)
        dmodel[xmms > -ddd] = 1.
        dmodel[xmms > ddd] = (dd - xmms[xmms > ddd]) / (dd - ddd)
        dmodel[xmms > dd] = 0.
        model += ww * dmodel
    return 1. - model

def binary_xcorr(guess_rvs, xs, data, ivars, dx, ms=None, harps_mask=True,
                 mask_file='../code/G2.mas'):
    (N,M) = np.shape(data)
    x_lo, x_hi = min(xs), max(xs)
    if harps_mask:
        # load HARPS mask
        mask_wis, mask_wfs, harps_ws = np.loadtxt(mask_file,unpack=True,dtype=np.float64)
        ind = (mask_wis >= x_lo) & (mask_wfs <= x_hi)  # keep only relevant lines
        if len(ind) <= 1:
            print "Not enough lines found in this wavelength region."
            return None
        mask_wis, mask_wfs, harps_ws = mask_wis[ind], mask_wfs[ind], harps_ws[ind]
        harps_hws = (mask_wis - mask_wfs) / 2.
        harps_ms =  (mask_wis + mask_wfs) / 2.
        #if logflux:
        #    harps_ws = np.log(harps_ws)
        args = (xs, harps_ws, harps_ms, 0.5 * dx, harps_hws)
    else:
        halfwidth = 0.06 # A; half-width of binary mask
        hws = np.zeros_like(ms) + halfwidth
        ws = np.ones_like(ms)
        args = (xs, ws, ms, 0.5 * dx, hws)
    rvs_0 = np.zeros(N)
    for n in range(N):
        rvs, objs = get_objective_on_grid(data[n], ivars[n], make_mask, args, xcorr, guess_rvs[n], 1024.)
        rvs_0[n] = quadratic_max(rvs, objs)
    return rvs_0
