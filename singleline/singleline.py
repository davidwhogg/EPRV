"""
This code is part of the EPRV project.
Copyright 2017 David W Hogg (NYU).
"""

import numpy as np

c = 299792458. # m/s

def doppler(rv):
    beta = rv / c
    return np.sqrt((1. - beta) / (1. + beta))

def oned_gaussian(xs, mm, sig):
    return np.exp(-0.5 * (xs - mm) ** 2 / sig ** 2) / np.sqrt(2. * np.pi * sig)

def make_synth(rv, xs, mm, sig):
    """
    `rv`: radial velocity in m/s (or same units as `c` above
    `xs`: `[M]` array of wavelength values
    `mm`: location of the line center in rest wavelength
    `sig`: sigma of pixel-convolved LSF
    """
    synths = np.ones_like(xs)
    synths -= oned_gaussian(xs * doppler(rv), mm, sig) / 8.
    return synths

def make_mask(rv, xs, mm, w1, w2):
    """
    `rv`: radial velocity in m/s (or same units as `c` above
    `xs`: [M] array of wavelength values
    `mm`: location of the center of the binary mask tophat in rest wavelength
    `w1`: half-width of a wavelength pixel
    `w2`: half-width of the binary mask tophat in rest wavelength
    """
    assert w1 > 0
    assert w2 > 0
    drv = doppler(rv)
    ddd = np.abs(w1 * drv - w2)
    dd = w1 * drv + w2
    xmms = xs * drv - mm
    model = np.zeros_like(xs)
    model[xmms > -dd] = (dd + xmms[xmms > -dd]) / (dd - ddd)
    model[xmms > -ddd] = 1.
    model[xmms > ddd] = (dd - xmms[xmms > ddd]) / (dd - ddd)
    model[xmms > dd] = 0.
    return 1. - model

def dsynth_dv(rv, xs, mm, sig):
    dv = 10. # m/s
    f2 = make_synth(rv + dv, xs, mm, sig)
    f1 = make_synth(rv - dv, xs, mm, sig)
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
        data[n, :] = make_synth(0., xs, mm, sig)
        data[n, :] += np.random.normal(size=M) / np.sqrt(ivar[n, :])
    return data, ivar

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
    return -0.5 * np.sum((data - model) * ivar * (data - model))

def get_objective_on_grid(data, ivar, template, args, method):
    """
    `data`: `[M]` array of pixel values
    `ivar`: matched array of inverse variance values
    `args`: list of inputs to go into `template` function after `rv`
    `template`: function that makes the spectral template or mask
    `method`: objective-function generator; currently `xcorr` or `chisq`
    """
    drv = 25.
    rvs = np.arange(-600. + 0.5 * drv, 600., drv)
    objs = np.zeros_like(rvs)
    for i, rv in enumerate(rvs):
        model = template(rv, *args)
        objs[i] = method(data, model, ivar)
    return rvs, objs

def quadratic_max(xs, ys):
    """
    Find the maximum from a list, using a quadratic interpolation.
    REQUIRES that the xs grid is uniform.
    """
    delta = xs[1] - xs[0]
    assert np.allclose(xs[1:] - xs[:-1], delta), "quadratic_max: not uniform grid!"
    ii = np.argmax(ys)
    if ii == 0:
        return xs[ii]
    if (ii + 1) == len(ys):
        return xs[ii]
    return xs[ii] + 0.5 * delta * (ys[ii-1] - ys[ii+1]) / (ys[ii-1] - 2. * ys[ii] + ys[ii+1])

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
    dmodel_dv = dsynth_dv(rv, xs, mm, sig)
    plt.clf()
    drv = 1000. # m/s
    rvs = np.arange(-5000. + 0.5 * drv, 5000., drv)
    for rv in rvs:
        model = make_synth(rv, xs, mm, sig)
        plt.plot(xs, model, "k-")
    plt.plot(xs, 4096. * dmodel_dv, "r-")
    plt.title("models and velocity derivative (times 4096)")
    plt.savefig("synth.png")

    # compute CRLBs
    rv = 0.
    dmodel_dv = dsynth_dv(rv, xs, mm, sig)
    crlbs = np.zeros(N)
    for n in range(N):
        crlbs[n] = np.sum(dmodel_dv * ivar[n, :] * dmodel_dv)
    crlb = 1. / np.sqrt(np.mean(crlbs))
    print("CRLB:", crlb, "m/s")

    # get best-fit velocities
    width = 0.075
    options = ([make_synth, (xs, mm, sig), xcorr],
               [make_synth, (xs, mm, sig), chisq],
               [make_mask, (xs, mm, 0.5 * dx, width), xcorr],
               [make_mask, (xs, mm, 0.5 * dx, width), chisq])
    best_rvs = np.zeros((N, len(options)))
    for n in range(N):
        if n == 0:
            plt.clf()
        for j, (template, args, method) in enumerate(options):
            rvs, objs = get_objective_on_grid(data[n], ivar[n], template, args, method)
            best_rvs[n,j] = quadratic_max(rvs, objs)
            if n == 0:
                plt.plot(rvs, objs, marker=".", alpha=0.5)
                plt.axvline(best_rvs[n, j], alpha=0.5)
        if n == 0:
            plt.title("grids of objective values")
            plt.savefig("objective.png")

    # plot best_rvs
    for j in range(len(options)):
        plt.clf()
        n, bins, patches = plt.hist(best_rvs[:, j], 50, normed=1, color="k", alpha=0.75)
        plt.plot(bins, oned_gaussian(bins, 0., crlb) / (bins[2] - bins[1]), "r-")
        plt.title("measured RVs")
        plt.savefig("best_rvs_{:02d}.png".format(j))

    # make and plot mask
    plt.clf()
    for rv in rvs:
        mask = make_mask(rv, xs, mm, 0.075, 0.5 * dx)
        plt.step(xs, mask + 0.0011 * rv, "k-")
    plt.title("pixel-convolved binary masks")
    plt.savefig("mask.png")
