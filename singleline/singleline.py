"""
This code is part of the EPRV project.
Copyright 2017 David W Hogg (NYU).

## To-do:
- Optimize width of binary mask window.
- Look at errors as a function of synthetic-spectrum wrongness.
- Make a model with 2 lines or 4 to test scalings.
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

    Notes: The doppler "shift" dilates the input wavelengths,
      and also the w1 half-width.
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
    ivars = np.zeros((N, M))
    rvs = 30000. * np.random.uniform(-1., 1., size=N)
    for n, rv in enumerate(rvs):
        ivars[n, :] = 10000.
        data[n, :] = make_synth(rv, xs, mm, sig)
        data[n, :] += np.random.normal(size=M) / np.sqrt(ivars[n, :])
    return data, ivars, rvs

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

def chisq(data, model, ivars):
    """
    `data`: `[M]` array of pixels
    `model`: `[M]` array of predicted pixels
    `ivars`: `[M]` array of inverse variance values

    Note: Presumes `ivars` is a list of diagonal entries; no
      capacity for a dense inverse-variance matrix.
    """
    return -0.5 * np.sum((data - model) * ivars * (data - model))

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
    ii = np.argmax(ys)
    if ii == 0:
        print("quadratic_max: warning: grid edge")
        return xs[ii]
    if (ii + 1) == len(ys):
        print("quadratic_max: warning: grid edge")
        return xs[ii]
    return xs[ii] + 0.5 * delta * (ys[ii-1] - ys[ii+1]) / (ys[ii-1] - 2. * ys[ii] + ys[ii+1])

if __name__ == "__main__":
    import pylab as plt
                         
    # set parameters
    np.random.seed(42)
    mm = 5000.0 # A
    sig = 0.05 # A
    dx = 0.01 # A
    xs = np.arange(4998. + 0.5 * dx, 5002., dx) # A

    # make and plot fake data
    N = 16
    data, ivars, true_rvs = make_data(N, xs, mm, sig)
    plt.clf()
    for n in range(8):
        plt.step(xs, data[n, :] + 0.25 * n, color="k")
    plt.title("examples of (fake) data")
    plt.savefig("data.png")

    # make and plot model
    dmodel_dv = dsynth_dv(0., xs, mm, sig)
    plt.clf()
    drv = 1000. # m/s; just for plotting purposes
    rvs = np.arange(-5000. + 0.5 * drv, 5000., drv) # m/s; just for plotting purposes
    for rv in rvs:
        model = make_synth(rv, xs, mm, sig)
        plt.plot(xs, model, "k-")
    plt.plot(xs, 4096. * dmodel_dv, "r-")
    plt.title("models and velocity derivative (times 4096)")
    plt.savefig("synth.png")

    # compute CRLBs
    dmodel_dv = dsynth_dv(0., xs, mm, sig)
    crlbs = np.zeros(N)
    for n in range(N):
        crlbs[n] = np.sum(dmodel_dv * ivars[n, :] * dmodel_dv)
    crlb = 1. / np.sqrt(np.mean(crlbs))
    print("CRLB:", crlb, "m/s")

    # get best-fit velocities
    width = 0.075 # A
    options = ([make_synth, (xs, mm, sig), xcorr],
               [make_synth, (xs, mm, sig), chisq],
               [make_mask, (xs, mm, 0.5 * dx, width), xcorr],
               [make_mask, (xs, mm, 0.5 * dx, width), chisq])
    best_rvs = np.zeros((N, len(options)))
    for n in range(N):
        if n == 0:
            plt.clf()
        for j, (template, args, method) in enumerate(options):
            rvs, objs = get_objective_on_grid(data[n], ivars[n], template, args, method, true_rvs[n], 6000.)
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
        plt.plot(true_rvs, best_rvs[:, j], "k.")
        plt.title("method {}".format(j))
        plt.xlabel("true RVs")
        plt.ylabel("measured RVs")
        plt.savefig("rv_measurements_{:02d}.png".format(j))

        plt.clf()
        plt.plot(true_rvs, best_rvs[:, j] - true_rvs, "k.")
        plt.title("method {}".format(j))
        plt.xlabel("true RVs")
        plt.ylabel("RV mistake")
        plt.savefig("rv_mistakes_{:02d}.png".format(j))

    # make and plot mask
    plt.clf()
    for rv in rvs:
        mask = make_mask(rv, xs, mm, 0.075, 0.5 * dx)
        plt.step(xs, mask + 0.0011 * rv, "k-")
    plt.title("pixel-convolved binary masks")
    plt.savefig("mask.png")
