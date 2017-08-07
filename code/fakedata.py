"""
This code is part of the EPRV project.
Copyright 2017 David W Hogg (NYU).

## To-do:
- Make EW of line an input variable in make_synth and propagate.
- Optimize width of binary mask window.
-- Do this with a function that is general-purpose
- Look at errors as a function of synthetic-spectrum wrongness
-- in amplitude
-- in width
-- in line shape
- Consider continuum noise or wrongness
- Make a model with 2 lines or 4 to test scalings.
"""

import numpy as np

c = 299792458. # m/s

def doppler(rv):
    beta = rv / c
    return np.sqrt((1. - beta) / (1. + beta))

def oned_gaussian(xs, mm, sig):
    return np.exp(-0.5 * (xs - mm) ** 2 / sig ** 2) / np.sqrt(2. * np.pi * sig)

def make_synth(rv, xs, ds, ms, sig):
    """
    `rv`: radial velocity in m/s (or same units as `c` above
    `xs`: `[M]` array of wavelength values
    `ds`: depth-like (really EW) parameters for lines
    `ms`: locations of the line centers in rest wavelength
    `sig`: sigma of pixel-convolved LSF
    """
    synths = np.ones_like(xs)
    for d, m in zip(ds, ms):
        synths *= np.exp(-d *
            oned_gaussian(xs * doppler(rv), m, sig))
    return synths

def make_mask(rv, xs, ws, ms, w1, w2):
    """
    `rv`: radial velocity in m/s (or same units as `c` above
    `xs`: [M] array of wavelength values
    `ws`: weights to apply to the binary mask tophats
    `ms`: locations of the centers of the binary mask tophats in rest wavelength
    `w1`: half-width of a wavelength pixel
    `w2`: half-width of the binary mask tophat in rest wavelength

    Notes: The doppler "shift" dilates the input wavelengths,
      and also the w1 half-width.
    
    Bugs: Super-inefficient code.
    """
    assert w1 > 0
    assert w2 > 0
    drv = doppler(rv)
    ddd = np.abs(w1 * drv - w2)
    dd = w1 * drv + w2
    model = np.zeros_like(xs)
    for ww, mm in zip(ws, ms):
        xmms = xs * drv - mm
        dmodel = np.zeros_like(xs)
        dmodel[xmms > -dd] = (dd + xmms[xmms > -dd]) / (dd - ddd)
        dmodel[xmms > -ddd] = 1.
        dmodel[xmms > ddd] = (dd - xmms[xmms > ddd]) / (dd - ddd)
        dmodel[xmms > dd] = 0.
        model += ww * dmodel
    return 1. - model

def dsynth_dv(rv, xs, ds, ms, sig):
    dv = 10. # m/s
    f2 = make_synth(rv + dv, xs, ds, ms, sig)
    f1 = make_synth(rv - dv, xs, ds, ms, sig)
    return (f2 - f1) / (2. * dv)

def make_data(N, xs, ds, ms, sig):
    """
    `N`: number of spectra to make
    `xs`: `[M]` array of wavelength values
    `ds`: depth-like parameters for lines
    `ms`: locations of the line centers in rest wavelength
    `sig`: sigma of pixel-convolved LSF
    """
    M = len(xs)
    data = np.zeros((N, M))
    ivars = np.zeros((N, M))
    rvs = 30000. * np.random.uniform(-1., 1., size=N) # 30 km/s bc Earth ; MAGIC
    for n, rv in enumerate(rvs):
        ivars[n, :] = 10000. # s/n = 100 ; MAGIC
        data[n, :] = make_synth(rv, xs, ds, ms, sig)
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
    singleline = False
    
    if singleline:
        # set parameters
        np.random.seed(42)
        ds = [1. / 8., ] # EW units (A), sort-of
        ws = [1., ] # dimensionless weights
        ms = [5000.0, ] # A
        sig = 0.05 # A
        dx = 0.01 # A
        xs = np.arange(4998. + 0.5 * dx, 5002., dx) # A
        plotprefix = "single"
    else:
        # set parameters
        np.random.seed(42)
        ds = [1. / 2., 1. / 8., 1., 1. / 32.] # EW units (A), sort-of
        ws = [1., 1., 1., 1.] # dimensionless weights
        ms = np.arange(4999.25, 5001., 0.5)
        sig = 0.05 # A
        dx = 0.01 # A
        xs = np.arange(4998. + 0.5 * dx, 5002., dx) # A
        plotprefix = "four"

    # make and plot fake data
    N = 512
    data, ivars, true_rvs = make_data(N, xs, ds, ms, sig)
    plt.clf()
    for n in range(8):
        plt.step(xs, data[n, :] + 0.25 * n, color="k")
    plt.title("examples of (fake) data")
    plt.savefig(plotprefix+"_data.png")

    # compute CRLBs
    dmodel_dv = dsynth_dv(0., xs, ds, ms, sig)
    crlbs = np.zeros(N)
    for n in range(N): # average CRLB; averaging over true RV
        crlbs[n] = np.sum(dmodel_dv * ivars[n, :] * dmodel_dv)
    crlb = 1. / np.sqrt(np.mean(crlbs))
    print("CRLB:", crlb, "m/s")

    # get best-fit velocities
    halfwidth = 0.075 # A; half-width of binary mask
    options = ([make_synth, (xs, ds, ms, sig), xcorr],
               [make_synth, (xs, ds, ms, sig), chisq],
               [make_mask, (xs, ws, ms, 0.5 * dx, halfwidth), xcorr],
               [make_mask, (xs, ws, ms, 0.5 * dx, halfwidth), chisq])
    best_rvs = np.zeros((N, len(options)))
    for n in range(N):
        if n == 0:
            plt.clf()
        for j, (template, args, method) in enumerate(options):
            rvs, objs = get_objective_on_grid(data[n], ivars[n], template, args, method, true_rvs[n], 1024.)
            best_rvs[n,j] = quadratic_max(rvs, objs)
            if n == 0:
                plt.plot(rvs, objs, marker=".", alpha=0.5)
                plt.axvline(best_rvs[n, j], alpha=0.5)
        if n == 0:
            plt.title("grids of objective values")
            plt.savefig(plotprefix+"_objective.png")

    # plot best_rvs
    for j, options in enumerate(options):
        rms = np.sqrt(np.var(best_rvs[:,j] - true_rvs, ddof=1)) # m/s
        titlestr = "{}: {} / {}: {:.2f} m/s".format(j, options[0].__name__, options[2].__name__, rms)

        plt.clf()
        plt.plot(true_rvs, best_rvs[:, j], "k.", alpha=0.5)
        plt.title(titlestr)
        plt.xlabel("true RVs")
        plt.ylabel("measured RVs")
        plt.savefig(plotprefix+"_rv_measurements_{:02d}.png".format(j))

        plt.clf()
        plt.plot(true_rvs, best_rvs[:, j] - true_rvs, "k.", alpha=0.5)
        plt.axhline(2. * crlb, color="k", lw=0.5, alpha=0.5)
        plt.axhline(-2. * crlb, color="k", lw=0.5, alpha=0.5)
        plt.title(titlestr)
        plt.xlabel("true RVs")
        plt.ylabel("RV mistake")
        plt.ylim(-500., 500.)
        plt.savefig(plotprefix+"_rv_mistakes_{:02d}.png".format(j))

    # make and plot mask
    rvs -= np.min(rvs)
    plt.clf()
    for rv in rvs[::4]:
        mask = make_mask(rv, xs, ws, ms, 0.075, 0.5 * dx)
        plt.step(xs, mask + 0.0011 * rv, "k-")
    plt.title("pixel-convolved binary masks")
    plt.savefig(plotprefix+"_mask.png")

    # make and plot synth model
    plt.clf()
    for rv in rvs[::4]:
        model = make_synth(rv, xs, ds, ms, sig)
        plt.plot(xs, model + 0.0011 * rv, "k-")
    plt.plot(xs, 4096. * dmodel_dv, "r-")
    plt.title("models and velocity derivative (times 4096)")
    plt.savefig(plotprefix+"_synth.png")

    # look at rms as a function of binary-mask width
    method = xcorr
    template = make_mask
    tiny = 1. / 128.
    halfwidths = np.arange(1./64. + 0.5 * tiny, 1./8., tiny)
    best_rvs_hw = np.zeros((N, len(halfwidths)))
    for j,halfwidth in enumerate(halfwidths):
        print(j)
        for n in range(N):
            args = (xs, ws, ms, 0.5 * dx, halfwidth)
            rvs, objs = get_objective_on_grid(data[n], ivars[n], template, args, method, true_rvs[n], 1024.)
            best_rvs_hw[n,j] = quadratic_max(rvs, objs)

    # plot best_rvs
    plt.clf()
    plt.plot(halfwidths, np.sqrt(np.var(best_rvs_hw - true_rvs[:, None], axis=0, ddof=1)), "k.", alpha=0.5)
    plt.axhline(crlb, color="k", lw=0.5, alpha=0.5)
    plt.title("dependence on binary mask half-width")
    plt.xlabel("binary mask half-width")
    plt.ylabel("RV std")
    plt.ylim(0., 500.)
    plt.savefig(plotprefix+"_rv_std_hw.png".format(j))
