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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

c = 299792458. # m/s

def doppler(rv):
    beta = rv / c
    return np.sqrt((1. - beta) / (1. + beta))

def oned_gaussian(xs, mm, sig):
    return np.exp(-0.5 * (xs - mm) ** 2 / sig ** 2) / np.sqrt(2. * np.pi * sig)

def make_synth(rv, xs, ds, ms, sigs):
    """
    `rv`: radial velocity in m/s (or same units as `c` above
    `xs`: `[M]` array of wavelength values
    `ds`: depths at line centers
    `ms`: locations of the line centers in rest wavelength
    `sigs`: Gaussian sigmas of lines
    """
    synths = np.ones_like(xs)
    for d, m, sig in zip(ds, ms, sigs):
        synths *= np.exp(d *
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
    
def make_template(all_data, rvs, xs, dx, plot=False, plotname='template.png'):
    """
    `all_data`: `[N, M]` array of pixels
    `rvs`: `[N]` array of RVs
    `xs`: `[M]` array of wavelength values
    `dx`: linear spacing desired for template wavelength grid (A)
    """
    foo = 30.
    (N,M) = np.shape(all_data)
    all_xs = np.empty_like(all_data)
    for i in range(N):
        all_xs[i,:] = xs * doppler(rvs[i]) # shift to rest frame
    all_data, all_xs = np.ravel(all_data), np.ravel(all_xs)
    template_xs = np.arange(min(all_xs), max(all_xs), dx)
    template_ys = np.empty_like(template_xs)
    for i,t in enumerate(template_xs):
        ind = (all_xs >= t-dx/2.) & (all_xs < t+dx/2.)
        template_ys[i] = np.sum(all_data[ind]) / np.sum(ind)
    if plot == True:
        plt.clf()
        plt.scatter(all_xs, all_data, marker=".", alpha=0.25)
        plt.plot(template_xs, template_ys, color='black', lw=2)
        plt.title('Fitting a template to all data')
        plt.savefig(plotname)
    return template_xs, template_ys
    
def shift_template(rv, xs, template_xs, template_ys):
    f = interp1d(template_xs / doppler(rv), template_ys, bounds_error = False, 
            fill_value = 1.)
    return f(xs)
    

def dsynth_dv(rv, xs, ds, ms, sigs):
    dv = 10. # m/s
    f2 = make_synth(rv + dv, xs, ds, ms, sigs)
    f1 = make_synth(rv - dv, xs, ds, ms, sigs)
    return (f2 - f1) / (2. * dv)

def make_data(N, xs, ds, ms, sigs):
    """
    `N`: number of spectra to make
    `xs`: `[M]` array of wavelength values
    `ds`: depth-like parameters for lines
    `ms`: locations of the line centers in rest wavelength
    `sigs`: Gaussian sigmas of lines
    """
    M = len(xs)
    data = np.zeros((N, M))
    ivars = np.zeros((N, M))
    rvs = 30000. * np.random.uniform(-1., 1., size=N) # 30 km/s bc Earth ; MAGIC
    for n, rv in enumerate(rvs):
        ivars[n, :] = 10000. # s/n = 100 ; MAGIC
        data[n, :] = make_synth(rv, xs, ds, ms, sigs)
        data[n, :] += np.random.normal(size=M) / np.sqrt(ivars[n, :])
    return data, ivars, rvs
    
def add_tellurics(xs, all_data, true_rvs, lambdas, strengths, dx, plot=False, plotname='tellurics.png'):
    N, M = np.shape(all_data)
    tellurics = np.ones_like(xs)
    for ll, s in zip(lambdas, strengths):
        tellurics *= np.exp(-s * oned_gaussian(xs, ll, dx))
    all_data *= np.repeat([tellurics,],N,axis=0)
    if plot:
        plt.clf()
        plt.plot(xs, tellurics)
        plt.title('tellurics model')
        plt.savefig(plotname)
    return all_data
    
def quadratic(xs, p0, p1, p2):
    return p0 + p1*xs + p2*xs**2
    
def continuum_normalize(xs, data, ivars, percents=(80., 99.), plot=False, plotname='continuum_normalization.png'):
    """
    `xs`: `[M]` array of wavelength values
    `data`: `[M]` array of pixels
    `ivars`: `[M]` array of inverse variance values

    Note: Presumes `ivars` is a list of diagonal entries; no
      capacity for a dense inverse-variance matrix.
    """
    (lo, hi) = np.percentile(data, percents)
    ind_fit = (data >= lo) & (data <= hi)
    popt, pcov = curve_fit(quadratic, xs[ind_fit], data[ind_fit], 
                    sigma=np.sqrt(1./ivars[ind_fit]), absolute_sigma=True)
    continuum = quadratic(xs, popt[0], popt[1], popt[2])
    normalized_data = data / continuum
    normalized_ivars = ivars * continuum**2
    if plot:
        plt.clf()
        plt.scatter(xs, data, color='k', alpha=0.5)
        plt.scatter(xs[ind_fit], data[ind_fit], color='red')
        plt.plot(xs, continuum)
        plt.title('continuum normalization')
        plt.savefig(plotname)
    return normalized_data, normalized_ivars
    

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
    ii = np.nanargmax(ys)
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
        sigs = [0.05, ] # A
        dx = 0.01 # A
        xs = np.arange(4998. + 0.5 * dx, 5002., dx) # A
        plotprefix = "single"
    else:
        # set parameters
        np.random.seed(42)
        fwhms = [0.1077, 0.1113, 0.1044, 0.1083, 0.1364, 0.1, 0.1281,
                    0.1212, 0.1292, 0.1526, 0.1575, 0.1879] # FWHM of Gaussian fit to line (A)
        sigs = np.asarray(fwhms) / 2. / np.sqrt(2. * np.log(2.)) # Gaussian sigma (A)
        ms = [4997.967, 4998.228, 4998.543, 4999.116, 4999.508, 5000.206, 5000.348,
                5000.734, 5000.991, 5001.229, 5001.483, 5001.87] # line center (A)
        ds = [-0.113524, -0.533461, -0.030569, -0.351709, -0.792123, -0.234712, -0.610711,
                -0.123613, -0.421898, -0.072386, -0.147218, -0.757536] # depth of line center (normalized flux)
        ws = np.ones_like(ds) # dimensionless weights
        dx = 0.01 # A
        xs = np.arange(4998. + 0.5 * dx, 5002., dx) # A
        plotprefix = "realistic"

    # make fake data
    N = 128
    data, ivars, true_rvs = make_data(N, xs, ds, ms, sigs)
    
    # add tellurics
    if True:
        n_tellurics = 8
        np.random.seed(75)
        lambdas = np.random.uniform(xs[0], xs[-1], n_tellurics)
        strengths = 1. - np.random.power(700., n_tellurics)
        data = add_tellurics(xs, data, true_rvs, lambdas, strengths, dx, plot=True)
        np.random.seed(42)
        
    # plot the data    
    plt.clf()
    for n in range(8):
        plt.step(xs, data[n, :] + 0.25 * n, color="k")
    plt.title("examples of (fake) data")
    plt.savefig(plotprefix+"_data.png")
    
    # continuum normalize
    for n in range(N):
        if n < 4:
            data[n, :], ivars[n, :] = continuum_normalize(xs, data[n, :], ivars[n, :], plot=True,
                plotname=plotprefix+"_normalization{0}.png".format(n))
        else:
            data[n, :], ivars[n, :] = continuum_normalize(xs, data[n, :], ivars[n, :])
    
    # make a perfect template from stacked observations
    template_xs, template_ys = make_template(data, true_rvs, xs, dx, plot=True, 
                    plotname=plotprefix+'_perfecttemplate.png')

    # compute CRLBs
    dmodel_dv = dsynth_dv(0., xs, ds, ms, sigs)
    crlbs = np.zeros(N)
    for n in range(N): # average CRLB; averaging over true RV
        crlbs[n] = np.sum(dmodel_dv * ivars[n, :] * dmodel_dv)
    crlb = 1. / np.sqrt(np.mean(crlbs))
    print("CRLB:", crlb, "m/s")
    
    
    # compute first-guess RVs with binary mask
    halfwidth = 0.06 # A; half-width of binary mask
    guess_rvs = true_rvs + np.random.normal(0., 100., size=N) # add in some random dispersion
    args = (xs, ws, ms, 0.5 * dx, halfwidth)
    rvs_0 = np.zeros(N)
    for n in range(N):
        rvs, objs = get_objective_on_grid(data[n], ivars[n], make_mask, args, xcorr, guess_rvs[n], 1024.)
        rvs_0[n] = quadratic_max(rvs, objs)
        
    plt.clf()
    resid = rvs_0 - true_rvs
    plt.plot(true_rvs, resid - np.median(resid), "k.", alpha=0.5)
    plt.axhline(2. * crlb, color="k", lw=0.5, alpha=0.5)
    plt.axhline(-2. * crlb, color="k", lw=0.5, alpha=0.5)
    plt.title("round 0: binary mask xcorr")
    plt.xlabel("true RVs")
    plt.ylabel("RV mistake - offset")
    plt.ylim(-500., 500.)
    plt.savefig("round0_rv_mistakes.png")
    
    rms = np.sqrt(np.nanvar(rvs_0 - true_rvs, ddof=1)) # m/s    
    print "Round 0: RV RMS = {0:.2f} m/s".format(rms)
    
    # make a mask and iterate:
    n_iter = 3
    best_rvs = rvs_0
    for i in range(n_iter):
        template_xs, template_ys = make_template(data, best_rvs, xs, dx, plot=True, 
                    plotname='template_round{}.png'.format(i+1))
        args = (xs, template_xs, template_ys)
        for n in range(N):
            rvs, objs = get_objective_on_grid(data[n], ivars[n], shift_template, args, xcorr, best_rvs[n], 1024.)
            rv = quadratic_max(rvs, objs)  # update best guess
            if np.isfinite(rv):
                best_rvs[n] = rv
            
            
        rms = np.sqrt(np.nanvar(best_rvs - true_rvs, ddof=1)) # m/s    
        rmeds = np.sqrt(np.median((best_rvs - true_rvs) ** 2))
        print "Round {0}: RV RMS = {1:.2f} m/s".format(i+1, rms)
        
        plt.clf()
        resid = best_rvs - true_rvs
        plt.plot(true_rvs, resid - np.median(resid), "k.", alpha=0.5)
        plt.axhline(2. * crlb, color="k", lw=0.5, alpha=0.5)
        plt.axhline(-2. * crlb, color="k", lw=0.5, alpha=0.5)
        plt.title("round {}: stacked template xcorr".format(i+1))
        plt.xlabel("true RVs")
        plt.ylabel("RV mistake - offset")
        plt.ylim(-500., 500.)
        plt.savefig("round{}_rv_mistakes.png".format(i+1))
      


    if False:
        # comparative tests of methods        
        # get best-fit velocities
        halfwidth = 0.075 # A; half-width of binary mask
        options = ([make_synth, (xs, ds, ms, sigs), xcorr],
                   [make_synth, (xs, ds, ms, sigs), chisq],
                   [make_mask, (xs, ws, ms, 0.5 * dx, halfwidth), xcorr],
                   [make_mask, (xs, ws, ms, 0.5 * dx, halfwidth), chisq],
                   [shift_template, (xs, template_xs, template_ys), xcorr],
                   [shift_template, (xs, template_xs, template_ys), chisq])
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
            rms = np.sqrt(np.nanvar(best_rvs[:,j] - true_rvs, ddof=1)) # m/s
            titlestr = "{}: {} / {}: {:.2f} m/s".format(j, options[0].__name__, options[2].__name__, rms)
        
            print "{} / {}: {}".format(options[0].__name__, options[2].__name__, best_rvs[:,j])

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
            model = make_synth(rv, xs, ds, ms, sigs)
            plt.plot(xs, model + 0.0011 * rv, "k-")
        plt.plot(xs, 4096. * dmodel_dv, "r-")
        plt.title("models and velocity derivative (times 4096)")
        plt.savefig(plotprefix+"_synth.png")
    
    if False:
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
        
