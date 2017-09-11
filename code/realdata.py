"""
This code is part of the EPRV project.
Copyright 2017 David W Hogg (NYU), Megan Bedell (Flatiron, UChicago).

"""

import numpy as np
import matplotlib.pyplot as plt
from fakedata import *

c = 299792458. # m/s

if __name__ == "__main__":
    harps_mask = True # mask choice
    
    # load the data for quiet star HIP54287 (HARPS RMS 1.3 m/s)
    # BUGS: directories and dependencies will only work on Megan's computer...
    from scipy.io.idl import readsav
    from wobble import read_harps, rv_model    
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    s = readsav(data_dir+'HIP54287_result.dat')
    Star = rv_model.RV_Model()
    
    if False:
        true_rvs = (s.berv + s.rv - 54.9) * 1.e3  # m/s
        drift = s.drift # m/s
        dx = 0.01 # A
        xs = np.arange(4987.4, 5043.5, dx)
        N = len(s.files)  # number of epochs
        M = len(xs)
        data = np.empty((N, M))
        ivars = np.empty_like(data)
        for n,(f,b,snr) in enumerate(zip(s.files, s.berv, s.snr)):
            # read in the spectrum
            spec_file = str.replace(f, 'ccf_G2', 's1d')
            wave, spec = read_harps.read_spec(spec_file)
            # re-introduce barycentric velocity
            wave /= doppler(b*1.e3)
            # remove systemic RV shift so we're looking at the same lines as example
            wave *= doppler(54.9 * 1.e3)
            # save the relevant bit
            f = interp1d(wave, spec)
            data[n,:] = f(xs)
            ivars[n,:] = snr**2
        plotprefix = 'harpsdata'
    
    if True:
        true_rvs = (s.berv + s.rv - 54.9) * 1.e3  # m/s
        drift = s.drift # m/s
        dx = 0.01 # A
        xs = np.arange(4990.0, 5040.0, dx)
        N = len(s.files)  # number of epochs
        M = len(xs)
        data = np.empty((N, M))
        ivars = np.empty_like(data)
        delete_ind = []
        for n,(f,b,snr) in enumerate(zip(s.files, s.berv, s.snr)):
            # read in the spectrum
            spec_file = str.replace(f, 'ccf_G2', 'e2ds')
            try:
                wave, spec = read_harps.read_spec_2d(spec_file)
                wave, spec = wave[39], spec[39]
            except:
                delete_ind = np.append(delete_ind, n)
                continue
            # re-introduce barycentric velocity
            wave /= doppler(b*1.e3)
            # remove systemic RV shift so we're looking at the same lines as example
            wave *= doppler(54.9 * 1.e3)
            # save the relevant bit
            f = interp1d(wave, spec)
            data[n,:] = f(xs)
            ivars[n,:] = snr**2
        # remove epochs without data:
        delete_ind = np.asarray(delete_ind, dtype=int)
        data = np.delete(data, delete_ind, axis=0)
        ivars = np.delete(ivars, delete_ind, axis=0)
        true_rvs = np.delete(true_rvs, delete_ind)
        drift = np.delete(drift, delete_ind)
        N -= len(delete_ind)
        
        #Star.get_data(s.files)
        #order_rvs = Star.data[:,order,1] 
        plotprefix = 'harpsorder'
    
    # continuum normalize
    for n in range(N):
        if n < 4:
            data[n, :], ivars[n, :] = continuum_normalize(xs, data[n, :], ivars[n, :], plot=True,
                plotname=plotprefix+"_normalization{0}.png".format(n))
        else:
            data[n, :], ivars[n, :] = continuum_normalize(xs, data[n, :], ivars[n, :])
    
    # plot the data    
    plot_data(xs, data, tellurics=False, plotname=plotprefix+"_data.png")
    
    
    # make a perfect template from stacked observations
    template_xs, template_ys = make_template(data, true_rvs, xs, dx, plot=True, 
                    plotname=plotprefix+'_perfecttemplate.png')   
    
    # adopt CRLB from fake data experiment
    crlb = 10.5 # m/s
    
    # compute first-guess RVs with binary mask
    guess_rvs = s.berv * 1.e3
    ms = [4997.967, 4998.228, 4998.543, 4999.116, 4999.508, 5000.206, 5000.348,
            5000.734, 5000.991, 5001.229, 5001.483, 5001.87] # line center (A)
    ws = np.ones_like(ms)
    rvs_0 = binary_xcorr(guess_rvs, xs, data, ivars, dx, ms, 
                harps_mask=True, mask_file='G2.mas', plotprefix=plotprefix)
    
    if True:
        # plot an example binary mask
        plot_d = np.copy(data[4,:])
        plot_x = np.copy(xs) * doppler(true_rvs[4])
        plot_mask(plot_x, plot_d, 'G2.mas', ms, ws, plotprefix=plotprefix)
    
    plot_resids(rvs_0, true_rvs, crlb=crlb, title='round 0: binary mask xcorr', 
                plotname=plotprefix+'_round0_rv_mistakes.png')
    rms = np.sqrt(np.nanvar(rvs_0 - true_rvs, ddof=1)) # m/s    
    print "Round 0: RV RMS = {0:.2f} m/s".format(rms)
    
    # make a mask and iterate:
    n_iter = 3
    best_rvs = rvs_0
    for i in range(n_iter):
        template_xs, template_ys = make_template(data, best_rvs, xs, dx, plot=True, 
                    plotname=plotprefix+'_template_round{}.png'.format(i+1))
        args = (xs, template_xs, template_ys)
        for n in range(N):
            rvs, objs = get_objective_on_grid(data[n], ivars[n], shift_template, args, xcorr, best_rvs[n], 1024.)
            rv = quadratic_max(rvs, objs)  # update best guess
            if np.isfinite(rv):
                best_rvs[n] = rv
            '''''
            if n == 0:
                plt.clf()
                plt.plot(rvs, objs, marker=".", alpha=0.5)
                plt.axvline(best_rvs[n], alpha=0.5)
            if n == 0:
                plt.title("grids of objective values")
                plt.savefig(plotprefix+"_objective_round{0}.png".format(i))
            '''
            
            
        rms = np.sqrt(np.nanvar(best_rvs - true_rvs, ddof=1)) # m/s    
        rmeds = np.sqrt(np.median((best_rvs - true_rvs) ** 2))
        print "Round {0}: RV RMS = {1:.2f} m/s".format(i+1, rms)
        
        plot_resids(best_rvs, true_rvs, crlb=crlb, title="round {}: stacked template xcorr".format(i+1), 
                    plotname=plotprefix+'_round{}_rv_mistakes.png'.format(i+1))
    
    rms = np.sqrt(np.nanvar(best_rvs - true_rvs - drift, ddof=1)) # m/s                    
    print "RV RMS after drift correction = {0:.2f} m/s".format(rms)

    