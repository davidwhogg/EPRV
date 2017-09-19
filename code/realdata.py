"""
This code is part of the EPRV project.
Copyright 2017 David W Hogg (NYU), Megan Bedell (Flatiron, UChicago).

"""

import numpy as np
import matplotlib.pyplot as plt
from fakedata import *
from scipy.interpolate import interp1d

c = 299792458. # m/s

def make_template_and_pca(all_data, rvs, xs, dx, plot=False, plotname='template.png'):
    """
    `all_data`: `[N, M]` array of pixels
    `rvs`: `[N]` array of RVs
    `xs`: `[M]` array of wavelength values
    `dx`: linear spacing desired for template wavelength grid (A)
    """
    (N,M) = np.shape(all_data)
    all_xs = np.empty_like(all_data)
    rectangular_data = np.empty_like(all_data)
    for i in range(N):
        all_xs[i,:] = xs * doppler(rvs[i]) # shift to rest frame
        f = interp1d(all_xs[i,:], all_data[i,:]) # interpolation is bad
        rectangular_data[i,:] = f(xs)
    bad = np.isnan(rectangular_data)
    rectangular_data[bad] = 1.0
    weights = np.ones_like(rectangular_data)
    weights[bad] = 0.0
    template_ys = np.exp(np.sum(weights * np.log(rectangular_data), axis=0) / np.sum(weights, axis=0)) # horrifying
    
    if plot == True:
        plt.clf()
        plt.scatter(all_xs, all_data, marker=".", alpha=0.25)
        plt.plot(xs, template_ys, color='black', lw=2)
        plt.title('Fitting a template to all data')
        plt.savefig(plotname)
    
    amps = np.zeros(N)
    pca1 = np.zeros(M)    
    rectangular_resids = np.log(rectangular_data) - np.log(template_ys[None,:])
    niter = 3
    for iter in range(niter):
        rectangular_resids[bad] = (amps[:,None] * pca1[None,:])[bad]
        u, s, v = np.linalg.svd(rectangular_resids, full_matrices=False)
        pca1 = v[0,:]
        amps = u[:,0] * s[0]
        
    for n in range(N):
        if n in [0,10,20]:
            # plot
            plt.clf()
            plt.plot(xs, np.exp(rectangular_resids[n,:]), color='k')
            plt.plot(xs, np.exp(amps[n] * pca1), color='red')                
            plt.title('resids + first PCA component for epoch #{0}'.format(n))
            plt.savefig(plotprefix+'_pcaresids{0}.png'.format(n))
    
    return template_ys, pca1, amps

if __name__ == "__main__":
    harps_mask = True # mask choice
    
    # load the data for quiet star HIP54287 (HARPS RMS 1.3 m/s)
    # BUGS: directories and dependencies will only work on Megan's computer...
    from scipy.io.idl import readsav
    from wobble import read_harps, rv_model    
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    s = readsav(data_dir+'HIP54287_result.dat')
    Star = rv_model.RV_Model()
    
    if True:
        # grab a little section from s1d files
        true_rvs = ( -s.berv + s.rv - 54.9) * 1.e3  # m/s
        #true_rvs = (s.rv - 54.9) * 1.e3 
        drift = s.drift # m/s
        dx = 0.01 # A
        #xs = np.arange(4998.0, 5002.0, dx)
        xs = np.arange(5910.0, 5925.0, dx) # tellurics region
        N = len(s.files)  # number of epochs
        M = len(xs)
        data = np.empty((N, M))
        ivars = np.empty_like(data)
        for n,(f,b,snr) in enumerate(zip(s.files, s.berv, s.snr)):
            # read in the spectrum
            spec_file = str.replace(f, 'ccf_G2', 's1d')
            wave, spec = read_harps.read_spec(spec_file)
            # re-introduce barycentric velocity
            wave *= doppler(b*1.e3)
            # remove systemic RV shift so we're looking at the same lines as example
            wave *= doppler(54.9 * 1.e3)
            # save the relevant bit
            f = interp1d(wave, spec)
            data[n,:] = f(xs)
            ivars[n,:] = snr**2
            
        p0 = None # starting guess for continuum normalization
        iterate = 1
        plotprefix = 'harpsdata'
        
        # make a mask since HARPS doesn't give one for this region:
        fwhms = [0.1299, 0.1309, 0.1409, 0.2118, 0.1251, 0.1206, 
                    0.1264, 0.1264] # FWHM of Gaussian fit to line (A)
        sigs = np.asarray(fwhms) / 2. / np.sqrt(2. * np.log(2.)) # Gaussian sigma (A)
        ms = [5905.675, 5906.842, 5909.979, 5914.165, 5916.255, 5922.117,
                    5927.792, 5929.679] # line center (A)
        ds = [-0.464135, -0.169625, -0.29082, -0.652963, -0.472075, -0.231232,
                    -0.36773, -0.354367] # depth of line center (normalized flux)
    
    if False:
        # grab a full order from e2ds files
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
        #p0 = [-1.e9, 4.e5, -3.e1] # starting guess for continuum normalization
        p0 = None
        iterate = 2
        plotprefix = 'harpsorder'
    
    raw_data = np.copy(data)
    # continuum normalize
    for n in range(N):
        if n < 4:
            data[n, :], ivars[n, :] = continuum_normalize(xs, data[n, :], ivars[n, :], plot=True,
                plotname=plotprefix+"_normalization{0}.png".format(n), p0=p0, iterate=iterate)
        else:
            data[n, :], ivars[n, :] = continuum_normalize(xs, data[n, :], ivars[n, :], 
                p0=p0, iterate=iterate)
    '''
    for n in range(N):
        continuum = np.percentile(data[n,:], 95.)
        data[n, :] /= continuum
        ivars[n, :] *= continuum**2
    '''''
    
    # plot the data    
    plot_data(xs, data, tellurics=False, plotname=plotprefix+"_data.png")
    
    # move to log(flux)
    #data = np.log(data)
    #ivars = -0.5 * np.log(ivars)
    #ivars = np.log(ivars)
    #logflux = True
    logflux = False
    
    # make a perfect template from stacked observations
    template_xs, template_ys = make_template(data, true_rvs, xs, dx, plot=True, 
                    plotname=plotprefix+'_perfecttemplate.png')
                    
    if True:
        # tellurics experiment
        
        # adopt CRLB from fake data experiment
        crlb = 10.5 # m/s
    
        # compute first-guess RVs with binary mask
        guess_rvs = -s.berv * 1.e3
        #ms = [0.0]
        #ws = np.ones_like(ms)
        rvs_0 = binary_xcorr(guess_rvs, xs, data, ivars, dx, ms, logflux=logflux, 
                    harps_mask=False, mask_file='G2.mas', plotprefix=plotprefix)
        #rvs_0 = guess_rvs
    
        '''if True:
            # plot an example binary mask
            plot_d = np.copy(data[4,:])
            plot_x = np.copy(xs) * doppler(true_rvs[4])
            plot_mask(plot_x, plot_d, 'G2.mas', ms, ws, plotprefix=plotprefix)
                    '''
    
        plot_resids(rvs_0, true_rvs, crlb=crlb, title='pre-telluric correction: binary mask xcorr', 
                    plotname=plotprefix+'_pretelluric_rv_mistakes.png')
        rms = np.sqrt(np.nanvar(rvs_0 - true_rvs, ddof=1)) # m/s    
        print "pre-telluric correction: RV RMS = {0:.2f} m/s".format(rms)
        
        # get telluric template
        telluric_data = np.copy(data)
        for n in range(N):
            if n in [0,10,20]:
                # plot
                plt.clf()
                plt.plot(xs, data[n,:], color='k')
                plt.plot(xs, shift_template(rvs_0[n], xs, template_xs, template_ys), color='red')                
                plt.title('data + shifted stellar template for epoch #{0}'.format(n))
                plt.savefig(plotprefix+'_dividetemplate{0}.png'.format(n))
            
            shifted_template = shift_template(rvs_0[n], xs, template_xs, template_ys)
            #data[n, :] /= shifted_template
            telluric_data[n, :] = np.exp(np.log(data[n, :]) - np.log(shifted_template))
            
        plot_data(xs, telluric_data, tellurics=False, plotname=plotprefix+"_tellurics.png")
        
        telluric_rvs = np.zeros_like(true_rvs)
        telluric_xs = xs
        telluric_ys, pca1, amps = make_template_and_pca(telluric_data, telluric_rvs, xs, dx, plot=True, 
                    plotname=plotprefix+'_tellurictemplate.png')
                    
        telluric_ys, _ = continuum_normalize(telluric_xs, telluric_ys, ivars[0, :], plot=True,
            plotname=plotprefix+"_telluricnorm.png", p0=p0, iterate=iterate)
        
        # shift & subtract tellurics from star
        for n in range(N):
            if n in [0,10,20]:
                # plot
                plt.clf()
                plt.plot(xs, data[n,:], color='k')
                this_telluric = np.exp(np.log(telluric_ys) + amps[n] * pca1)
                plt.plot(xs, shift_template(telluric_rvs[n], xs, telluric_xs, this_telluric), color='blue', alpha=0.7)                
                plt.plot(xs, shift_template(telluric_rvs[n], xs, telluric_xs, telluric_ys), color='red', alpha=0.5)                
                plt.title('data + shifted telluric template for epoch #{0}'.format(n))
                plt.savefig(plotprefix+'_dividetellurics{0}.png'.format(n))
                
            
            shifted_tellurics = shift_template(telluric_rvs[n], xs, telluric_xs, telluric_ys)
            #data[n, :] /= shifted_template
            data[n, :] = np.exp(np.log(data[n, :]) - np.log(shifted_tellurics)) 
            
        # plot the data    
        plot_data(xs, data, tellurics=False, plotname=plotprefix+"_posttelluric_data.png")
    
        # make a template from stacked observations
        template_xs, template_ys = make_template(data, true_rvs, xs, dx, plot=True, 
                        plotname=plotprefix+'_posttelluric_template.png')
            
        # get RVs from cleaned stellar spectra:
        rvs_1 = binary_xcorr(rvs_0, xs, data, ivars, dx, ms, logflux=logflux, 
                    harps_mask=False, mask_file='G2.mas', plotprefix=plotprefix)
        plot_resids(rvs_1, true_rvs, crlb=crlb, title='post-telluric correction: binary mask xcorr', 
                    plotname=plotprefix+'_posttelluric_rv_mistakes.png')
        rms = np.sqrt(np.nanvar(rvs_1 - true_rvs, ddof=1)) # m/s    
        print "post-telluric correction: RV RMS = {0:.2f} m/s".format(rms)
        
        if False:
        
            # subtract both star and tellurics from non-continuum normalized spectra:
            for n in range(N):
               if n in [0,10,20]:
                   # plot
                   plt.clf()
                   plt.plot(xs, raw_data[n,:], color='k')
                   plt.plot(xs, shift_template(telluric_rvs[n], xs, telluric_xs, telluric_ys), color='red')                
                   plt.plot(xs, shift_template(rvs_0[n], xs, template_xs, template_ys), color='blue')                
                   plt.title('data + shifted templates for epoch #{0}'.format(n))
                   plt.savefig(plotprefix+'_divideall{0}.png'.format(n))
           
               shifted_tellurics = shift_template(telluric_rvs[n], xs, telluric_xs, telluric_ys)
               #data[n, :] /= shifted_template
               raw_data[n, :] = np.exp(np.log(raw_data[n, :]) - np.log(shifted_tellurics)) 
           
               shifted_template = shift_template(rvs_1[n], xs, template_xs, template_ys)
               #data[n, :] /= shifted_template
               raw_data[n, :] = np.exp(np.log(raw_data[n, :]) - np.log(shifted_template))
           
               plot_data(xs, raw_data, tellurics=False, plotname=plotprefix+"_continuum_data.png")
           

                    
    if False: 
        # try to get RVs
        
        # adopt CRLB from fake data experiment
        crlb = 10.5 # m/s
    
        # compute first-guess RVs with binary mask
        guess_rvs = s.berv * 1.e3
        ms = [0.0]
        #ws = np.ones_like(ms)
        rvs_0 = binary_xcorr(guess_rvs, xs, data, ivars, dx, ms, logflux=logflux, 
                    harps_mask=True, mask_file='G2.mas', plotprefix=plotprefix)
    
        '''if True:
            # plot an example binary mask
            plot_d = np.copy(data[4,:])
            plot_x = np.copy(xs) * doppler(true_rvs[4])
            plot_mask(plot_x, plot_d, 'G2.mas', ms, ws, plotprefix=plotprefix)
                    '''
    
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

    