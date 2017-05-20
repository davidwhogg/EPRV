# The information about a stellar radial velocity resident in a single absorption line.

*Note to self: Need better title!*

**David W. Hogg** / *CCPP, NYU Physics* / *CDS, NYU* / *MPIA* / *CCA, Flatiron Institute*

**Megan Bedell** / *University of Chicago*

**Abstract:** In the world of extreme precision radial-velocity measurements, enormous numbers of photons are gathered at very high spectral resolution to obtain supremely accurate measurements of stellar radial velocities. These projects have been incredibly successful, discovering and characterizing many hundreds of planets around other stars. A question arises late in the data-analysis chain: How best to measure a radial velocity of a star, given a good spectrum? Most pipelines adopt a cross-correlation methodology, using some kind of mask or template. Here we discuss the connection of these techniques to model fitting, make suggestions for the in-detail execution of the cross-correlations, and show how to build the cross-correlation template that delivers the highest possible precision on the radial velocity measurements. We quantitatively assess how much information is being lost from stellar velocity measurements because present-day projects are using improper templates. We find XXX and YYY. The analyses in this project are performed on a spectrum with a single spectral line; we discuss how to scale these up to full stellar spectra (with many thousands of lines).

## Introduction
Importance of EPRV; locations where information may be being lost in the data-analysis pipelines themselves. That is, the unconscionable losses. Introduce the concept of the CRLB and the Fisher information. Emphasize the great importance of having open data archives.

## Cross-correlation and maximum likelihood
How a matched filter is the same as maximum likelihood *under conditions*. How to do a cross-correlation to make sure these conditions are met. How to get an uncertainty out of a cross-correlation, and not just a best fit.

## Experiments with a single emission line
How much better you do with an accurate line shape than with a binary mask. Details of how you do this. Philosophical comments on the pixel-convolved line-spread function. How cross-correlation results degrade with line shape and line depth. On the latter, the issues of uncertainty estimation matter (because depth scales out).

## Scaling to full spectrum
Introduction to information theory as it applies to this project; connect to Fisher information. Discussion of the implications for the full stellar spectrum. Information loss from using binary masks, even in the best possible scenario. Importance of generating a good synthetic spectrum. Prospects for *Avast*.

## Discussion
What this might mean for present-day projects. Things that might limit the applicability of these results to real data sets. Address the apparent conflict with other experiments that have looked at these things.
