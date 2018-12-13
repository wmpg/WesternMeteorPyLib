""" Method for fitting the population and mass index based on fitting the gamma distribution on observed
    data by MLE method and extracting the slope in the magnitude completeness region. 
"""

from __future__ import print_function, division, absolute_import


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.misc


def line(x, m, k):
    return m*x + k

def logline(x, m, k):
    return 10**line(x, m, k)


def estimateIndex(input_data, mass=False, show_plots=False, nbins=None):
    """ Estimate the mass or population index from the meteor data by fitting a gamma function to it using
        MLE and estimating the slope in the completeness region.

    Arguments:
        input_data: [array like] List of peak magnitudes or logarithms of mass (in kg).

    Keyword arguments:
        mass: [bool] If true, the mass index will be computed. False by default, in which case the population
            index is computed.
        show_plots: [bool] Set to True to show the plots. False by default.
        nbins: [int] Number of bins for the histogram. None by default, in which case it will be estimated
            as the square root of the total number of data points.
    """


    input_data = np.array(input_data).astype(np.float64)

    # Reverse the signs of magnitudes to conform with the gamma distribution definition
    if not mass:
        input_data = -input_data
    

    ### FITTING ###

    # Fit a chi-squared distribution to the data
    df = 4
    params = scipy.stats.gamma.fit(input_data, df)

    # Kolmogorov-Smirnov test
    kstest = scipy.stats.kstest(input_data, scipy.stats.gamma.cdf, params)

    # Compute the derivative of the PDF
    x_arr = np.linspace(np.min(input_data), np.max(input_data), 100)
    pdf_derivative = scipy.misc.derivative(scipy.stats.gamma.pdf, x_arr, dx=0.01, args=params)

    # Find the minimum of the PDF derivative
    turnover_point = x_arr[pdf_derivative.argmin()]

    if not mass:
        # Take one magnitude brighter than the turnover point (the addition here is because the turnover point 
        #   is reverse)
        ref_point = turnover_point + 1

    else:
        ref_point = turnover_point + 0.4


    # Find the slope of the log survival function
    def logsf(*args, **kwargs):
        return np.log10(scipy.stats.gamma.sf(*args, **kwargs))

    slope = 1.0/(10**scipy.misc.derivative(logsf, ref_point, dx=0.01, args=params))
    slope = np.log10(slope)


    # Compute the slope for reporting, the sign for plotting
    if mass:
        slope_report = 1 + slope
        sign = 1
    else:
        slope_report = 10**slope
        sign = -1
        

    ### ###


    if show_plots:

        if mass:
            xlabel = 'Log of mass (kg)'

        else:
            xlabel = 'Magnitude'

        ### PLOTTING ###

        # Compute the number of bins for the histogram
        if nbins is None:
            nbins = int(np.ceil(np.sqrt(len(input_data))))

        # Find the slope at the reference point for PDF plotting
        slope_pdf = scipy.stats.gamma.pdf(ref_point, *params)

        plt.hist(sign*input_data, bins=nbins, density=True, color='k', histtype='step')
        plt.plot(sign*x_arr, scipy.stats.gamma.pdf(x_arr, *params))
        plt.scatter(sign*ref_point, slope_pdf, color='r', zorder=3, \
            label='Reference point = {:.2f}'.format(sign*ref_point))
        plt.scatter(sign*turnover_point, scipy.stats.gamma.pdf(turnover_point, *params), color='r', marker='x', \
            zorder=3, label='Turnover point = {:.2f}'.format(sign*turnover_point))

        plt.ylabel('Normalized count')
        plt.legend()

        plt.xlabel(xlabel)

        plt.show()


        ####

        # Compute intercept of the line on the cumulative plot
        y_temp = np.log10(scipy.stats.gamma.sf(ref_point, *params))
        intercept = y_temp + slope*ref_point

        plt.hist(sign*input_data, bins=nbins, cumulative=-sign, density=True, log=True, histtype='step', color='k')
        plt.plot(sign*x_arr, scipy.stats.gamma.sf(x_arr, *params))

        # Plot the turnover point
        plt.scatter(sign*ref_point, 10**y_temp, c='r', zorder=3, \
            label='Reference point = {:.2f}'.format(sign*ref_point))

        plt.plot(sign*x_arr, logline(-x_arr, slope, intercept), color='k', \
            label='Slope = {:.2f}'.format(slope_report), linestyle='--')


        plt.xlabel(xlabel)
        plt.ylabel('Cumulative count')
        plt.legend()

        plt.show()




    return params, sign*ref_point, sign*turnover_point, slope_report, kstest