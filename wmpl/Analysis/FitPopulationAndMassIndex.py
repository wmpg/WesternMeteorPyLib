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



def fitSlope(input_data, mass):

    # Fit a gamma distribution to the data
    df = 10
    params = scipy.stats.gamma.fit(input_data, df, loc=np.min(input_data))

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


    return params, x_arr, turnover_point, ref_point, slope, slope_report, sign, kstest




def estimateIndex(input_data, mass=False, show_plots=False):
    """ Estimate the mass or population index from the meteor data by fitting a gamma function to it using
        MLE and estimating the slope in the completeness region.

    Arguments:
        input_data: [array like] List of peak magnitudes or logarithms of mass

    Keyword arguments:
        mass: [bool] If true, the mass index will be computed. False by default, in which case the population
            index is computed.
        show_plots: [bool] Set to True to show the plots. False by default.
    """


    input_data = np.array(input_data).astype(np.float64)

    # Reverse the signs of magnitudes to conform with the gamma distribution definition
    if not mass:
        input_data = -input_data
    

    # Fit the slope to the data
    params, x_arr, turnover_point, ref_point, slope, slope_report, sign, kstest = fitSlope(input_data, mass)


    # Estimate the uncertainty of the slope by sampling the fitted distribution 100 times and refitting it
    #   to the same number amples as the number of input data points
    unc_results_list = []
    for i in range(100):

        # Sample the distribution
        sampled_data = scipy.stats.gamma.rvs(*params, size=len(input_data))

        # Fit the slope again
        unc_results = fitSlope(sampled_data, mass)

        unc_results_list.append(unc_results)



    if show_plots:

        if mass:
            xlabel = 'Log of mass (kg)'

        else:
            xlabel = 'Magnitude'

        ### PLOTTING ###

        # Compute the number of bins for the histogram
        nbins = int(np.ceil(np.sqrt(len(input_data))))

        # Find the slope at the reference point for PDF plotting
        slope_pdf = scipy.stats.gamma.pdf(ref_point, *params)

        plt.hist(sign*input_data, bins=nbins, density=True, color='k', histtype='step')
        
        plt.plot(sign*x_arr, scipy.stats.gamma.pdf(x_arr, *params))



        # Plot positions of all points found during uncertainty estimation
        ref_point_unc_list = []
        for unc_results in unc_results_list:
            
            params_unc, _, turnover_point_unc, ref_point_unc, _, _, _, _ = unc_results
            ref_point_unc_list.append(ref_point_unc)

            # Find the slope at the reference point for PDF plotting
            slope_pdf_unc = scipy.stats.gamma.pdf(ref_point_unc, *params_unc)

            plt.scatter(sign*ref_point_unc, slope_pdf_unc, color='k', zorder=3, alpha=0.05)

            plt.scatter(sign*turnover_point_unc, scipy.stats.gamma.pdf(turnover_point_unc, *params_unc), \
                color='k', zorder=3, alpha=0.1)


        # Compute standard deviation of reference point
        # The sddev is not computed for turnover point as it has the same value
        ref_point_std = np.std(ref_point_unc_list)
        
        plt.scatter(sign*ref_point, slope_pdf, color='r', zorder=4, \
            label='Reference point = {:.2f} $\pm$ {:.2f}'.format(sign*ref_point, ref_point_std))

        plt.scatter(sign*turnover_point, scipy.stats.gamma.pdf(turnover_point, *params), color='r', \
            marker='x', zorder=4, label='Turnover point = {:.2f}'.format(sign*turnover_point))


        plt.ylabel('Normalized count')
        plt.legend()

        plt.xlabel(xlabel)

        plt.show()


        ####

        # Compute intercept of the line on the reverse cumulative plot (survival function)
        y_temp = np.log10(scipy.stats.gamma.sf(ref_point, *params))
        intercept = y_temp + slope*ref_point

        plt.hist(sign*input_data, bins=nbins, cumulative=-sign, density=True, log=True, histtype='step', 
            color='k', zorder=4)
        plt.plot(sign*x_arr, scipy.stats.gamma.sf(x_arr, *params))


        # Plot slopes of all lines found during uncertainty estimation
        slope_report_unc_list = []
        for unc_results in unc_results_list:
            
            _, _, _, ref_point_unc, slope_unc, slope_report_unc, _, _ = unc_results
            slope_report_unc_list.append(slope_report_unc)

            # Compute intercept of the line on the reverse cumulative plot (survival function)
            y_temp_unc = np.log10(scipy.stats.gamma.sf(ref_point_unc, *params))
            intercept_unc = y_temp_unc + slope_unc*ref_point_unc

            # Plot the tangential line with the slope
            plt.plot(sign*x_arr, logline(-x_arr, slope_unc, intercept_unc), color='k',\
                alpha=0.05, zorder=3)

        # Compute the slope standard deviation
        slope_report_std = np.std(slope_report_unc_list)


        # Plot the turnover point
        plt.scatter(sign*ref_point, 10**y_temp, c='r', \
            label='Reference point = {:.2f} $\pm$ {:.2f}'.format(sign*ref_point, ref_point_std), zorder=4)

        # Plot the tangential line with the slope
        plt.plot(sign*x_arr, logline(-x_arr, slope, intercept), color='r', \
            label='Slope = {:.2f} $\pm$ {:.2f}'.format(slope_report, slope_report_std), zorder=4)


        plt.xlabel(xlabel)
        plt.ylabel('Cumulative count')
        plt.legend()

        plt.show()




    return params, sign*ref_point, sign*turnover_point, slope_report, kstest