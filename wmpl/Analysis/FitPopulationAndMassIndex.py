""" Method for fitting the population and mass index based on fitting the gamma distribution on observed
    data by MLE method and extracting the slope in the magnitude completeness region. 
"""

from __future__ import print_function, division, absolute_import

import os

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
    inflection_point = x_arr[pdf_derivative.argmin()]

    if not mass:
        # Take one magnitude brighter than the inflection point (the addition here is because the inflection
        #   point is reverse)
        ref_point = inflection_point + 1

    else:
        ref_point = inflection_point + 0.4


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


    return params, x_arr, inflection_point, ref_point, slope, slope_report, sign, kstest




def estimateIndex(input_data, mass=False, show_plots=False, plot_save_path=None, nsamples=100, \
    mass_as_intensity=False):
    """ Estimate the mass or population index from the meteor data by fitting a gamma function to it using
        MLE and estimating the slope in the completeness region.

    Arguments:
        input_data: [array like] List of peak magnitudes or logarithms of mass

    Keyword arguments:
        mass: [bool] If true, the mass index will be computed. False by default, in which case the population
            index is computed.
        show_plots: [bool] Set to True to show the plots. False by default.
        plot_save_path: [str] Path where to save the plots. None by default, in which case the plots will not
            be saved.
        nsamples: [int] Number of samples for uncertainty estimation. 100 by default.
        mass_as_intensity: [bool] If True, it indicates that the integrated intensity in zero-magnitude units
            is given instead of the mass. This will just change the axis labels. False by default.
    """


    input_data = np.array(input_data).astype(np.float64)

    
    # Skip NaN values
    input_data = input_data[~np.isnan(input_data)]


    # Reverse the signs of magnitudes to conform with the gamma distribution definition
    if not mass:
        input_data = -input_data
    

    # Fit the slope to the data
    params, x_arr, inflection_point, ref_point, slope, slope_report, sign, kstest = fitSlope(input_data, mass)


    # Estimate the uncertainty of the slope using the bootstrap method
    unc_results_list = []
    for i in range(nsamples):

        # Sample the distribution with replacement
        sampled_data = np.random.choice(input_data, len(input_data))

        # Fit the slope again
        unc_results = fitSlope(sampled_data, mass)

        unc_results_list.append(unc_results)

    
    # Reject all 3 sigma slope outliers
    slopes = [scipy.stats.gamma.pdf(entry[3], *entry[0]) for entry in unc_results_list]
    median_slope = np.median(slopes)
    slope_std = np.std(slopes)

    unc_results_list_filtered = []
    for i, unc_slope in enumerate(slopes):

        # Reject all 3 sigma outliers
        if abs(unc_slope - median_slope) > 3*slope_std:
            continue

        unc_results_list_filtered.append(unc_results_list[i])

    unc_results_list = unc_results_list_filtered


    # Compute the standard deviation of the slope
    slope_report_unc_list = []
    for unc_results in unc_results_list:
        _, _, _, ref_point_unc, slope_unc, slope_report_unc, _, _ = unc_results
        slope_report_unc_list.append(slope_report_unc)

    # Compute the slope standard deviation
    slope_report_std = np.std(slope_report_unc_list)

    # If the standard deviation is larger than 10 or is nan, set it to 10
    if (slope_report_std > 10) or np.isnan(slope_report_std):
        slope_report_std = 10

    # Reject all 3 sigma outliers and recompute the std
    slope_report_unc_list = [slp for slp in slope_report_unc_list if (slp < slope_report \
        + 3*slope_report_std) and (slp > slope_report - 3*slope_report_std)]

    # Recompute the standard deviation
    slope_report_std = np.std(slope_report_unc_list)


    # Make plots
    if show_plots or (plot_save_path is not None):

        if mass:

            if mass_as_intensity:
                xlabel = 'Integ. intensity (zero mag. units)'
            else:
                xlabel = 'Log of mass (kg)'

            plot_save_name = 'mass'
            slope_name = 's'

        else:
            xlabel = 'Magnitude'
            plot_save_name = 'magnitude'
            slope_name = 'r'


        # Compute the transparency correction
        alpha_factor = 1.0
        if nsamples > 100:
            alpha_factor = np.sqrt(100/nsamples)

        ### PLOTTING ###

        # Compute the number of bins for the histogram
        nbins = int(np.ceil(np.sqrt(len(input_data))))

        # Find the slope at the reference point for PDF plotting
        slope_pdf = scipy.stats.gamma.pdf(ref_point, *params)

        # Plot the histogram
        plt.hist(sign*input_data, bins=nbins, density=True, color='k', histtype='step')
        
        # Plot the estimated slope
        plt.plot(sign*x_arr, scipy.stats.gamma.pdf(x_arr, *params), zorder=4)

        # Get Y axis range
        y_min, y_max = plt.gca().get_ylim()


        # Plot positions of all points found during uncertainty estimation
        ref_point_unc_list = []
        for unc_results in unc_results_list:
            
            params_unc, _, inflection_point_unc, ref_point_unc, _, _, _, _ = unc_results
            ref_point_unc_list.append(ref_point_unc)

            # Find the slope at the reference point for PDF plotting
            slope_pdf_unc = scipy.stats.gamma.pdf(ref_point_unc, *params_unc)

            # Alpha (transparency) is normazlied for 100 sampling runs
            plt.plot(sign*x_arr, scipy.stats.gamma.pdf(x_arr, *params_unc), color='k', \
                alpha=0.05*alpha_factor)

            plt.scatter(sign*ref_point_unc, slope_pdf_unc, color='g', zorder=3, alpha=0.1*alpha_factor)

            plt.scatter(sign*inflection_point_unc, scipy.stats.gamma.pdf(inflection_point_unc, *params_unc), \
                color='y', zorder=3, alpha=0.1*alpha_factor)


        # Compute standard deviation of reference point
        # The sddev is not computed for inflection point as it has the same value
        ref_point_std = np.std(ref_point_unc_list)
        
        plt.scatter(sign*ref_point, slope_pdf, color='r', zorder=5, \
            label='Reference point = {:.2f} $\pm$ {:.2f}'.format(sign*ref_point, ref_point_std))

        plt.scatter(sign*inflection_point, scipy.stats.gamma.pdf(inflection_point, *params), color='r', \
            marker='x', zorder=5, label='Inflection point = {:.2f}'.format(sign*inflection_point))


        plt.ylabel('Normalized count')
        plt.legend()

        plt.xlabel(xlabel)

        plt.ylim([y_min, y_max])

        # Save the plot
        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path, plot_save_name + "_distribution.png"), dpi=300)

        # Show the plot
        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()


        ####

        # Compute intercept of the line on the reverse cumulative plot (survival function)
        y_temp = np.log10(scipy.stats.gamma.sf(ref_point, *params))
        intercept = y_temp + slope*ref_point

        # Plot data histogram
        plt.hist(sign*input_data, bins=len(input_data), cumulative=-sign, density=True, log=True, histtype='step', 
            color='k', zorder=4)

        # Get Y axis range
        y_min, y_max = plt.gca().get_ylim()
        
        # # Plot fitted survival function
        # plt.plot(sign*x_arr, scipy.stats.gamma.sf(x_arr, *params))


        # Plot slopes of all lines found during uncertainty estimation
        for unc_results in unc_results_list:
            
            params_unc, _, _, ref_point_unc, slope_unc, slope_report_unc, _, _ = unc_results

            # Compute intercept of the line on the reverse cumulative plot (survival function)
            y_temp_unc = np.log10(scipy.stats.gamma.sf(ref_point_unc, *params_unc))
            intercept_unc = y_temp_unc + slope_unc*ref_point_unc

            # Plot the tangential line with the slope
            plt.plot(sign*x_arr, logline(-x_arr, slope_unc, intercept_unc), color='k',\
                alpha=0.05*alpha_factor, zorder=3)


        # Plot the inflection point
        plt.scatter(sign*ref_point, 10**y_temp, c='r', \
            label='Reference point = {:.2f} $\pm$ {:.2f}'.format(sign*ref_point, ref_point_std), zorder=5)

        # Plot the tangential line with the slope
        plt.plot(sign*x_arr, logline(-x_arr, slope, intercept), color='r', \
            label='{:s} = {:.2f} $\pm$ {:.2f} \nKS test D = {:.3f} \nKS test p-value = {:.3f}'.format(\
                slope_name, slope_report, slope_report_std, kstest.statistic, kstest.pvalue), zorder=5)


        # # Limit Y axis range to the maximum of the fitted line
        # y_max = np.max(logline(-x_arr, slope, intercept))
        # plt.ylim(ymax=y_max)

        plt.ylim([y_min, y_max])

        plt.xlim([np.min(sign*x_arr), np.max(sign*x_arr)])
        


        plt.xlabel(xlabel)
        plt.ylabel('Cumulative count')
        plt.legend()

        # Save the plot
        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path, plot_save_name + "_cumulative.png"), dpi=300)

        # Show the plot
        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()




    return params, sign*ref_point, sign*inflection_point, slope_report, slope_report_std, kstest