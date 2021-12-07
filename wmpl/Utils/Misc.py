from __future__ import print_function, division, absolute_import

import numpy as np


def valueFormat(value_format, value, unc_format, unc, std_name, multi=1.0, deg=False):
    """ Returns the formatted value with an uncertanty an confidence interval, if the uncertanty is given.

    Arguments:
        value_format: [str] String format for the value.
        value: [float] Value to put in the format.
        unc_format: [str] String format for the uncertainty.
        unc: [MCUncertainties]
        std_name: [str] Name of the uncertanty attribute, e.g. if it is 'x', then the uncertanty is 
            stored in uncertainties.x.

    Keyword arguments:
        multi: [float] Uncertanty multiplier. 1.0 by default. This is used to scale the uncertanty to
            different units (e.g. from m/s to km/s).
        deg: [bool] Converet radians to degrees if True. False by default.
        """

    if deg:
        multi *= np.degrees(1.0)


    # Format the value
    ret_str = value_format.format(value*multi)

    # Add computed uncertainties
    if unc is not None:

        # Construct symmetrical 1 sigma uncertainty
        ret_str += " +/- " + unc_format.format(getattr(unc, std_name)*multi)

        # Add confidence interval if available
        if hasattr(unc, std_name + "_ci"):

            # Get the confidence interval
            ci_l, ci_u = np.array(getattr(unc, std_name + "_ci"))*multi

            # Format confidence interval
            ret_str += ", {:d}% CI [{:s}, {:s}]".format(int(unc.ci), \
                value_format.format(ci_l), value_format.format(ci_u))


    return ret_str