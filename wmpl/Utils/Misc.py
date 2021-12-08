from __future__ import print_function, division, absolute_import

import numpy as np


def valueFormat(value_format, value, unc_format, unc, std_name, multi=None, deg=False, callable_val=None, \
    callable_ci=None):
    """ Returns the formatted value with an uncertanty an confidence interval, if the uncertanty is given.

    Arguments:
        value_format: [str] String format for the value.
        value: [float] Value to put in the format.
        unc_format: [str] String format for the uncertainty.
        unc: [MCUncertainties]
        std_name: [str] Name of the uncertanty attribute, e.g. if it is 'x', then the uncertanty is 
            stored in uncertainties.x.

    Keyword arguments:
        multi: [float] Uncertanty multiplier. None by default. This is used to scale the uncertanty to
            different units (e.g. from m/s to km/s). The multiplier is applied after the callable function.
        deg: [bool] Converet radians to degrees if True. False by default.
        callable_val: [function] Call this function on the provided value. None by default.
        callable_ci: [function] Call this function on the provided confidence interval. None by default.
        """


    # Convert value from radiants to degrees
    if deg:
        if multi is None:
            multi = 1.0

        multi *= np.degrees(1.0)


    # Apply callable_val function
    if callable_val is not None:
        value = callable_val(value)


    # Apply the value multiplier, if given
    if multi is not None:
        value *= multi


    # Format the value
    ret_str = value_format.format(value)


    # Add computed uncertainties
    if unc is not None:

        # Fetch the 1 sigma uncertainty
        sig_unc = getattr(unc, std_name)

        if multi is not None:
            sig_unc *= multi

        # Construct symmetrical 1 sigma uncertainty
        ret_str += " +/- " + unc_format.format(sig_unc)

        # Add confidence interval if available
        if hasattr(unc, std_name + "_ci"):

            # Get the confidence interval
            ci_l, ci_u = np.array(getattr(unc, std_name + "_ci"))

            if callable_ci is not None:
                ci_l = callable_ci(ci_l)
                ci_u = callable_ci(ci_u)

            if multi is not None:
                ci_l *= multi
                ci_u *= multi

            # Format confidence interval
            ret_str += ", {:d}% CI [{:s}, {:s}]".format(int(unc.ci), \
                value_format.format(ci_l), value_format.format(ci_u))


    return ret_str