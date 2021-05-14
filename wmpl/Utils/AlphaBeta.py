""" Defines functions for the alpha-beta fireball characterization by Gritsevich 2012. 
Adapted from: https://github.com/desertfireballnetwork/alpha_beta_modules
"""


import numpy as np
import scipy.optimize




# Height normalization constant
HT_NORM_CONST = 7160.0


def minimizeAlphaBeta(v_normed, ht_normed):
    """ initiates and calls the Q4 minimisation given in Gritsevich 2007 -
        'Validity of the photometric formula for estimating the mass of a fireball projectile'
    """

    def _alphaBetaMinimization(x, v_normed, ht_normed):
        """minimises equation 7 using Q4 minimisation given in equation 10 of 
           Gritsevich 2007 - 'Validity of the photometric formula for estimating 
           the mass of a fireball projectile'

        """ 

        alpha, beta = x

        res = 0.0

        # Compute the sum of squared residuals of fit
        res = np.sum(pow(2*alpha*np.exp(-ht_normed) \
                         - (scipy.special.expi(beta) \
                            - scipy.special.expi(beta*v_normed**2))*np.exp(-beta), 2))

        return res


    # params = np.vstack((v_normed, ht_normed))

    # Compute initial alpha-beta guess
    b0 = 1.0
    a0 = np.exp(ht_normed[-1])/(2.0*b0)
    x0 = [a0, b0]

    # Set alpha-beta limits
    xmin = [    0.001,  0.00001]
    xmax = [10000.0,   50.0]
    bnds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]))

    # Compute best-fit alpha-beta values
    res = scipy.optimize.minimize(_alphaBetaMinimization, x0, args=(v_normed, ht_normed), bounds=bnds)

    return res.x



def fitAlphaBeta(v_data, ht_data, v_init=None):
    """ Fit the alpha and beta parameters to the given velocity and height data. 
    
    Arguments:
        v_data: [ndarray] Velocity data (m/s).
        ht_data: [ndarray] Height data (m).

    Keyword arguments:
        v_init: [float] Initial velocity (m/s). If None, it will be determined from the first 20% of point
            (or a minimum of 10 points).

    Return:
        (v_init, alpha, beta):
            - v_init: [float] Input or derived initial velocity (m/s).
            - alpha: [float] Balistic coefficient.
            - beta: [float] Mass loss.
    """


    # Compute the initial velocity, if it wasn't given already
    if v_init is None:

        max_index = int(0.2*len(v_data))
        if max_index < 10:
            max_index = 10

        v_init = np.median(v_data[:max_index])



    # Normalize the velocity
    v_normed = v_data/v_init

    # Normalize the height
    ht_normed = ht_data/HT_NORM_CONST


    # Fit alpha and beta
    alpha, beta = minimizeAlphaBeta(v_normed, ht_normed)


    return v_init, alpha, beta



def alphaBetaHeight(vel_data, alpha, beta, v_init):
    """ Compute the height given the velocity and alpha, beta parameters.

    Arguments:
        vel_data: [ndarray] Velocity data (m/s).
        alpha: [float] Balistic coefficient.
        beta: [float] Mass loss.
        v_init: [float] Input or derived initial velocity (m/s).

    Return:
        ht_data: [ndarray] Height data (m).
    
    """

    # Normalize the velocity
    vel_normed = vel_data/v_init

    # Compute the normalized height
    ht_normed = np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta*vel_normed**2))/2)

    # Compute the height in m
    ht_data = ht_normed*HT_NORM_CONST

    return ht_data



def alphaBetaVelocity(ht_data, alpha, beta, v_init):
    """ Compute the velocity given the height and alpha, beta parameters. Unfortunately there is no 
        analytical inverse to the exponential integral, so the solution is found numerically.

    Arguments:
        ht_data: [ndarray] Height data (m).
        alpha: [float] Balistic coefficient.
        beta: [float] Mass loss.
        v_init: [float] Input or derived initial velocity (m/s).

    Return:
        vel_data: [ndarray] Velocity data (m/s).
    
    """

    def _diff(v, alpha, beta, ht_target):
        """ Function to minimize the height. """
        
        # Compute the height using a guess velocity
        ht_guess = np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta*v**2))/2)
        
        return (ht_guess - ht_target)**2

    # Normalize the height
    ht_normed = ht_data/HT_NORM_CONST

    # Numerically compute the normalized velocity
    vel_normed = []
    
    v_first_guess = 0.5
    bounds = [(0.0000001, 0.9999999)]
    for ht_n in ht_normed:

        # Minimize the forward function to find the velocity at the given height
        res = scipy.optimize.minimize(_diff, v_first_guess, args=(alpha, beta, ht_n), bounds=bounds)
        vel_normed.append(res.x)

    vel_normed = np.array(vel_normed)

    # Compute the velocity in m/s
    vel_data = vel_normed*v_init

    return vel_data



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Height (m), velocity (m/s)
    input_data = np.array([
        [83681.0215156, 17640.9079043],
        [83115.2043356, 21738.210588],
        [82667.2323107, 17229.3030977],
        [82205.16874, 17788.5556192],
        [81720.7954369, 18666.3386775],
        [81311.5861921, 15784.8581581],
        [80795.4457445, 19929.4459506],
        [80346.1029557, 17368.2836897],
        [79929.9709969, 16099.6372199],
        [79440.9610398, 18937.7398444],
        [78989.9618378, 17483.5240729],
        [78520.2057116, 18228.8946485],
        [78028.4348476, 19103.1785038],
        [77593.5820248, 16909.2582178],
        [77160.1939563, 16868.3210314],
        [76703.7753714, 17782.0583169],
        [76228.0916978, 18551.6220006],
        [75780.0116806, 17492.87328],
        [75311.9978669, 18289.5704834],
        [74865.7275078, 17457.5077072],
        [74397.1073002, 18350.4138997],
        [73935.5087991, 18094.1542407],
        [73494.4889556, 17304.8853494],
        [73054.6105159, 17277.0823985],
        [72600.2502069, 17863.7516712],
        [72140.2465104, 18104.1833689],
        [71696.7922973, 17470.5832886],
        [71241.2214444, 17966.1236285],
        [70790.6254124, 17788.1234402],
        [70325.6546885, 18374.6185794],
        [69907.8932431, 16525.5475434],
        [69463.9945446, 17576.6748037],
        [69018.9109104, 17641.449356],
        [68586.5036082, 17156.1850336],
        [68105.0959818, 19120.3036409],
        [67693.6798044, 16357.1580997],
        [67244.9463149, 17858.47373],
        [66802.5003205, 17626.3326231],
        [66339.897621, 18448.6111149],
        [65929.2353315, 16393.7833404],
        [65488.9457817, 17593.8687598],
        [65047.6052004, 17653.948825],
        [64633.6975273, 16573.1150336],
        [64207.3219617, 17089.0929273],
        [63771.9106074, 17468.8511397],
        [63319.0503533, 18187.8418027],
        [62888.5377557, 17308.272412],
        [62457.658419, 17340.6031692],
        [62044.0753169, 16661.1234526],
        [61628.6904629, 16750.1269718],
        [61195.5424224, 17483.9985421],
        [60773.3635024, 17058.5606553],
        [60332.8187156, 17818.9521018],
        [59928.3247354, 16377.2972957],
        [59529.6086453, 16158.8789266],
        [59120.1592054, 16609.9572187],
        [58699.1780828, 17094.8049249],
        [58296.9717338, 16348.6122603],
        [57907.1198554, 15861.5876386],
        [57495.2725451, 16772.7496274],
        [57101.1348098, 16067.1997136],
        [56703.7176405, 16216.4670862],
        [56317.3483881, 15780.7046376],
        [55918.7425435, 16296.0825456],
        [55509.251914, 16757.6221384],
        [55156.7129653, 14440.4520845],
        [54764.5437031, 16078.4475594],
        [54392.840789, 15253.6728233],
        [54034.4700671, 14719.7902665],
        [53633.6075146, 16480.5484838],
        [53309.5933381, 13333.0525322],
        [52939.3494293, 15248.5028132],
        [52616.7653049, 13297.0678182],
        [52248.4094022, 15196.8595596],
        [51917.2901026, 13672.5442947],
        [51572.4695619, 14250.318074],
        [51232.9315025, 14044.0294958],
        [50951.8414222, 11635.5414157],
        [50605.9098419, 14330.8905801],
        [50359.1027647, 10232.0885526],
        [50016.7389018, 14204.2047434],
        [49720.5878298, 12296.8024016],
        [49432.4897207, 11971.2633929],
        [49141.4612413, 12101.897341],
        [48893.4022872, 10322.1392339],
        [48625.757255, 11144.4432081],
        [48419.7730391, 8582.12152838],
        [48139.5273914, 11683.3788851]])


    ht_data, vel_data = input_data.T


    # Estimate the alpha, beta parameters
    v_init, alpha, beta = fitAlphaBeta(vel_data, ht_data)

    print("Alpha:", alpha)
    print("Beta:", beta)

    # Predict velocity from height
    ht_arr = np.linspace(30000, 100000, 100)
    vel_arr = alphaBetaVelocity(ht_arr, alpha, beta, v_init)


    # Plot the data
    plt.scatter(vel_data, ht_data, s=5)

    # Plot the fit
    plt.plot(vel_arr, ht_arr)

    plt.show()