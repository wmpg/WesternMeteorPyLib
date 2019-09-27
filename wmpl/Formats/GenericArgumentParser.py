""" Function that adds common arguments for all input formats to the argument parser handle. """


def addSolverOptions(arg_parser, skip_velpart=False):
    """ Adds common arguments for all input formats to the argument parser handle. """

    arg_parser.add_argument('-s', '--solver', metavar='SOLVER', help="""Trajectory solver to use. \n
        - 'original' - Monte Carlo solver
        - 'gural0' - Gural constant velocity
        - 'gural1' - Gural linear deceleration
        - 'gural2' - Gural quadratic deceleration
        - 'gural3' - Gural exponential deceleration
         """, type=str, nargs='?', default='original')

    arg_parser.add_argument('-t', '--maxtoffset', metavar='MAX_TOFFSET', nargs=1, \
        help='Maximum time offset between the stations.', type=float)

    arg_parser.add_argument('-v', '--vinitht', metavar='V_INIT_HT', nargs=1, \
        help='The initial veloicty will be estimated as the average velocity above this height (in km). If not given, the initial velocity will be estimated using the sliding fit which can be controlled with the --velpart option.', \
        type=float)

    if not skip_velpart:
        arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
            help='Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.25 (25 percent), but for noisier data this might be bumped up to 0.5.', \
            type=float, default=0.25)

    arg_parser.add_argument('-d', '--disablemc', \
        help='Do not use the Monte Carlo solver, but only run the geometric solution.', action="store_true")
    
    arg_parser.add_argument('-r', '--mcruns', metavar="MC_RUNS", nargs='?', \
        help='Number of Monte Carlo runs.', type=int, default=100)

    arg_parser.add_argument('-u', '--uncertgeom', \
        help='Compute purely geometric uncertainties.', action="store_true")
    
    arg_parser.add_argument('-g', '--disablegravity', \
        help='Disable gravity compensation.', action="store_true")

    arg_parser.add_argument('-l', '--plotallspatial', \
        help='Plot a collection of plots showing the residuals vs. time, lenght and height.', \
        action="store_true")

    arg_parser.add_argument('-i', '--imgformat', metavar='IMG_FORMAT', nargs=1, \
        help="Plot image format. 'png' by default, can be 'pdf', 'eps',... ", type=str, default='png')

    arg_parser.add_argument('-x', '--hideplots', \
        help="Don't show generated plots on the screen, just save them to disk.", action="store_true")


    return arg_parser