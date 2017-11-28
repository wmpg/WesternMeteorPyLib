
//###################################################################################
//                     Meteor Trajectory Iterative Solution
//###################################################################################
//
// This meteor trajectory module solves for the atmospheric track of a meteor given
// video measurements from multiple cameras. It uses a multi-parameter fit approach
// that assumes that for each camera, the timing information for measurements are
// precisely known. That is, the measurement angles are as extracted from the focal
// plane meteor positions and can be noisy, but the timing is provided by the user,
// whom most often assumes a constant video rate. So a given measurement's time,
// relative to others for a given camera, is well known. But if the precise time
// of each measurement is obtainable, then that can be used instead and it does
// not need to be uniformly spaced.
//
// IMPORTANT NOTE: ----------------------------------------------------------------
// This version "B" allows for the site positions to move during the frame-to-frame
// time steps accounting for Earth's rotation under the meteor. Thus one should NOT
// remove the coriolis effect later in calculating orbits as it is accounted for in
// the geocentric radiant information reported herein. ----------------------------
//
// The multi-parameter fit algorithm and implemented software can handle two or more
// cameras and also deal with multiple cameras from the same site. It uses a
// particle swarm optimzation (PSO) algorithm to fit the best propagating linear-path
// motion-model to the measurements provided. The parameters solved for are the
// radiant direction, begin point velocity, deceleration terms, timing offsets
// between the unsynchronized camera measurements, begin and end point LLA positions,
// and LLA position, range, and velocity at each measurement and motion model point.
//
// It solves the problem in several steps via a bootstrapping approach to help ensure
// the non-linear iterative solver finds the global minimum of the cost function. The
// PSO is used to find the minimum of the residual angles between the measurement rays
// and points along the 3D linear-path motion-model.
//
// For video systems, the time of the measurements are typically assumed to be
// uniformly spaced, but can be non-uniformly spaced and/or have drop-outs. Also each
// camera can be unsynchronized with respect to the others, and the solver will
// estimate the sub-frame timing offsets between them. In a post-solution step,
// a Monte Carlo loop provides an estimate of the standard deviations for the
// principle parameters by adding 2D Gaussian noise (normally distributed) to the
// measurements and re-solving the trajectory. Note the final trajectory returned is
// for the no added noise case and is NOT the mean of the Monte Carlo solutions.
//
// IMPORTANT NOTES:
//
// Results are for geocentric radiants and velocities and have not been corrected
// for zenith attraction. The trajectory estimator also does not correct for
// stellar aberration.
//
//###################################################################################
//
// Software Interface: The trajectory solver relies on a number of functions that are
//                     self contained within this file. In addition, the processing
//                     approach relies heavily on the particle swarm optimization
//                     module (via the include file ParticleSwarmFunctions.h). The
//                     following lines of pseudo-code gives a calling sequence
//                     example. For each trajectory to be solved, the user should
//                     RESET the trajectory structure, then feed measurements to the
//                     trajectory_info structure per camera via INFILL. After all the
//                     measurements have been populated into the trajectory_info
//                     structure, then the function MeteorTrajectory should be
//                     invoked to perform a simultaneous solve for all parameters
//                     via a low fidelity to high fidelity bootstrapping technique.
//                     --------------------------------------------------------------
//
//  #include  TrajectorySolution.h
//
//  #include  ParticleSwarmFunctions.h
//
//
//  struct trajectory_info  traj;
//
//
//  InitTrajectoryStructure( maxcameras_expected, &traj );    //... Called at program startup
//
//  ReadTrajectoryPSOconfig( "PSO_Trajectory_Config.txt", &traj );  //... Get PSO parameters
//
//  ...
//
//  //-------- For every solution to be computed, always RESET the trajectory structure providing:
//                  reference julian date/time, max timing offset, velocity model mode,
//                  #monte carlo trials, measurement type mode, verbose flag
//
//  ResetTrajectoryStructure( jdt_ref, max_toffset, velmodel, nummonte, meastype, verbose, &traj );
//
//  //--------- Next loop over each camera track (set of measurements) to INFILL the trajectory structure
//  //              where the loop count will set the camera count in the structure. User to provide the
//  //              number of measurements for each camera, the two anglar measurement vectors (radians),
//  //              time relative to the jdt reference time (seconds), and noise std dev estimate per
//  //              measurement (radians). Also the camera's site latitude, longitude, and height
//  //              in standard GPS coordinates (radians and kilometers).
//
//  Loop over cameras (each multi-frame measurement sequence) to infill the trajectory structure
//
//       InfillTrajectoryStructure( #measurements, ra, dec, deltatime, noise, latitude, longitude, height, &traj );
//
//  End camera loop
//
//
//  MeteorTrajectory( &traj );   //... Solve for best trajectory fit parameters
//
//
//  //--------- Solution complete, start next trajectory solution with a RESET call ...
//  ...
//  ...
//  ...
//
//
//  FreeTrajectoryStructure( &traj );  //... Called at program completion to free memory
//
//
//###################################################################################
//
// Bootstrapping Algorithm:
//
//          Step 1 uses the intersecting planes method between all pairs of camera
//                 measurements, obtaining the position and radiant for the pair
//                 with the largest convergence angle.
//          Step 2 is based upon Jiri Borivicka's LMS approach from the paper "The
//                 Comparison of Two Methods of Determining Meteor Trajectories
//                 from Photographs" where the position and radiant direction are
//                 refined from step 1 relying solely on minimizing the sum of
//                 angles (vice distances in the paper) between all camera
//                 measurements and a 3D line, ignoring any temporal information.
//          Step 3 uses a least mean square solution to the velocity by projecting
//                 all measurement rays to the radiant line and fitting the CPA
//                 points nearest the line for ALL camera measurement sets
//                 simultaneously assuming a common velocity. Obtains the timing
//                 offset estimates as a bonus.
//          Step 4 uses the PSO to refine ALL parameters except deceleration by
//                 assuming a constant velocity model.
//          Step 5 uses the PSO to make estimates of the deceleration parameters
//                 keeping all other parameters fixed.
//          Step 6 uses the PSO to iteratively solve for ALL the variable parameters
//                 simultaneously given that a good estimate is now available
//                 for each of the unknowns.
//          Step 7 slides the timing offsets to minimize the cost function then
//                 solve for ALL the variable parameters simultaneously again.
//          Step 8 loops over Monte Carlo trials using PSO to quickly solve for
//                 a solution having the initial PSO guess start around the no
//                 noise solution. Each Monte Carlo trial adds limited extent
//                 2D Gaussian noise to the measurements, and obtain a set of
//                 new solutions which feeds a statistical standard deviation
//                 for a subset of the critical parameters (Radiant Rah/Dec,
//                 Vbegin, Decel1, Decel2).
//
//###################################################################################
//
// Date         Ver   Author        Comment
// -----------  ---   ------------  -------------------------------------------------
// Feb 06 2011  1.0   Pete Gural    Initial C implementation with ameoba-simplex
// Oct 15 2016  2.0   Pete Gural    Revised with PSO, modularized, new I/O interface,
//                                  weighted LMS velocity, weighted cost functions
// Jan 11 2017  2.1   Pete Gural    Added PSO particles, tighter convergence, and
//                                  adjusted the bootstrap processing
// Jan 18 2017  2.2   Pete Gural    Added ref time parameter to fit for acceleration
//                                  models and added guess for time offsets
// Jan 19 2017  2.3   Pete Gural    Fixed propagation function
// Jan 26 2017  2.4   Pete Gural    Moved hardcoded PSO parameters to config file
// Feb 08 2017  2.5   Pete Gural    Added closed form guess for deceleration
//
//###################################################################################



//###################################################################################
//                        Includes and Constants
//###################################################################################

#define   RADEC         1   // "meastype" is RA and Dec
#define   NAZEL         2   // Azim +east of north, altitude angle
#define   SAZZA         3   // Azim +west of south, zenith angle
#define   EAZZA         4   // Azim +north of east, zenith angle

#define   POSITION      0   // "propagation_state"
#define   VELOCITY      1

#define   CONSTANT      0   // "velmodel" velocity motion model
#define   LINEAR        1
#define   QUADRATIC     2
#define   EXPONENT      3

#define   NO_NOISE      0   // noise flag for adding measurement noise
#define   ADD_NOISE     1

#define   MEAS2LINE     0   // cost function selection
#define   MEAS2MODEL    1

#define   QUICK_FIT     0   // "pso_accuracy_config"  settings
#define   GOOD_FIT      1
#define   ACCURATE_FIT  2
#define   EXTREME_FIT   3
#define   NFIT_TYPES    4   // = last "fit" mnemonic token value + 1
#define   NPSO_CALLS    6   // # unique ParameterRefinementViaPSO calls in trajectory

#define   LLA_BEG       0   // "LLA_position"  extreme begin or end point
#define   LLA_END       1





//###################################################################################
//                      Trajectory Structure Definitions
//###################################################################################

struct PSO_info
{
	//======= Particle Swarm Optimizer
	int       number_particles;           // Number of particles to be used
    int       maximum_iterations;         // Maximum number of iterations criteria
    int       boundary_flag;              // Periodic or reflective boundary behavior
    int       limits_flag;                // Strict or loose boundary limits
    int       particle_distribution_flag; // Uniformly random or Gaussian
    double    epsilon_convergence;        // Relative residual error criteria
    double    weight_inertia;             // Weight for inertia - (typically 0.8)
    double    weight_stubborness;         // Weight for stubborness (typically 1.0)
    double    weight_grouppressure;       // Weight for grouppressure (typically 2.0)
};


//.......................................................................................................


struct trajectory_info
{
    //======= PARAMETERS REFLECTING INPUT SETTINGS and CONTROL ==========================================

    //----------------------------- Max memory handling, intermediate reporting
    int       maxcameras;        // Maximum number of cameras expected (to initially allocate arrays)
    int       verbose;           // Flag to turn on intermediate product reporting during function call
                                 //     0 = no writes to the console
                                 //     1 = all step's intermediate values displayed on console
                                 //     2 = only final step solution parameters displayed on console
                                 //     3 = TBD measurements and model positions sent to console

    //----------------------------- Modeling parameters
    int       nummonte;          // Number of Monte Carlo trials for standard deviation calculation
    int       velmodel;          // Velocity propagation model
                                 //     0 = constant   v(t) = vinf
                                 //     1 = linear     v(t) = vinf - |acc1| * t
                                 //     2 = quadratic  v(t) = vinf - |acc1| * t + acc2 * t^2
                                 //     3 = exponent   v(t) = vinf - |acc1| * |acc2| * exp( |acc2| * t )
    int       meastype;          // Flag to indicate the type of angle measurements the user is providing
                                 //    for meas1 and meas2 below. The following are all in radians:
                                 //        1 = Right Ascension for meas1, Declination for meas2.
                                 //        2 = Azimuth +east of due north for meas1, Elevation angle
                                 //            above the horizon for meas2
                                 //        3 = Azimuth +west of due south for meas1, Zenith angle for meas2
                                 //        4 = Azimuth +north of due east for meas1, Zenith angle for meas2

    //----------------------------- Reference timing and offset constraint
    double    jdt_ref;           // Reference Julian date/time that the measurements times are provided
                                 //     relative to. This is user selectable and can be the time of the
                                 //     first camera, or the first measurement, or some average time for
                                 //     the meteor, but should be close to the time of passage of
                                 //     the meteor. This same reference date/time will be used on all
                                 //     camera measurements for the purposes of computing local sidereal
                                 //     time and making  geocentric coordinate transformations.
    double    max_toffset;       // Maximum allowed time offset between cameras in seconds

	//----------------------------- Particle Swarm Optimizer (PSO) settings

	struct PSO_info    PSOfit[NFIT_TYPES];  // Structure containing the various PSO settings per fit

	int       PSO_fit_control[NPSO_CALLS];  // Fit option for each ParameterRefinementViaPSO call



    //======= SOLUTION STARTUP and DERIVED PRODUCTS ==========================================================

    //----------------------------- Memory allocation handling
    int      *malloced;          // Vectors and arrays memory allocation flag per camera (0=freed,1=allocated)

    //----------------------------- Camera/site information
    int       numcameras;        // Number of cameras having measurements passed in and is determined by
                                 //    the number of calls to InfillTrajectoryStructure. Measurements are
                                 //    typically from multiple sites but can include multiple cameras from
                                 //    the same site. There must be one pair of well separated site cameras
                                 //    for triangulation to proceed successfully.
    double   *camera_lat;        // Vector of GEODETIC latitudes of the cameras in radians (GPS latitude)
    double   *camera_lon;        // Vector of longitudes (positive east) of the cameras in radians
    double   *camera_hkm;        // Vector of camera heights in km above a WGS84 ellipsoid (GPS height)
    double   *camera_LST;        // Vector of local sidereal times for each camera (based on jdt_ref)
    double ***rcamera_ECI;       // Camera site ECI radius vector from Earth center (#cameras x #measurements x XYZ)

    //----------------------------- Measurement information
    int      *nummeas;           // Vector containing the number of measurements per camera

                                 // ------The following are dimensioned #cameras x #measurements(for that camera)
    double  **meas1;             // Array of 1st measurement type (see meastype), typically RA, typically in radians
    double  **meas2;             // Array of 2nd measurement type (see meastype), typically Dec, typically in radians
    double  **dtime;             // Array of time measurements in seconds relative to jdt_ref
    double  **noise;             // Array of measurement standard deviations in radians (per measurement)

    double ***meashat_ECI;       // Measurement ray unit vectors (#cameras x #measurements x XYZT)
	                             //    XYZ in is Earth Centered Inertial (ECI) coordinates
	                             //    T is dtime minus the reference time dtime_ref
    double    ttbeg;             // Begin position time relative to ref time dtime_ref
    double    ttend;             // End position time relative to ref time dtime_ref
	double    ttzero;            // Tzero model start time relative to ref time dtime_ref


    //----------------------------- Trajectory fitting parameters to feed to the particle swarm optimizer
    int       numparams;         // Number of total parameters for a full fit
    double   *params;            // Working vector of parameter values of length numparams
    double   *limits;            // Each parameter's value search limit of length numparams
    double   *xguess;            // Initial starting guess for the parameters of length numparams
    double   *xshift;            // Plus/minus limit around the guess values of length numparams (if set
                                 //    to zero, this parameter will remain fixed during the PSO)


    //======= SOLUTION OUTPUT PRODUCTS ====================================================================

    //------------------ Note: The output RA/Dec equatorial coordinates epoch is the same as that
    //                         of the input measurements and represent Geocentric values, but they
    //                         are NOT corrected for zenith attraction.
    //                         The velocity at the begin point "vbegin" could be considered equivalent
    //                         to the velocity at the top of the atmosphere Vinfinity, and is also NOT
    //                         corrected for zenith attraction.

    //----------------------------- Best solution vector of the parameter values of length numparams
    double   *solution;          // { Rx, Ry, Rz, Vx, Vy, Vz, Decel1, Decel2, tzero, tref_offsets[*] }
                                 // Note that R and V are in ECI (ECEF)

    //----------------------------- Primary output products and their standard deviations (sigma)

    double    ra_radiant;        // Radiant right ascension in radians (multi-parameter fit)
    double    dec_radiant;       // Radiant declination in radians (multi-parameter fit)
    double    vbegin;            // Meteor solution velocity at the begin point in km/sec
    double    decel1;            // Deceleration term 1 defined by the given velocity model
    double    decel2;            // Deceleration term 2 defined by the given velocity model

    double    ra_sigma;          // Standard deviation of radiant right ascension in radians
    double    dec_sigma;         // Standard deviation of radiant declination in radians
    double    vbegin_sigma;      // Standard deviation of vbegin in km/sec
    double    decel1_sigma;      // Standard deviation of decceleration term 1
    double    decel2_sigma;      // Standard deviation of decceleration term 2

    //----------------------------- Intermediate bootstrapping solutions

                                 // Intersecting planes solution for best convergence angle pair
    double    max_convergence;   // Max convergence angle between all camera pairs in radians
    double    ra_radiant_IP;     // Radiant right ascension in radians
    double    dec_radiant_IP;    // Radiant declination in radians

                                 // Intersecting planes solution for weighted multiple tracks
    double    ra_radiant_IPW;    // Radiant right ascension in radians
    double    dec_radiant_IPW;   // Radiant declination in radians

                                 // Borovicka's least mean squares solution for the radiant
    double    ra_radiant_LMS;    // Radiant right ascension in radians
    double    dec_radiant_LMS;   // Radiant declination in radians

    //----------------------------- Timing output relative to jdt_ref in seconds
	double    dtime_ref;         // The input dtime associated with the ref starting position Rx, Ry, Rz (!= beg)
	double    dtime_tzero;       // The input dtime that represents t=0 in the velocity fit model
	double    dtime_beg;         // The input dtime that represents the begin position reported
	double    dtime_end;         // The input dtime that represents the end   position reported
    double   *tref_offsets;      // Vector of timing offsets in seconds per camera

    //----------------------------- Measurement and model LLA, range and velocity arrays
    //                                  with dimension #cameras x #measurements(camera)
    double  **meas_lat;          // Array of geodetic latitudes closest to trail for each measurement
    double  **meas_lon;          // Array of +east longitudes closest to trail for each measurement
    double  **meas_hkm;          // Array of heights re WGS84 closest to trail for each measurement
    double  **meas_range;        // Array of ranges from site along measurement to the CPA of the trail
    double  **meas_vel;          // Array of velocity along the trail for each measurement

    double  **model_lat;         // Array of geodetic latitudes for the model positions
    double  **model_lon;         // Array of +east longitudes for the model positions
    double  **model_hkm;         // Array of heights re WGS84 for the model positions
    double  **model_range;       // Array of ranges from site to the model positions
    double  **model_vel;         // Array of velocity on the trail at each model position

    //----------------------------- Model fit vectors which follow the same "meastype" on output
    //                                  with dimension #cameras x #measurements(camera)
    double  **model_fit1;        // Array of 1st data sequence containing the model fit in meastype format
    double  **model_fit2;        // Array of 2nd data sequence containing the model fit in meastype format
    double  **model_time;        // Array of model time which includes offsets relative to the reference time

    //----------------------------- BEGIN position and standard deviation in LLA
    double    rbeg_lat;          // Position on radiant line as GEODETIC latitude in radians
    double    rbeg_lon;          // Position on radiant line as +EAST longitude in radians
    double    rbeg_hkm;          // Position on radiant line as height in km relative WGS84

    double    rbeg_lat_sigma;    // Standard deviation of rbeg_lat in radians
    double    rbeg_lon_sigma;    // Standard deviation of rbeg_lon in radians
    double    rbeg_hkm_sigma;    // Standard deviation of rbeg_hkm in km

    //----------------------------- END position and standard deviation in LLA
    double    rend_lat;          // Position on radiant line as GEODETIC latitude in radians
    double    rend_lon;          // Position on radiant line as +EAST longitude in radians
    double    rend_hkm;          // Position on radiant line as height in km relative WGS84

    double    rend_lat_sigma;    // Standard deviation of rend_lat in radians
    double    rend_lon_sigma;    // Standard deviation of rend_lon in radians
    double    rend_hkm_sigma;    // Standard deviation of rend_hkm in km


};  //... end trajectory_info structure definition


//###################################################################################
//                          Prototype Definitions
//###################################################################################

int     MeteorTrajectory( struct trajectory_info *traj );

void    InitTrajectoryStructure( int maxcameras, struct trajectory_info *traj );

void    FreeTrajectoryStructure( struct trajectory_info *traj );

void    ResetTrajectoryStructure( double jdt_ref, double max_toffset,
                                  int velmodel, int nummonte, int meastype, int verbose,
                                  struct trajectory_info *traj );

void    InfillTrajectoryStructure( int nummeas, double *meas1, double *meas2, double *dtime, double *noise,
                                   double site_latitude, double site_longitude, double site_height,
                                   struct trajectory_info *traj );

int     ReadTrajectoryPSOconfig( char* PSOconfig_pathname, struct trajectory_info *traj );



