//###################################################################################
//                     Meteor Trajectory Iterative Solution
//###################################################################################
//
// See TrajectorySolutionB.h for full documentation & usage info
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
// Mar 21 2017  2.6   Pete Gural    Adjusted site position at each frame time step
// Dec 11 2017  2.7   Pete Gural    Put in 0.05" minimum noise floor per Denis Vida
// Dec 22 2017  2.8   Pete Gural    Set negative deceleration coefs to 0 in the first
//                                      guess estimate from function VelDecelFit_LMS
//                                  Compute weights from std dev & normalized to unity
//                                  Const V guess based on <= 10 meas per camera
//                                  Added gravitational acceleration to propagation
//                                  Added meastype of RA calculated for LST(time)
//
//###################################################################################

#pragma warning(disable: 4996)  // disable warning on strcpy, fopen, ...

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     // strcpy, strcat

#include "../common/ParticleSwarmFunctions.h"
#include "../common/System_FileFunctions.h"
#include "TrajectorySolution.h"


#ifdef _WIN32 /************* WINDOWS ******************************/

#else  /********************  LINUX  ******************************/

  #include <unistd.h>

#endif /***********************************************************/

// Moved from TrajectorySolutionB.h - shouldn't be needed for the public interface
// and this causes problems when included in other source files that use X, Y, Z, etc.
// as symbol names...
#define   X   0
#define   Y   1
#define   Z   2
#define   T   3

//############################################################################################
//                                   Prototypes
//############################################################################################

void    AllocateTrajectoryMemory4Infill( int kcamera, int nummeas, struct trajectory_info *traj );

void    InfillTrajectoryStructure( int nummeas, double *meas1, double *meas2, double *dtime, double *noise,
                                   double site_latitude, double site_longitude, double site_height,
                                   struct trajectory_info *traj );
									   
void    IntermediateConsoleOutput( char *ctitle, double *params, int ncams );

void    EnsureNonZeroStdDev( struct trajectory_info *traj );

void    MeasurementWeightsFromStdDev( struct trajectory_info *traj );

void    Angles2SiteMeasurements( double   camera_lat,  double   camera_lon,  double  camera_hkm,  double camera_LST,
                                 int      nummeas,     int      meastype,
                                 double  *meas1,       double  *meas2,       double *dtime,
                                 double  *noise,       int      noiseflag,   double  dtime_ref,  double tshift,
                                 double **meashat_ECI, double **rcamera_ECI  );

void    IntersectingPlanes( int     nmeas_camera1, double **meashat_camera1, double *rcamera1,
                            int     nmeas_camera2, double **meashat_camera2, double *rcamera2,
                            double *radiant_hat, double  *convergence_angle, double *rcpa_camera1, double *rcpa_camera2  );

void    IntersectingPlanes_MultiTrack( struct trajectory_info *traj, double *radiant_hat );

void    IntersectingPlanes_BestConvergence( struct trajectory_info *traj,
                                            int    *kcamera1, 
                                            int    *kcamera2, 
											double *max_convergence,
                                            double *radiant_bestconv, 
											double *r_bestconv  );

void    Normal2MeteorMeasurements( int nmeas, double **meashat_ECI, double *planenormal );

void    TwoLineCPA( double *position1, double *vector1, double *r1,
                    double *position2, double *vector2, double *r2, 
					double *d21 );

double  Propagation( int propmodel, int velmodel, double tt, double ttzero, double vinf, double acc1, double acc2 );

void    InitializeTzeroParameters( struct trajectory_info *traj, double *ttzero, double *ttzero_limit );

double  VelocityFit_Differencing( struct trajectory_info *traj );

void    VelocityFit_LMS( struct trajectory_info *traj, double *velocityLMS, double *tshift );

void    VelDecelFit_LMS( struct trajectory_info *traj, double velocity_input, double *velocityLMS, double *decel1LMS, double *decel2LMS );

void    ConstantVelocity_MultiTrackFit( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt,
                                        double *velocityLMS, double *xo, double *rmserror, double *err );

void    LinearVelocity_MultiTrackFit( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt,
                                      double *velocityLMS, double *decel1LMS, double *decel2LMS, double *rmserror, double *err );

void    QuadraticVelocity_MultiTrackFit( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt,
                                         double *velocityLMS, double *decel1LMS, double *decel2LMS, double *rmserror, double *err );

void    ExponentVelocity_MultiTrackApprox( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt,
                                           double *velocityLMS, double *decel1LMS, double *decel2LMS, double *rmserror, double *err );

void    ExponentVelocity_MultiTrackFit( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt, double velocity_input,
                                        double *velocityLMS, double *decel1LMS, double *decel2LMS, double *rmserror, double *err );

void    ParameterRefinementViaPSO( struct trajectory_info *traj, int cost_function, int velocity_model, int pso_fit_control );

int     TimeOffsetRefinement( struct trajectory_info *traj );

double  Anglesum_Measurements2Line( double *ro_ECI, double *radiant_ECI, struct trajectory_info *traj );

double  Anglesum_Measurements2Model( int velmodel,  double *ro_ECI, double *radiant_ECI,
                                     double decel1,  double decel2, double ttzero, double *toffset,
                                     struct trajectory_info *traj );

void    ReportFill_LLAVT_Meas_Model( struct trajectory_info *traj );

void    ReportFill_LLA_Beg_End( int LLA_position, struct trajectory_info *traj );

void    MonteCarlo_ErrorEstimate( struct trajectory_info *traj );

double  VectorDotProduct( double *a, double *b );

void    VectorCrossProduct( double *a, double *b, double *c );

void    VectorNorm( double *a, double *b, double *c );

double  RandomGauss( double sigma, double maxvalue );

void    ECI2RahDec( double *eci, double *rah, double *dec );

void    RahDec2ECI( double rah, double dec, double *eci );

void    AzimuthElevation2RADec( double  azim,
                                double  elev,
                                double  geodetic_latitude,
                                double  LST,
                                double *RA,
                                double *DEC );

void    RADec2AzimuthElevation( double  RA,
                                double  DEC,
                                double  geodetic_latitude,
                                double  LST,
                                double *Azim,
                                double *Elev  );

double  LocalSiderealTimeE( double jdt, double longitude_east );

double  LST2LongitudeEast( double jdt, double LST );

void    LatLonAlt2ECEF( double lat, double lon, double alt_km, double *ecef_km  );

void    ECEF2LatLonAlt( double *ecef_km, double *lat, double *lon, double *alt_km );


//############################################################################################
//                                   Functions
//############################################################################################

int   MeteorTrajectory( struct trajectory_info *traj )
{
int     kcamera, kcamera1, kcamera2, kcamera_ref, kmeas_ref, k, offsets_shifted;
double  radiant_hat[3], radiant_bestconv[3], ttzero, ttzero_limit;
double  Rbestconv[3], Rdummy[3], Rref[3], dist, magdummy;
double  max_convergence, vbegin, vapprox, decel1, decel2;


    //======== Since we now use an inverse of the noise variance per measurement to weight
    //         the minimzation cost function, first look for any zero valued standard 
    //         deviations input by the user (reseting them to the minimum sigma found), 
    //         then compute the weights as the inverse variance with the weight sum
    //         normalized to unity.

	MeasurementWeightsFromStdDev( traj );


    //======== Site positions and measured coordinates are converted into measurement unit vectors
    //           that have a common coordinate system (Earth Centered Inertial or ECI) where the
    //           conversion depends on the user's selection of input measurement type. 
	//         The times are adjusted to be relative to the first camera, first measurement and 
	//           will later be associated to the first 3 components of the parameter search (which 
	//           is NOT necessarily the earliest begin point in time). 
    //         Zero measurement noise is added to the measurements at this stage of processing.

	kcamera_ref = 0;  //... 1st camera
    kmeas_ref   = 0;  //... 1st measurement

	traj->dtime_ref = traj->dtime[kcamera_ref][kmeas_ref];


    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        Angles2SiteMeasurements( traj->camera_lat[kcamera],
                                 traj->camera_lon[kcamera],
                                 traj->camera_hkm[kcamera],
                                 traj->camera_LST[kcamera],
                                 traj->nummeas[kcamera],
                                 traj->meastype,
                                 traj->meas1[kcamera],
                                 traj->meas2[kcamera],
                                 traj->dtime[kcamera],
                                 traj->noise[kcamera],
                                 NO_NOISE,
								 traj->dtime_ref,
								 0.0,  //... no time offsets yet
                                 traj->meashat_ECI[kcamera],
                                 traj->rcamera_ECI[kcamera]  );

    } //... end of camera loop


	//======== Zero all the parameter's initial values and set their search limits

    traj->numparams = 9 + traj->numcameras;


    for( k=0; k<traj->numparams;     k++ )  traj->params[k]  = 0.0;                //... All parameters initialized to zero

    for( k=0; k<=2;                  k++ )  traj->limits[k]  = 5.0;                //... ECI begin position search limit in km
    for( k=3; k<=5;                  k++ )  traj->limits[k]  = 5.0;                //... ECI begin velocity search limit in km/sec
    for( k=6; k<=6;                  k++ )  traj->limits[k]  = 2.0;                //... Deceleration coefficient 1 search limit
    for( k=7; k<=7;                  k++ )  traj->limits[k]  = 2.0;                //... Deceleration coefficient 2 search limit
    for( k=8; k<=8;                  k++ )  traj->limits[k]  = 1.0;                //... Tzero search limit in seconds

    for( k=9; k<=8+traj->numcameras; k++ )  traj->limits[k]  = traj->max_toffset;  //... Timing offset search limit in seconds
	                                        traj->limits[9]  = 0.0;                //... First timing offset always fixed at zero
	

    //======== Tzero is a relative time fitting parameter that indicates where the 
	//           propagation models take over from a purely constant velocity.
	//         Get the starting tzero by setting it to the midpoint of all the 
    //           user provided times. Limit the tzero range from min to max
	//           unless using the constant velocity model where tzero is fixed.

    InitializeTzeroParameters( traj, &ttzero, &ttzero_limit );

	traj->params[8] = ttzero;
	traj->limits[8] = ttzero_limit;


	//======== Get a weighted multi-track intersecting planes solution (currently only reported on, not used internally)

    IntersectingPlanes_MultiTrack( traj, radiant_hat );

    ECI2RahDec( radiant_hat, &traj->ra_radiant_IPW, &traj->dec_radiant_IPW );

    if( traj->verbose == 1 )  printf(" Multi-Track Weighted IP:  RA = %lf   Dec = %lf\n", traj->ra_radiant_IPW * 57.296, traj->dec_radiant_IPW * 57.296 );


    //======== Get an initial guess for the radiant by pairing all track combinations
    //         and selecting the intersecting planes solution ONLY for the pair with the 
    //         largest convergence angle.

    IntersectingPlanes_BestConvergence( traj, &kcamera1, &kcamera2, &max_convergence, radiant_bestconv, Rbestconv  );

    ECI2RahDec( radiant_bestconv, &traj->ra_radiant_IP, &traj->dec_radiant_IP );

    if( traj->verbose == 1 )  printf(" Best Convergence Ang IP:  RA = %lf   Dec = %lf\n", traj->ra_radiant_IP * 57.296, traj->dec_radiant_IP * 57.296 );


    traj->max_convergence = max_convergence;

    if( traj->verbose == 1 )  printf(" Cameras used were %i & %i with convergence angle %lf\n\n", kcamera1, kcamera2, traj->max_convergence * 57.296 );


    //======== Use the measurement associated with dtime_ref, to find a starting
    //         position in 3D space (ECI coordinates). The starting 3D position
    //         is based on the measurement ray's CPA to the estimated radiant
    //         line with a point that lies along the measurement unit vector.

    TwoLineCPA( traj->rcamera_ECI[kcamera_ref][kmeas_ref], traj->meashat_ECI[kcamera_ref][kmeas_ref], Rref,
                Rbestconv, radiant_bestconv, Rdummy, 
				&dist );

    traj->params[0] = Rref[X];   //... Initial 3D position state vector in ECI spatial coordinates
    traj->params[1] = Rref[Y];
    traj->params[2] = Rref[Z];


    //======== Use the best radiant from the intersecting planes solution as an initial guess for the
    //         unity velocity state vector (radiant direction only) in ECI coordinates

    traj->params[3] = radiant_bestconv[X];  //... Initial 3D velocity for the IP's best convergence angle
    traj->params[4] = radiant_bestconv[Y];
    traj->params[5] = radiant_bestconv[Z];

    if( traj->verbose == 1 )  IntermediateConsoleOutput( "Intersecting Planes Solution", traj->params, traj->numcameras );


    //======== Refine position and velocity (radiant direction) via Jiri Borivicka's LMS method using the PSO

    for( k=0; k<traj->numparams; k++ )  traj->xguess[k] = traj->params[k];
    for( k=0; k<traj->numparams; k++ )  traj->xshift[k] = 0.0;

    for( k=0; k<=2; k++ )  traj->xshift[k] = traj->limits[k];  //... Vary starting position
    for( k=3; k<=5; k++ )  traj->xshift[k] = 0.02;             //... Vary the velocity (a.k.a. radiant UNIT vector at this stage)

    ParameterRefinementViaPSO( traj, MEAS2LINE, CONSTANT, traj->PSO_fit_control[0] );  //... call index 0

    VectorNorm( &traj->params[3], &traj->params[3], &magdummy );  //... Ensure radiant direction is a unit vector (set |V|=1 for now)

    ECI2RahDec( &traj->params[3], &traj->ra_radiant_LMS, &traj->dec_radiant_LMS );

    if( traj->verbose == 1 )  IntermediateConsoleOutput( "Position & Radiant LMS Solution", traj->params, traj->numcameras );


    //======== Use the earliest 10 measurements of each camera track (or all of them if 
	//           less than 10 or the fit model is constant velocity), to compute the
    //           initial velocity estimate working from absolute positions (CPA) of the
    //           rays to the current 3D radiant line. A least mean squares estimate is 
    //           made of the velocity with outlier removal. 
	//         As a by-product, the timing offsets are also estimated. Set them all 
	//           relative to the first camera.

    VelocityFit_LMS( traj, &vbegin, traj->tref_offsets );

	for( kcamera=0; kcamera<traj->maxcameras; kcamera++ )  traj->params[9+kcamera] = traj->tref_offsets[kcamera] 
	                                                                               - traj->tref_offsets[0];

	for( kcamera=0; kcamera<traj->maxcameras; kcamera++ )  traj->tref_offsets[kcamera] = traj->tref_offsets[kcamera] 
	                                                                                   - traj->tref_offsets[0];


	//======== For the non-constant velocity models, estimate the velocity with deceleration terms given
	//            the timing offsets just estimated. This will revert to the constant velocity solution
	//            if deceleration terms that should be positive are evaluated as negative (set to zero).

	if( traj->velmodel != CONSTANT )  {

		VelDecelFit_LMS( traj, vbegin, &vapprox, &decel1, &decel2 );

        traj->params[6] = fabs( decel1 );

		if( traj->velmodel == EXPONENT )  traj->params[7] = fabs( decel2 );
		else                              traj->params[7] = decel2;

		////printf(" INITIAL DECEL GUESS  %12.3le  %8.3lf \n", traj->params[6], traj->params[7] );

	}


	//======== Velocity state vector is now scaled up from a radiant unit vector using the
	//             velocity magnitude "vbegin" estimated from the early part of the
	//             measurements. That is, assuming a constant velocity model in the early part
	//             of the meteor's recorded track.

    traj->params[3] *= vbegin;
    traj->params[4] *= vbegin;
    traj->params[5] *= vbegin;


	//======== Refine velocity & deceleration terms only (all other parameters fixed) via the PSO for
	//            the non-constant velocity motion models.

	if( traj->velmodel != CONSTANT )  { 

        for( k=0; k<traj->numparams; k++ )  traj->xguess[k] = traj->params[k];
        for( k=0; k<traj->numparams; k++ )  traj->xshift[k] = 0.0;

		for( k=3; k<6;               k++ )  traj->xshift[k] = traj->limits[k];  //... Vary velocity (radiant direction)

        if( traj->velmodel >= 1 )  traj->xshift[6] = traj->limits[6];  //... Vary decel1
        if( traj->velmodel >= 2 )  traj->xshift[7] = traj->limits[7];  //... Vary decel2
        if( traj->velmodel == 1 )  traj->xshift[8] = traj->limits[8];  //... Vary ttzero
        if( traj->velmodel == 2 )  traj->xshift[8] = traj->limits[8];  //... Vary ttzero

        ParameterRefinementViaPSO( traj, MEAS2MODEL, traj->velmodel, traj->PSO_fit_control[1] );  //... call index 1

	    if( traj->verbose == 1 )  IntermediateConsoleOutput( "PSO: Velocity & Deceleration Estimation", traj->params, traj->numcameras );

	}


	//======== Refine ALL parameters EXCEPT deceleration terms to improve the general solution via the PSO

    for( k=0; k<traj->numparams; k++ )  traj->xguess[k] = traj->params[k];
    for( k=0; k<traj->numparams; k++ )  traj->xshift[k] = traj->limits[k];

	traj->xshift[6] = 0.0;  //... fix decel1
	traj->xshift[7] = 0.0;  //... fix decel2
	traj->xshift[8] = 0.0;  //... fix ttzero

    ParameterRefinementViaPSO( traj, MEAS2MODEL, traj->velmodel, traj->PSO_fit_control[2] );  //... call index 2

    if( traj->verbose == 1 )  IntermediateConsoleOutput( "PSO: ALL parameters EXCEPT deceleration", traj->params, traj->numcameras );


	//======== Update the site position vectors since there is a timing offset which shifts the LST slightly

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        Angles2SiteMeasurements( traj->camera_lat[kcamera],
                                 traj->camera_lon[kcamera],
                                 traj->camera_hkm[kcamera],
                                 traj->camera_LST[kcamera],
                                 traj->nummeas[kcamera],
                                 traj->meastype,
                                 traj->meas1[kcamera],
                                 traj->meas2[kcamera],
                                 traj->dtime[kcamera],
                                 traj->noise[kcamera],
                                 NO_NOISE,
								 traj->dtime_ref,
								 traj->params[9+kcamera],
                                 traj->meashat_ECI[kcamera],
                                 traj->rcamera_ECI[kcamera]  );

    } //... end of camera loop


    //======== Refine ALL parameters simultaneously to get an accurate solution via the PSO

    for( k=0; k<traj->numparams; k++ )  traj->xguess[k] = traj->params[k];
    for( k=0; k<traj->numparams; k++ )  traj->xshift[k] = traj->limits[k];

    if( traj->velmodel == 0 )  traj->xshift[6] = 0.0;  //... Constrain decel1 due to velocity model
    if( traj->velmodel <= 1 )  traj->xshift[7] = 0.0;  //... Constrain decel2 due to velocity model
    if( traj->velmodel == 0 )  traj->xshift[8] = 0.0;  //... Constrain ttzero due to velocity model
    if( traj->velmodel == 3 )  traj->xshift[8] = 0.0;  //... Constrain ttzero due to velocity model
	 
    ParameterRefinementViaPSO( traj, MEAS2MODEL, traj->velmodel, traj->PSO_fit_control[3] );  //... call index 3

    if( traj->verbose == 1 )  IntermediateConsoleOutput( "PSO: 1st Solution on ALL parameters", traj->params, traj->numcameras );


	//======== Shift timing offsets by the +-3.5 times the sample time spacing to avoid any local minima.

	offsets_shifted = TimeOffsetRefinement( traj );


	//======== Update the measurements before the final solution since there is a timing offset which shifts the LST slightly

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        Angles2SiteMeasurements( traj->camera_lat[kcamera],
                                 traj->camera_lon[kcamera],
                                 traj->camera_hkm[kcamera],
                                 traj->camera_LST[kcamera],
                                 traj->nummeas[kcamera],
                                 traj->meastype,
                                 traj->meas1[kcamera],
                                 traj->meas2[kcamera],
                                 traj->dtime[kcamera],
                                 traj->noise[kcamera],
                                 NO_NOISE,
								 traj->dtime_ref,
								 traj->params[9+kcamera],
                                 traj->meashat_ECI[kcamera],
                                 traj->rcamera_ECI[kcamera]  );

    } //... end of camera loop


	//======== Perform another PSO pass to refine ALL parameters simultaneously to get FINAL SOLUTION.

    for( k=0; k<traj->numparams; k++ )  traj->xguess[k] = traj->params[k];
    for( k=0; k<traj->numparams; k++ )  traj->xshift[k] = traj->limits[k];

    if( traj->velmodel == 0 )  traj->xshift[6] = 0.0;  //... Constrain decel1 due to velocity model
    if( traj->velmodel <= 1 )  traj->xshift[7] = 0.0;  //... Constrain decel2 due to velocity model
    if( traj->velmodel == 0 )  traj->xshift[8] = 0.0;  //... Constrain ttzero due to velocity model
    if( traj->velmodel == 3 )  traj->xshift[8] = 0.0;  //... Constrain ttzero due to velocity model

	 
    ParameterRefinementViaPSO( traj, MEAS2MODEL, traj->velmodel, traj->PSO_fit_control[4] );  //... call index 4

    if( traj->verbose == 1 )  IntermediateConsoleOutput( "PSO: 2nd Solution on ALL parameters", traj->params, traj->numcameras );


    //======== Assign return parameters for final solution, radiant, velocity, deceleration, tzero, and time offsets

    for( k=0; k<traj->numparams; k++ )  traj->solution[k] = traj->params[k];

	                                    traj->solution[6] = fabs( traj->params[6] );  // force to be positive for ALL models

	if( traj->velmodel == EXPONENT )    traj->solution[7] = fabs( traj->params[7] );  // force to be positive for EXPONENT


    ECI2RahDec( &traj->solution[3], &traj->ra_radiant, &traj->dec_radiant );

    VectorNorm( &traj->solution[3], radiant_hat, &traj->vbegin );

    traj->decel1      = traj->solution[6];
    traj->decel2      = traj->solution[7];

	traj->ttzero      = traj->solution[8];

	traj->dtime_tzero = traj->solution[8] + traj->dtime_ref;

	if( traj->velmodel == CONSTANT )  traj->dtime_tzero = 0.0;
	if( traj->velmodel == EXPONENT )  traj->dtime_tzero = 0.0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ ) traj->tref_offsets[kcamera] = traj->solution[9+kcamera];  //... should be referenced to 1st camera


    //======== Assign return LLA, velocity and time parameters for the measurements
    //         at their closest point of approach (CPA) to the radiant line for
    //         both measurements and model positions.

    ReportFill_LLAVT_Meas_Model( traj );


    //======== Assign LLA parameters for the BEGIN and END positions and associated
    //         time offsets relative to the reference Julian date/time.

    ReportFill_LLA_Beg_End( LLA_BEG, traj );
    ReportFill_LLA_Beg_End( LLA_END, traj );

	traj->dtime_beg = traj->ttbeg + traj->dtime_ref;  //.. report in user time units (relative to jdt_ref)
	traj->dtime_end = traj->ttend + traj->dtime_ref;


    //======== Now run a noisy measurement Monte Carlo set of trials to estimate
    //         the error in each parameter. The no noise "solution" parameters
    //         will be used as the starting point for each minimization.

    MonteCarlo_ErrorEstimate( traj );


    //======== Diagnostic standard deviation output to console

    if( traj->verbose != 0 )  {
		
        printf("     Rah Dec sigma  %lf  %lf\n",        traj->ra_sigma * 57.29577951,     traj->dec_sigma * 57.29577951 );
        printf("     Vel Acc sigma  %lf  %lf  %lf\n",   traj->vbegin_sigma,               traj->decel1_sigma,               traj->decel2_sigma );
        printf("     LLA beg sigma  %lf  %lf  %lf\n",   traj->rbeg_lon_sigma * 57.29577951, traj->rbeg_lat_sigma * 57.29577951, traj->rbeg_hkm_sigma );
        printf("     LLA end sigma  %lf  %lf  %lf\n\n", traj->rend_lon_sigma * 57.29577951, traj->rend_lat_sigma * 57.29577951, traj->rend_hkm_sigma );
        
	}


    return(0);
}


//##################################################################################
//
//======== Function to report on critical parameters during intermediate steps using
//         the "verbose" mode.
//==================================================================================

void  IntermediateConsoleOutput( char *ctitle, double *params, int ncams )
{
double  rah, dec, radiant_hat[3], Vbegin;

    ECI2RahDec( &params[3], &rah, &dec );

    VectorNorm( &params[3], radiant_hat, &Vbegin );

	/*PETE
    printf(" %s\n", ctitle );
    printf(" ------------------------------------------------------\n" );
    printf("     Radiant at %lf %lf \n", rah * 57.29577951, dec * 57.29577951 );
    printf("     Ro         %lf %lfs %lf \n", params[0], params[1], params[2] );
    printf("     Vel Acc    %lf %le %le \n", Vbegin, fabs(params[6]), fabs(params[7]) );
    printf("     Tzero      %lf \n", params[8] );
	int kcam;
    for( kcam=0; kcam<ncams; kcam++ ) printf("     Timing Offsets %lf \n", params[9+kcam] );
    printf("\n");
	*/
}


//##################################################################################
//
//======== Function to reset any standard deviations that were input as zero to the
//         minimum non-zero value found amongst the full data set.
//==================================================================================

void    EnsureNonZeroStdDev( struct trajectory_info *traj )
{
int     kmeas, kcamera;
double  sigma, minsigma;


    //======== Find the smallest (nonzero) and largest standard deviation

    minsigma = +1.0e+30;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  {

            sigma = fabs( traj->noise[kcamera][kmeas] );

            if( sigma < minsigma  &&  sigma > 0.0 )  minsigma = sigma;

        }  //... end of measurement loop per camera

    } //... end of camera loop


	if( minsigma < 1.0e-8 )  minsigma = 1.0e-8;   //... Set to at least 0.05 arcseconds


    //======== Set the smallest standard deviation for any values that are too small

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  {

            if( traj->noise[kcamera][kmeas] < minsigma )  traj->noise[kcamera][kmeas] = minsigma;

        }  //... end of measurement loop per camera

    } //... end of camera loop


}

//##################################################################################
//
//======== Function to set the measurement weights based on the inverse of the input
//         noise variance of each measurement. This looks for any zero valued
//         standard deviations and sets them to the minimum found. The std dev is
//         never allowed to be less than 0.05 arc-seconds (1.0e-8 radians). All 
//         weights are normalized to a unity sum.
//==================================================================================

void    MeasurementWeightsFromStdDev( struct trajectory_info *traj )
{
int     kmeas, kcamera;
double  wnorm;


    //======== Make sure no sigma < 0.05 arc-seconds, resets traj->noise as necessary

    EnsureNonZeroStdDev( traj );


    //======== Set the measurement weights to the inverse of the variance per measurement

	wnorm = 0.0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  {

            traj->weight[kcamera][kmeas] = 1.0 / ( traj->noise[kcamera][kmeas] * traj->noise[kcamera][kmeas] );

			wnorm += traj->weight[kcamera][kmeas];

        }  //... end of measurement loop per camera

    } //... end of camera loop


    //======== Normalize the measurement weights to unity sum

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  {

            traj->weight[kcamera][kmeas] /= wnorm;

        }  //... end of measurement loop per camera

    } //... end of camera loop


}


//##################################################################################
//
//======== Function to convert angle-angle-time to measurement rays and obtain the
//         site position, both in ECI coordinates.
//==================================================================================

void   Angles2SiteMeasurements( double    camera_lat,  double   camera_lon,  double  camera_hkm,  double camera_LST,
                                int       nummeas,     int      meastype,
                                double   *meas_angle1, double  *meas_angle2, double *dtime,
                                double   *noise,       int      noise_flag,  double  dtime_ref,  double tshift,
                                double  **mhat_camera, double **rs_camera )
{
int     kmeas;
double  ecef[3], rhat[3], uhat[3], vhat[3], zhat[3], mvec[3], mag, LSTshift, RApersecond;
double  Rcenter, camera_lat_geocentric, sigma, azim, elev, ra, dec, pi;


    //======== Convert each measurement to ECI for a specific camera

    pi = 4.0 * atan(1.0);

	RApersecond = 2.0 * pi * 1.00273785 / 86400.0;

    LatLonAlt2ECEF( camera_lat, camera_lon, camera_hkm, ecef );

    Rcenter = sqrt( ecef[X]*ecef[X] + ecef[Y]*ecef[Y] + ecef[Z]*ecef[Z] );

    camera_lat_geocentric = atan( ecef[Z] / sqrt( ecef[X]*ecef[X] + ecef[Y]*ecef[Y] ) );

    for( kmeas=0; kmeas<nummeas; kmeas++ )  {

        //-------- Site position vectors in ECI coords relative to center of Earth

		LSTshift = RApersecond * ( dtime[kmeas] - dtime_ref + tshift );

		rs_camera[kmeas][X] = Rcenter * cos( camera_lat_geocentric ) * cos( camera_LST + LSTshift );
        rs_camera[kmeas][Y] = Rcenter * cos( camera_lat_geocentric ) * sin( camera_LST + LSTshift );
        rs_camera[kmeas][Z] = Rcenter * sin( camera_lat_geocentric );


        //-------- Measurement unit vectors in ECI (ECEF) coords
        //           NOTE: "azim" = +east of north,
        //                 "elev" = +altitude angle above horizon

        if( meastype == RODEC )  {  //... each RA was calculated using the LST of the measurement time

            ra  = meas_angle1[kmeas];

            dec = meas_angle2[kmeas];

        }
        else if( meastype == RADEC )  {  //... each RA was calculated at a fixed LST (assumed ref time)

            ra  = meas_angle1[kmeas] + LSTshift;

            dec = meas_angle2[kmeas];

        }
        else if( meastype == NAZEL )  {  //... convert from azimuth +east of north and altitude angle

            azim = meas_angle1[kmeas];

            if( azim <  0.0      )  azim += 2.0 * pi;
            if( azim >= 2.0 * pi )  azim -= 2.0 * pi;

            elev = meas_angle2[kmeas];

            AzimuthElevation2RADec( azim, elev, camera_lat, camera_LST + LSTshift, &ra, &dec );

        }
        else if( meastype == SAZZA )  {  //... convert from azimuth +west of south and zenith angle

            azim = pi + meas_angle1[kmeas];

            if( azim <  0.0      )  azim += 2.0 * pi;
            if( azim >= 2.0 * pi )  azim -= 2.0 * pi;

            elev = pi / 2.0 - meas_angle2[kmeas];

            AzimuthElevation2RADec( azim, elev, camera_lat, camera_LST + LSTshift, &ra, &dec );

        }
        else if( meastype == EAZZA )  {  //... convert from azimuth +north of east and zenith angle

            azim = pi / 2.0 - meas_angle1[kmeas];

            if( azim <  0.0      )  azim += 2.0 * pi;
            if( azim >= 2.0 * pi )  azim -= 2.0 * pi;

            elev = pi / 2.0 - meas_angle2[kmeas];

            AzimuthElevation2RADec( azim, elev, camera_lat, camera_LST + LSTshift, &ra, &dec );

        }
        else  {
            printf(" ====> ERROR in Angles2SiteMeasurements: meastype %i not implemented\n", meastype );
			Delay_msec(15000);
            exit(1);
        }


        //-------- Convert equatorial to ECI coordinates

        RahDec2ECI( ra, dec, rhat );  //...Returns a unit vector

        if( rhat[Z] < 0.0 )  {  //... Southern Hemisphere
            zhat[X] =  0.0;
            zhat[Y] =  0.0;
            zhat[Z] = +1.0;
            VectorCrossProduct( rhat, zhat, uhat );
            VectorNorm( uhat, uhat, &mag );
            VectorCrossProduct( uhat, rhat, vhat );
            VectorNorm( vhat, vhat, &mag );
        }
        else  {                 //... Northern Hemisphere
            zhat[X] =  0.0;
            zhat[Y] =  0.0;
            zhat[Z] = -1.0;
            VectorCrossProduct( zhat, rhat, uhat );
            VectorNorm( uhat, uhat, &mag );
            VectorCrossProduct( uhat, rhat, vhat );
            VectorNorm( vhat, vhat, &mag );
        }

        //-------- Noise added assumes a small angle approximation that tangent(sigma in radians) ~ sigma

        if( noise_flag == NO_NOISE )  sigma = 0.0;
        else                          sigma = noise[kmeas] / sqrt(2.0);  //...sqrt(2)/2 * noisesigma in each orthogonal dimension

        mvec[X] = rhat[X] + RandomGauss( sigma, 3.0*sigma ) * uhat[X] + RandomGauss( sigma, 3.0*sigma ) * vhat[X];
        mvec[Y] = rhat[Y] + RandomGauss( sigma, 3.0*sigma ) * uhat[Y] + RandomGauss( sigma, 3.0*sigma ) * vhat[Y];
        mvec[Z] = rhat[Z] + RandomGauss( sigma, 3.0*sigma ) * uhat[Z] + RandomGauss( sigma, 3.0*sigma ) * vhat[Z];

        VectorNorm( mvec, mhat_camera[kmeas], &mag );  //... Normalize to a unit vector

		//-------- Internally this module uses a time relative to a reference time that has been
		//            associated with a 3D reference point. The propagation model will compute
		//            distances from the new time = 0 (i.e. relative to the reference position)

        mhat_camera[kmeas][T] = dtime[kmeas] - dtime_ref;   //... seconds relative to a starting position reference time

    } //... end of measurement loop


}


//##################################################################################
//
//======== Function to compute the intersecting planes solution for two measurement
//         sets given number of measurements, ECI measurement rays X, Y, Z, and T,
//         and the site coords in ECI. Returns the radiant direction, convergence
//         angle, and CPA range vector to the 3D line from each site.
//==================================================================================

void   IntersectingPlanes( int     nmeas_camera1, double **meashat_camera1, double *rcamera1,
                           int     nmeas_camera2, double **meashat_camera2, double *rcamera2,
                           double *radiant_hat, double  *convergence_angle, double *rcpa_camera1, double *rcpa_camera2  )
{
int     kbeg, kend;
double  n1_hat[3], n2_hat[3], radiant[3], w1[3], w2[3], rcamera_diff[3];
double  dotw1w2, range_cpa1, range_cpa2, mag, cosangle;


    //======== This is a multi-ray solution to obtain the normal to each measurement plane.

    Normal2MeteorMeasurements( nmeas_camera1, meashat_camera1, n1_hat );

    Normal2MeteorMeasurements( nmeas_camera2, meashat_camera2, n2_hat );


    //======== Get the radiant unit vector via cross product of the normals

    VectorCrossProduct( n1_hat, n2_hat, radiant );

    VectorNorm( radiant, radiant_hat, &mag );

    //======== If closer to the anti-radiant, then reverse sign

    if( nmeas_camera1 >= 4 )  {
        kbeg = 1;
        kend = nmeas_camera1-2;
    }
    else  {
        kbeg = 0;
        kend = nmeas_camera1-1;
    }

    if( VectorDotProduct( meashat_camera1[kbeg], radiant_hat )
      < VectorDotProduct( meashat_camera1[kend], radiant_hat ) )  {
        radiant_hat[X] = -radiant_hat[X];
        radiant_hat[Y] = -radiant_hat[Y];
        radiant_hat[Z] = -radiant_hat[Z];
    }


    //======== Compute the convergence angle

    cosangle = VectorDotProduct( n1_hat, n2_hat );

    if( cosangle > +1.0 )  cosangle = +1.0;
    if( cosangle < -1.0 )  cosangle = -1.0;

    *convergence_angle = acos( fabs( cosangle ) );


    //======== Compute the cpa distance from the two cameras to the radiant line
    //         (i.e the range from a camera site to the line along the normal to
    //         the line) and return as a vector with that distance as magnitude.

    VectorCrossProduct( radiant_hat, n1_hat, w1 );

    VectorNorm( w1, w1, &mag );

    if( VectorDotProduct( w1, meashat_camera1[kbeg] ) < 0.0 )  {
        w1[X] = -w1[X];
        w1[Y] = -w1[Y];
        w1[Z] = -w1[Z];
    }


    VectorCrossProduct( radiant_hat, n2_hat, w2 );

    VectorNorm( w2, w2, &mag );

    if( VectorDotProduct( w2, meashat_camera2[kbeg] ) < 0.0 )  {
        w2[X] = -w2[X];
        w2[Y] = -w2[Y];
        w2[Z] = -w2[Z];
    }


    rcamera_diff[X] = rcamera1[X] - rcamera2[X];
    rcamera_diff[Y] = rcamera1[Y] - rcamera2[Y];
    rcamera_diff[Z] = rcamera1[Z] - rcamera2[Z];

    dotw1w2 = VectorDotProduct( w1, w2 );

    range_cpa1 = ( dotw1w2 * VectorDotProduct( rcamera_diff, w2 ) - VectorDotProduct( rcamera_diff, w1 ) )
                / ( 1.0 - dotw1w2 * dotw1w2 );

    rcpa_camera1[X] = range_cpa1 * w1[X];
    rcpa_camera1[Y] = range_cpa1 * w1[Y];
    rcpa_camera1[Z] = range_cpa1 * w1[Z];


    range_cpa2 = ( VectorDotProduct( rcamera_diff, w2 ) - dotw1w2 * VectorDotProduct( rcamera_diff, w1 ) )
                / ( 1.0 - dotw1w2 * dotw1w2 );

    rcpa_camera2[X] = range_cpa2 * w2[X];
    rcpa_camera2[Y] = range_cpa2 * w2[Y];
    rcpa_camera2[Z] = range_cpa2 * w2[Z];

}

//##################################################################################
//
//======== Pair all track combinations and select the intersecting planes solution
//         corresponding to the pair with the largest convergence angle.
//==================================================================================

void   IntersectingPlanes_BestConvergence( struct trajectory_info *traj,
                                           int    *kcamera1, 
										   int    *kcamera2, 
										   double *max_convergence,
                                           double *radiant_bestconv, 
										   double *r_bestconv  )
{
int     kcam1, kcam2;
double  radiant_hat[3], convergence_angle, rcpa_camera1[3], rcpa_camera2[3];


    *max_convergence = 0.0;

    *kcamera1 = 0;
    *kcamera2 = 1;

    for( kcam1=0; kcam1<traj->numcameras; kcam1++ ) {

        for( kcam2=kcam1+1; kcam2<traj->numcameras; kcam2++ ) {

             IntersectingPlanes( traj->nummeas[kcam1], traj->meashat_ECI[kcam1], traj->rcamera_ECI[kcam1][0],
                                 traj->nummeas[kcam2], traj->meashat_ECI[kcam2], traj->rcamera_ECI[kcam2][0],
                                 radiant_hat, &convergence_angle, rcpa_camera1, rcpa_camera2 );

             if( convergence_angle > *max_convergence )  {

                 *max_convergence = convergence_angle;

                 radiant_bestconv[X] = radiant_hat[X];
                 radiant_bestconv[Y] = radiant_hat[Y];
                 radiant_bestconv[Z] = radiant_hat[Z];

                 //... CPA position on the line w.r.t the camera in ECI (used later for starting position)

                 r_bestconv[X] = traj->rcamera_ECI[kcam1][0][X] + rcpa_camera1[X];
                 r_bestconv[Y] = traj->rcamera_ECI[kcam1][0][Y] + rcpa_camera1[Y];
                 r_bestconv[Z] = traj->rcamera_ECI[kcam1][0][Z] + rcpa_camera1[Z];

                 *kcamera1 = kcam1;
				 *kcamera2 = kcam2;

             }

        }  //... end kcamera2 loop

    }  //... end kcamera1 loop

}


//##################################################################################
//
//======== Uniquely pair all track combinations and perform a weighted combination
//         of intersecting planes solutions.
//==================================================================================

void   IntersectingPlanes_MultiTrack( struct trajectory_info *traj, double *radiant_hat  )
{
int     kcamera1, kcamera2, lastmeas;
double  radiant_sum[3], convergence_angle, rcpa_camera1[3], rcpa_camera2[3];
double  obsangle1, obsangle2, weight, wsum, rmagnitude;


    wsum = 0.0;

    radiant_sum[X] = 0.0;
    radiant_sum[Y] = 0.0;
    radiant_sum[Z] = 0.0;

    for( kcamera1=0; kcamera1<traj->numcameras; kcamera1++ ) {

        lastmeas = traj->nummeas[kcamera1] - 1;

        obsangle1 = acos( VectorDotProduct( traj->meashat_ECI[kcamera1][0], traj->meashat_ECI[kcamera1][lastmeas] ) );

        for( kcamera2=kcamera1+1; kcamera2<traj->numcameras; kcamera2++ ) {

            lastmeas = traj->nummeas[kcamera2] - 1;

            obsangle2 = acos( VectorDotProduct( traj->meashat_ECI[kcamera2][0], traj->meashat_ECI[kcamera2][lastmeas] ) );

            IntersectingPlanes( traj->nummeas[kcamera1], traj->meashat_ECI[kcamera1], traj->rcamera_ECI[kcamera1][0],
                                traj->nummeas[kcamera2], traj->meashat_ECI[kcamera2], traj->rcamera_ECI[kcamera2][0],
                                radiant_hat, &convergence_angle, rcpa_camera1, rcpa_camera2 );

            weight = obsangle1 * obsangle2 * sin( convergence_angle ) * sin( convergence_angle );

            wsum += weight;

            radiant_sum[X] += weight * radiant_hat[X];
            radiant_sum[Y] += weight * radiant_hat[Y];
            radiant_sum[Z] += weight * radiant_hat[Z];

        }  //... end kcamera2 loop

    }  //... end kcamera1 loop

    radiant_sum[X] /= wsum;
    radiant_sum[Y] /= wsum;
    radiant_sum[Z] /= wsum;

    VectorNorm( radiant_sum, radiant_hat, &rmagnitude );

}


//##################################################################################
//
//======== Function to compute the normal to the plane defined by a set of
//         measurement rays from one camera that followed the meteor track. The
//         normal is in the same coords as the measurement unit vectors (typ. ECI).
//==================================================================================

void  Normal2MeteorMeasurements( int nmeas, double **meashat_ECI, double *planenormal )
{
int     k, kbeg, kend;
double  xdotx, xdoty, xdotz, ydoty, ydotz;
double  sx, sy, smag, denom;


    //======== The plane is really undefined for a single measurement ray

    if( nmeas == 1 )  {
        smag = sqrt( meashat_ECI[0][X] * meashat_ECI[0][X] + meashat_ECI[0][Y] * meashat_ECI[0][Y] );
        planenormal[X] = -meashat_ECI[0][Y] / smag;
        planenormal[Y] = +meashat_ECI[0][X] / smag;
        planenormal[Z] =  0.0;
        return;
    }


    //======== Compute the plane using the second through second-to-last measurements
    //         unless there are fewer than four measurements, then first to last.

    if( nmeas >= 4 )  {
        kbeg = 1;
        kend = nmeas-2;
    }
    else  {
        kbeg = 0;
        kend = nmeas-1;
    }


    //======== Compute running sums of dot products

    xdotx = 0.0;
    xdoty = 0.0;
    xdotz = 0.0;
    ydoty = 0.0;
    ydotz = 0.0;

    for( k=kbeg; k<=kend; k++ )  {

         xdotx += meashat_ECI[k][X] * meashat_ECI[k][X];
         xdoty += meashat_ECI[k][X] * meashat_ECI[k][Y];
         xdotz += meashat_ECI[k][X] * meashat_ECI[k][Z];
         ydoty += meashat_ECI[k][Y] * meashat_ECI[k][Y];
         ydotz += meashat_ECI[k][Y] * meashat_ECI[k][Z];
    }


    //======== Solve for the unit vector normal to the meteor plane

    denom = xdotx * ydoty - xdoty * xdoty;

    sx = ( ydoty * xdotz - xdoty * ydotz ) / denom;

    sy = ( xdotx * ydotz - xdoty * xdotz ) / denom;

    smag = sqrt( 1.0 + sx*sx + sy*sy );

    planenormal[X] =   sx / smag;
    planenormal[Y] =   sy / smag;
    planenormal[Z] = -1.0 / smag;

}

//##################################################################################
//
//======== Function to compute the points on two lines that are the closest point
//         of approach between the two lines. Each line is defined by a 3D point
//         and a 3D unit vector along the line. The resultant points on the two lines
//         that are at CPA are "r1" and "r2" with the distance between them "d21".
//==================================================================================

void   TwoLineCPA( double *position1, double *vector1, double *r1,
                   double *position2, double *vector2, double *r2, double *d21 )
{
double  d1, d2, mag_dummy, unitvector1[3], unitvector2[3], p21[3], rxm[3], pxm[3], mdotm;


    VectorNorm( vector1, unitvector1, &mag_dummy );
    VectorNorm( vector2, unitvector2, &mag_dummy );

    p21[0] = position1[0] - position2[0];
    p21[1] = position1[1] - position2[1];
    p21[2] = position1[2] - position2[2];

    VectorCrossProduct( unitvector1, unitvector2, rxm );

    VectorCrossProduct( p21, rxm, pxm );

    mdotm = VectorDotProduct( rxm, rxm );

    d1 = VectorDotProduct( pxm, unitvector2 ) / mdotm;

    r1[0] = position1[0] + d1 * unitvector1[0];
    r1[1] = position1[1] + d1 * unitvector1[1];
    r1[2] = position1[2] + d1 * unitvector1[2];

    d2 = VectorDotProduct( pxm, unitvector1 ) / mdotm;

    r2[0] = position2[0] + d2 * unitvector2[0];
    r2[1] = position2[1] + d2 * unitvector2[1];
    r2[2] = position2[2] + d2 * unitvector2[2];

    *d21 = fabs( VectorDotProduct( p21, rxm ) ) / sqrt( mdotm );

}


//##################################################################################
//
//======== Function to compute a motion model for either position or velocity
//         propagation of the meteor. It uses either a constant velocity motion model,
//         a linear deceleration model, a quadratic deceleration model, or Jacchia's
//         exponential motion model. This function will compute the position distance
//         or velocity from time = 0 to the time "t". Tzero is the model starting time
//         where the deceleration models base their t=0 point. Note that all times
//         in this function are relative to a reference dtime_ref.
//===================================================================================


double  Propagation( int propagation_state, int velmodel, double tt, double ttzero, double vbegin, double decel1, double decel2 )
{
double  ttpoly, xconstV;


	//====== Note: The polynomial propagation models are used only for tt > ttzero, in addition to the 
    //             constant velocity model for the time period < ttzero. 

	if( tt < ttzero )  {
		xconstV = fabs(vbegin) * tt;     //... Constant velocity model contribution up to time tt
		ttpoly  = 0.0;                   //      and NO contribution to polynomial deceleration terms
	}
	else  {
		xconstV = fabs(vbegin) * ttzero; //... Constant velocity model contribution up to time ttzero
		ttpoly  = tt - ttzero;           //      plus contribution to polynomial deceleration terms
	}


    if( propagation_state == POSITION )  {
		
        if(      velmodel == CONSTANT  ) return( xconstV + fabs(vbegin) * ttpoly );
        else if( velmodel == LINEAR    ) return( xconstV + fabs(vbegin) * ttpoly - fabs(decel1) * ttpoly * ttpoly / 2.0 );
        else if( velmodel == QUADRATIC ) return( xconstV + fabs(vbegin) * ttpoly - fabs(decel1) * ttpoly * ttpoly / 2.0 + decel2 * ttpoly * ttpoly * ttpoly / 3.0 );
        else if( velmodel == EXPONENT  ) return(           fabs(vbegin) * tt     - fabs(decel1) * exp( fabs(decel2) * tt ) );

        else printf("ERROR--> In Propagation of MeteorTrajectory: velocity model %d not implemented \n", velmodel );
    }


    else if( propagation_state == VELOCITY )  {

        if(      velmodel == CONSTANT  ) return( fabs(vbegin) );
        else if( velmodel == LINEAR    ) return( fabs(vbegin) - fabs(decel1) * ttpoly );
        else if( velmodel == QUADRATIC ) return( fabs(vbegin) - fabs(decel1) * ttpoly + decel2 * ttpoly * ttpoly );
        else if( velmodel == EXPONENT  ) return( fabs(vbegin) - fabs(decel1 * decel2) * exp( fabs(decel2) * tt ) );

        else printf("ERROR--> In Propagation of MeteorTrajectory: velocity model %d not implemented \n", velmodel );
    }

    else  printf("ERROR--> In Propagation of MeteorTrajectory: propagation_state %d not implemented \n", propagation_state );

    return(-999.0);

}

//##################################################################################
//
//======== Function to find the starting reference time "tzero" as needed in some
//         of the velocity models. It is set to the midpoint between the min and
//         max times of all camera measurements with tzero_limit = half of the
//         full time range.
//==================================================================================

void   InitializeTzeroParameters( struct trajectory_info *traj, double *ttzero, double *ttzero_limit )
{
int       kcamera, kmeas;
double    ttmin, ttmax;


    //======== Constant velocity and exponential models have no time reference point

    if( traj->velmodel == CONSTANT  ||  traj->velmodel == EXPONENT )  {
		*ttzero       = 0.0;
		*ttzero_limit = 0.0;  //... constrained to not change in PSO fitting
		return;
	}


    //======== Compute the minimum and maximum relative time for all cameras

    ttmin  = +1.0e+30;
    ttmax  = -1.0e+30;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ ) {

        for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  {

            if( ttmin > traj->meashat_ECI[kcamera][kmeas][T] )
				ttmin = traj->meashat_ECI[kcamera][kmeas][T];

            if( ttmax < traj->meashat_ECI[kcamera][kmeas][T] )
				ttmax = traj->meashat_ECI[kcamera][kmeas][T];

        }
    }


	*ttzero       = 0.5 * ( ttmax + ttmin );
	*ttzero_limit = 0.5 * ( ttmax - ttmin );

}
	

//##################################################################################
//
//======== Function to find the velocity from a set of positions and times along
//         the radiant line by differencing adjacent-in-time positions. The set
//         of velocities are averaged and their standard deviation computed so
//         an iterative outlier removal can be used to obtain a more robust mean
//         velocity estimate.
//==================================================================================

double   VelocityFit_Differencing( struct trajectory_info *traj )
{
int       kcamera, kmeas, ntotalmeas, nmeas, k, kbeg, kend, kloop;
double   *vel, velsum, velssq, velave, velstd;
double    time_difference, rbeg[3], rend[3], rdummy[3], distance;


    //======== Allocate memory for velocity measurements

    ntotalmeas = 0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  ntotalmeas += traj->nummeas[kcamera];

    vel = (double*) malloc( ntotalmeas * sizeof(double) );


    //======== Compute the positional and time differencs to get the velocity mean and std dev

    velsum = 0.0;
    velssq = 0.0;
    kmeas = 0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ ) {

        //-------- Avoid first and last measurement, if possible

        kbeg = 1;
        kend = traj->nummeas[kcamera]-2;

        if( traj->nummeas[kcamera] == 3 )  {
            kbeg = 0;
            kend = 2;
        }

        if( traj->nummeas[kcamera] == 2 )  {
            kbeg = 0;
            kend = 1;
        }

        if( traj->nummeas[kcamera] <= 1 )  continue;

        //-------- Since this is a vbegin estimate - use no more than first
        //            60 interleaved frames (~1 second duration)

        if( traj->nummeas[kcamera]-2 > 60 )  kend = 60;

        //-------- Compute velocity for each temporally adjacent pair of measurements

        for( k=kbeg; k<kend; k++ )  {

            //........ Get an absolute ECI position along the measurement ray that has a CPA to the radiant line

            TwoLineCPA( traj->rcamera_ECI[kcamera][k],   traj->meashat_ECI[kcamera][k],   rbeg,
                        &traj->params[0], &traj->params[3], rdummy, 
						&distance );

            TwoLineCPA( traj->rcamera_ECI[kcamera][k+1], traj->meashat_ECI[kcamera][k+1], rend,
                        &traj->params[0], &traj->params[3], rdummy, 
						&distance );

            distance = sqrt( ( rend[X] - rbeg[X] ) * ( rend[X] - rbeg[X] ) +
                             ( rend[Y] - rbeg[Y] ) * ( rend[Y] - rbeg[Y] ) +
                             ( rend[Z] - rbeg[Z] ) * ( rend[Z] - rbeg[Z] )   );

            time_difference = traj->meashat_ECI[kcamera][k+1][T] - traj->meashat_ECI[kcamera][k][T];

            vel[kmeas] = distance / time_difference;   // Velocity estimate

            velsum += vel[kmeas];
            velssq += vel[kmeas] * vel[kmeas];

            kmeas++;

        }
    }

    nmeas = kmeas;
    velave = velsum / (double)kmeas;
    velstd = sqrt( ( velssq - (double)kmeas * velave * velave ) / (double)(kmeas-1) );


    //........ Iterate to remove outliers from mean velocity calculation

    for( kloop=0; kloop<4; kloop++ )  {

        velsum = 0.0;
        velssq = 0.0;
        kmeas  = 0;

        for( k=0; k<nmeas; k++ )  {
            if( fabs( vel[k] - velave ) < 2.0 * velstd )  {
                velsum += vel[k];
                velssq += vel[k] * vel[k];
                kmeas++;
            }
        }

        if( kmeas < 3 )  break;

        velave = velsum / (double)kmeas;
        velstd = sqrt( ( velssq - (double)kmeas * velave * velave ) / (double)(kmeas-1) );

    }

    free( vel );

    return( velave );

}

//##################################################################################
//
//======== Function to find the velocity from a set of positions and times along
//         the radiant line by using an LMS fit to multiple measurement sequences.
//         The RMS error is used on a second pass to remove outliers in the final
//         LMS fit. Note that for velocity models other than the constant velocity
//         model, only the first half of the measurements are used, but no less than
//         ten measurements if available.
//==================================================================================

void   VelocityFit_LMS( struct trajectory_info *traj, double *velocityLMS, double *tshift )
{
int       kcamera, kmeas, ntotalmeas, k, kbeg, kend, kmeas_kept, nmeas_kept;
int      *nmeas_per_camera;
double   *pos, *tim, *wgt, *xo, rmserror, *err;
double    rbeg[3], rend[3], rdummy[3], distance;



    //======== Allocate memory for the positional measurements and time stamps actually used,
    //         plus the measurement count and starting position per camera

    ntotalmeas = 0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  ntotalmeas += traj->nummeas[kcamera];

    pos           = (double*) malloc( ntotalmeas       * sizeof(double) );
    tim           = (double*) malloc( ntotalmeas       * sizeof(double) );
    wgt           = (double*) malloc( ntotalmeas       * sizeof(double) );
    err           = (double*) malloc( ntotalmeas       * sizeof(double) );
    xo            = (double*) malloc( traj->numcameras * sizeof(double) );
    nmeas_per_camera = (int*) malloc( traj->numcameras * sizeof(int)    );


    //======== Infill the positional and temporal measurements making a single vector
    //         of concatenated measurements for all the cameras.

    kmeas = 0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ ) {

        //-------- Avoid first and last measurement, unless 3 or less measurements.
		//         For up to 12 measurements, use all available dropping 1st and last
		//         For more than 12 measurements, use earliest 10 dropping the sfirst
		//         Use nearly all measurements if constant velocity model

        if( traj->nummeas[kcamera] <= 3 )  {
            kbeg = 0;
            kend = traj->nummeas[kcamera] - 1;
        }
		else if( traj->nummeas[kcamera] < 12 )  {
            kbeg = 1;
            kend = traj->nummeas[kcamera] - 2;
        }
		else  {
            kbeg = 1;
            kend = 10;
        }


        if( traj->velmodel == CONSTANT  &&  traj->nummeas[kcamera] > 3  )  {
            kbeg = 1;
            kend = traj->nummeas[kcamera] - 2;
        }


        nmeas_per_camera[kcamera] = kend - kbeg + 1;


        //-------- Set reference position to estimated begin point on the radiant line

        rbeg[X] = traj->params[0];
        rbeg[Y] = traj->params[1];
        rbeg[Z] = traj->params[2];


        //-------- Compute each positional and temporal measurement

        for( k=kbeg; k<=kend; k++ )  {

            //........ Get an absolute ECI position along the measurement ray at its CPA to the radiant line

            TwoLineCPA( traj->rcamera_ECI[kcamera][k], traj->meashat_ECI[kcamera][k], rend,
                        &traj->params[0], &traj->params[3], rdummy, 
						&distance );

            //........ Compute the distance from the reference point along the line to the measurement CPA

            pos[kmeas] = sqrt( ( rend[X] - rbeg[X] ) * ( rend[X] - rbeg[X] ) +
                               ( rend[Y] - rbeg[Y] ) * ( rend[Y] - rbeg[Y] ) +
                               ( rend[Z] - rbeg[Z] ) * ( rend[Z] - rbeg[Z] )   );

            wgt[kmeas] = traj->weight[kcamera][k];

            //........ Time of the measurement can be relative between cameras for this LMS fit, 
			//         therefore no need for tzero or time offset adjustments

            tim[kmeas] = traj->meashat_ECI[kcamera][k][T];

            //printf(" x t  %lf  %lf\n", pos[kmeas], tim[kmeas] );

            kmeas++;

        }  // end of measurement loop per camera

    }  //... end of camera loop


    //======== LMS solution for velocity and starting positions per camera

    ConstantVelocity_MultiTrackFit( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocityLMS, xo, &rmserror, err );


    //======== Perform outlier removal set at 2x the rmserror

    kmeas      = 0;
    kmeas_kept = 0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ ) {

        nmeas_kept = 0;

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            if( fabs( err[kmeas] ) <= 2.0 * rmserror )  {

                pos[kmeas_kept] = pos[kmeas];
                tim[kmeas_kept] = tim[kmeas];
                wgt[kmeas_kept] = wgt[kmeas];
                kmeas_kept++;
                nmeas_kept++;

            }

            kmeas++;

        }  // end of measurement loop per camera

        nmeas_per_camera[kcamera] = nmeas_kept;

    }  //... end of camera loop


    //======== LMS solution for velocity with outliers removed

    ConstantVelocity_MultiTrackFit( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocityLMS, xo, &rmserror, err );


	//======== Compute the time offsets from the fit starting positions

	if( *velocityLMS <= 0.0 )  *velocityLMS = 42.0;  //... km/sec

	for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  tshift[kcamera] = xo[kcamera] / *velocityLMS; 


    //======== Free memory and return LMS velocity estimate

    free( pos );
    free( tim );
    free( wgt );
    free( err );
    free( xo  );
    free( nmeas_per_camera );


}


//##################################################################################
//
//======== Function to estimate the velocity and deceleration terms from a set of 
//         positions and times along the radiant line by using an LMS fit to multiple
//         measurement sequences. The RMS error is used on a second pass to remove 
//         outliers in the final LMS fit. Requires an estimate of the timing offsets 
//         on entering this function. 
//==================================================================================

void   VelDecelFit_LMS( struct trajectory_info *traj, double velocity_input, double *velocityLMS, double *decel1LMS, double *decel2LMS )
{
int       kcamera, kmeas, ntotalmeas, k, kbeg, kend, kmeas_kept, nmeas_kept;
int      *nmeas_per_camera;
double   *pos, *tim, *wgt, rmserror, *err;
double    rbeg[3], rend[3], rdummy[3], distance;



    //======== Allocate memory for the positional measurements and time stamps actually used,
    //         plus the measurement count and starting position per camera

    ntotalmeas = 0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  ntotalmeas += traj->nummeas[kcamera];

    pos           = (double*) malloc( ntotalmeas       * sizeof(double) );
    tim           = (double*) malloc( ntotalmeas       * sizeof(double) );
    wgt           = (double*) malloc( ntotalmeas       * sizeof(double) );
    err           = (double*) malloc( ntotalmeas       * sizeof(double) );
    nmeas_per_camera = (int*) malloc( traj->numcameras * sizeof(int)    );


    //======== Infill the positional and temporal measurements making a single vector
    //         of concatenated measurements for all the cameras.

    kmeas = 0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ ) {

        //-------- Avoid first and last measurement

        kbeg = 1;
        kend = traj->nummeas[kcamera] / 2;

        nmeas_per_camera[kcamera] = kend - kbeg + 1;


        //-------- Set reference position to estimated begin point on the radiant line

        rbeg[X] = traj->params[0];
        rbeg[Y] = traj->params[1];
        rbeg[Z] = traj->params[2];


        //-------- Compute each positional and temporal measurement

        for( k=kbeg; k<=kend; k++ )  {

            //........ Get an absolute ECI position along the measurement ray at its CPA to the radiant line

            TwoLineCPA( traj->rcamera_ECI[kcamera][k], traj->meashat_ECI[kcamera][k], rend,
                        &traj->params[0], &traj->params[3], rdummy, 
						&distance );

            //........ Compute the distance from the reference point along the line to the measurement CPA

            pos[kmeas] = sqrt( ( rend[X] - rbeg[X] ) * ( rend[X] - rbeg[X] ) +
                               ( rend[Y] - rbeg[Y] ) * ( rend[Y] - rbeg[Y] ) +
                               ( rend[Z] - rbeg[Z] ) * ( rend[Z] - rbeg[Z] )   );

            wgt[kmeas] = traj->weight[kcamera][k];

            //........ Time of the measurement can be relative between cameras for this LMS fit, 
			//         so time offset adjustments relative to first

            tim[kmeas] = traj->meashat_ECI[kcamera][k][T] + traj->tref_offsets[kcamera];

            kmeas++;

        }  // end of measurement loop per camera

    }  //... end of camera loop


    //======== LMS solutions for non-constant velocity models

    if(      traj->velmodel == LINEAR    )     LinearVelocity_MultiTrackFit( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocityLMS,    decel1LMS, decel2LMS, &rmserror, err );
	else if( traj->velmodel == QUADRATIC )  QuadraticVelocity_MultiTrackFit( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocityLMS,    decel1LMS, decel2LMS, &rmserror, err );
	//else if( traj->velmodel == EXPONENT  )   ExponentVelocity_MultiTrackFit( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocity_input, velocityLMS, decel1LMS, decel2LMS, &rmserror, err );
	else if( traj->velmodel == EXPONENT  )   ExponentVelocity_MultiTrackApprox( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocityLMS, decel1LMS, decel2LMS, &rmserror, err );
	else  {
		printf("====> ERROR in function VelDecelFit_LMS of trajectory\n");
		printf("            Velocity model %d not implemented phase 1\n", traj->velmodel );
	}


    //======== Perform outlier removal, set at 2x the rmserror

    kmeas      = 0;
    kmeas_kept = 0;

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ ) {

        nmeas_kept = 0;

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            if( fabs( err[kmeas] ) <= 2.0 * rmserror )  {

                pos[kmeas_kept] = pos[kmeas];
                tim[kmeas_kept] = tim[kmeas];
                wgt[kmeas_kept] = wgt[kmeas];
                kmeas_kept++;
                nmeas_kept++;

            }

            kmeas++;

        }  // end of measurement loop per camera

        nmeas_per_camera[kcamera] = nmeas_kept;

    }  //... end of camera loop


    //======== LMS solution for velocity with outliers removed

    if(      traj->velmodel == LINEAR    )     LinearVelocity_MultiTrackFit( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocityLMS,    decel1LMS, decel2LMS, &rmserror, err );
	else if( traj->velmodel == QUADRATIC )  QuadraticVelocity_MultiTrackFit( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocityLMS,    decel1LMS, decel2LMS, &rmserror, err );
	//else if( traj->velmodel == EXPONENT  )   ExponentVelocity_MultiTrackFit( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocity_input, velocityLMS, decel1LMS, decel2LMS, &rmserror, err );
	else if( traj->velmodel == EXPONENT  )   ExponentVelocity_MultiTrackApprox( traj->numcameras, nmeas_per_camera, pos, tim, wgt, velocityLMS, decel1LMS, decel2LMS, &rmserror, err );
	else  {
		printf("====> ERROR in function VelDecelFit_LMS of trajectory\n");
		printf("            Velocity model %d not implemented phase 2\n", traj->velmodel );
	}


    //======== Free memory and return LMS velocity estimate

    free( pos );
    free( tim );
    free( wgt );
    free( err );
    free( nmeas_per_camera );


}


//############################################################################################

//======== This function computes a weighted LMS fit to velocity and starting positions of
//         sequences of position and time measurements obtained from several independent
//         tracks. Assumes all the tracks have the same constant motion velocity "velocityLMS"
//         but that each track is not synchronized in any way to another track sequence in
//         time or starting position "xo". The input data is comprised of a set of concatenated
//         track measurements in vectors for position "pos", time "tim", and weight "wgt"
//         which is the inverse of the variance per measurement.
//
//         Thus the user inputs the number of sequences or tracks "ncameras" and the number of
//         measurements for each camera "nmeas_per_camera" which does NOT have to be the same
//         measurement count per camera.
//============================================================================================

void  ConstantVelocity_MultiTrackFit( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt, double *velocityLMS, double *xo, double *rmserror, double *err )
{
int     kcamera, kmeas, k;
double  sumw, sumwx, sumwt, sumwxt, sumwtt, sumwtsumwtsumw, sumwxsumwtsumw, sumesq;



	//======== Compute sums for the LMS solution

    sumwxt         = 0.0;
    sumwtt         = 0.0;
    sumwtsumwtsumw = 0.0;
    sumwxsumwtsumw = 0.0;

    kmeas = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        sumw  = 0.0;
        sumwx = 0.0;
        sumwt = 0.0;

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            sumw   += wgt[kmeas];
            sumwx  += wgt[kmeas] * pos[kmeas];
            sumwt  += wgt[kmeas] * tim[kmeas];

            sumwxt += wgt[kmeas] * pos[kmeas] * tim[kmeas];
            sumwtt += wgt[kmeas] * tim[kmeas] * tim[kmeas];

            kmeas++;

        }  // end of measurement loop per camera

        sumwtsumwtsumw += sumwt * sumwt / sumw;
        sumwxsumwtsumw += sumwx * sumwt / sumw;

    }  //... end of camera loop


    //======== LMS solve for the velocity

    *velocityLMS = ( sumwxt - sumwxsumwtsumw ) / ( sumwtt - sumwtsumwtsumw );


    //======== LMS solve for the starting position

    kmeas = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        sumw  = 0.0;
        sumwx = 0.0;
        sumwt = 0.0;

        if( nmeas_per_camera[kcamera] > 0 )  {

            for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

                sumw  += wgt[kmeas];
                sumwx += wgt[kmeas] * pos[kmeas];
                sumwt += wgt[kmeas] * tim[kmeas];

                kmeas++;

            }  // end of measurement loop per camera

            xo[kcamera] = ( sumwx - *velocityLMS * sumwt ) / sumw;

        }

        else  xo[kcamera] = 0.0;

    }  //... end of camera loop


    //======== Determine the root-mean-square error

    sumesq = 0.0;

    kmeas  = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            err[kmeas] = pos[kmeas] - xo[kcamera] - *velocityLMS * tim[kmeas];

            sumesq += err[kmeas] * err[kmeas];

            kmeas++;

        }

    }

    *rmserror = sqrt( sumesq / (double)kmeas );


}

//############################################################################################

//======== This function computes a weighted LMS fit to velocity and deceleration terms from
//         sequences of position and time measurements obtained from several independent
//         tracks. Assumes all the tracks have the same velocity and deceleration and the
//         model chosen is LINEAR. The input data is comprised of a set of concatenated
//         track measurements in vectors for position "pos", time "tim", and weight "wgt"
//         which is a normalized inverse of the variance per measurement.
//
//         The time MUST include timing offsets between cameras.
//
//         If a deceleration coefficient that should be greater than zero is found
//         to be a negative value, it is set to zero rather than try to do a
//         constrained LMS for positive coefficients (since this is a first guess
//         function it should be OK to initialize the solution with zero deceleration).
//
//         Thus the user inputs the number of sequences or tracks "ncameras" and the number of
//         measurements for each camera "nmeas_per_camera" which does NOT have to be the same
//         measurement count per camera.
//============================================================================================

void  LinearVelocity_MultiTrackFit( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt, double *velocityLMS, double *decel1LMS, double *decel2LMS, double *rmserror, double *err )
{
int     kcamera, kmeas, k;
double  sumwt2, sumwt3, sumwt4, sumwxt1, sumwxt2, sumesq;
double  matinv[2][2], denom, alpha;
double  vel, vel_const, xo_const, rmserr_const;




    //======== Compute sums for the LMS solution

    sumwt2         = 0.0;
    sumwt3         = 0.0;
    sumwt4         = 0.0;
    sumwxt1        = 0.0;
    sumwxt2        = 0.0;

    kmeas = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            sumwt2  += wgt[kmeas] * tim[kmeas] * tim[kmeas];
            sumwt3  -= wgt[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];
            sumwt4  += wgt[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];

            sumwxt1 += wgt[kmeas] * pos[kmeas] * tim[kmeas];
            sumwxt2 -= wgt[kmeas] * pos[kmeas] * tim[kmeas] * tim[kmeas];

            kmeas++;

        }  // end of measurement loop for the given camera

    }  //... end of camera loop


    //======== LMS solve for the velocity and deceleration terms via 3x3 matrix inverse

	matinv[0][0] = +sumwt4;
	matinv[1][0] = -sumwt3;

	matinv[0][1] = -sumwt3;
	matinv[1][1] = +sumwt2;

	denom = sumwt2 * sumwt4 - sumwt3 * sumwt3;

	vel   = ( sumwxt1 * matinv[0][0] + sumwxt2 * matinv[0][1] ) / denom;

    alpha = ( sumwxt1 * matinv[1][0] + sumwxt2 * matinv[1][1] ) / denom;


	if( alpha > 0.0 )  {

        *velocityLMS = vel;

	    *decel1LMS = 2.0 * alpha;

	    *decel2LMS = 0.0;

	}
	else  {

		ConstantVelocity_MultiTrackFit( ncameras, nmeas_per_camera, pos, tim, wgt, &vel_const, &xo_const, &rmserr_const, err );

        *velocityLMS = vel_const;

	    *decel1LMS = 0.0;

	    *decel2LMS = 0.0;

	}


    //======== Determine the root-mean-square error

    sumesq = 0.0;

    kmeas  = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            err[kmeas] = pos[kmeas] - *velocityLMS * tim[kmeas] 
			                        +   *decel1LMS * tim[kmeas] * tim[kmeas] / 2.0;

            sumesq += err[kmeas] * err[kmeas];

            kmeas++;

        }

    }

    *rmserror = sqrt( sumesq / (double)kmeas );


}

//############################################################################################

//======== This function computes a weighted LMS fit to velocity and deceleration terms from
//         sequences of position and time measurements obtained from several independent
//         tracks. Assumes all the tracks have the same velocity and deceleration and the
//         model chosen is QUADRATIC. The input data is comprised of a set of concatenated
//         track measurements in vectors for position "pos", time "tim", and weight "wgt"
//         which is a normalized inverse of the variance per measurement.
//
//         The time MUST include timing offsets between cameras.
//
//         If the t^2 deceleration coefficient that should be greater than zero is found
//         to be a negative value, all deceleration is set to zero rather than try to do a
//         constrained LMS for positive coefficients (since this is a first guess
//         function it should be OK to initialize the solution with zero deceleration).
//
//         Thus the user inputs the number of sequences or tracks "ncameras" and the number of
//         measurements for each camera "nmeas_per_camera" which does NOT have to be the same
//         measurement count per camera.
//============================================================================================

void  QuadraticVelocity_MultiTrackFit( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt, double *velocityLMS, double *decel1LMS, double *decel2LMS, double *rmserror, double *err )
{
int     kcamera, kmeas, k;
double  sumwt2, sumwt3, sumwt4, sumwt5, sumwt6, sumwxt1, sumwxt2, sumwxt3, sumesq;
double  matinv[3][3], denom, alpha, beta;
double  vel, vel_const, xo_const, rmserr_const;



	//======== Compute sums for the LMS solution

    sumwt2         = 0.0;
    sumwt3         = 0.0;
    sumwt4         = 0.0;
    sumwt5         = 0.0;
    sumwt6         = 0.0;
    sumwxt1        = 0.0;
    sumwxt2        = 0.0;
    sumwxt3        = 0.0;

    kmeas = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            sumwt2  += wgt[kmeas] * tim[kmeas] * tim[kmeas];
            sumwt3  -= wgt[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];
            sumwt4  += wgt[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];
            sumwt5  -= wgt[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];
            sumwt6  += wgt[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];

            sumwxt1 += wgt[kmeas] * pos[kmeas] * tim[kmeas];
            sumwxt2 -= wgt[kmeas] * pos[kmeas] * tim[kmeas] * tim[kmeas];
            sumwxt3 += wgt[kmeas] * pos[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];

            kmeas++;

        }  // end of measurement loop for the given camera

    }  //... end of camera loop


    //======== LMS solve for the velocity and deceleration terms via 3x3 matrix inverse

	matinv[0][0] = sumwt4 * sumwt6 - sumwt5 * sumwt5;
	matinv[1][0] = sumwt5 * sumwt4 - sumwt3 * sumwt6;
	matinv[2][0] = sumwt3 * sumwt5 - sumwt4 * sumwt4;

	matinv[0][1] = sumwt4 * sumwt5 - sumwt3 * sumwt6;
	matinv[1][1] = sumwt2 * sumwt6 - sumwt4 * sumwt4;
	matinv[2][1] = sumwt3 * sumwt4 - sumwt2 * sumwt5;

	matinv[0][2] = sumwt3 * sumwt5 - sumwt4 * sumwt4;
	matinv[1][2] = sumwt4 * sumwt3 - sumwt2 * sumwt5;
	matinv[2][2] = sumwt2 * sumwt4 - sumwt3 * sumwt3;

	denom = sumwt2 * matinv[0][0] + sumwt3 * matinv[1][0] + sumwt4 * matinv[2][0];


    vel   = ( sumwxt1 * matinv[0][0] + sumwxt2 * matinv[0][1] + sumwxt3 * matinv[0][2] ) / denom;

    alpha = ( sumwxt1 * matinv[1][0] + sumwxt2 * matinv[1][1] + sumwxt3 * matinv[1][2] ) / denom;

    beta  = ( sumwxt1 * matinv[2][0] + sumwxt2 * matinv[2][1] + sumwxt3 * matinv[2][2] ) / denom;

	
	if( alpha > 0.0 )  {

        *velocityLMS = vel;

	    *decel1LMS = 2.0 * alpha;

	    *decel2LMS = 3.0 * beta;

	}
	else  {

		ConstantVelocity_MultiTrackFit( ncameras, nmeas_per_camera, pos, tim, wgt, &vel_const, &xo_const, &rmserr_const, err );

        *velocityLMS = vel_const;

	    *decel1LMS = 0.0;

	    *decel2LMS = 0.0;

	}


    //======== Determine the root-mean-square error

    sumesq = 0.0;

    kmeas  = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            err[kmeas] = pos[kmeas] - *velocityLMS * tim[kmeas] 
			                        +   *decel1LMS * tim[kmeas] * tim[kmeas] / 2.0
								    -   *decel2LMS * tim[kmeas] * tim[kmeas] * tim[kmeas] / 3.0;

            sumesq += err[kmeas] * err[kmeas];

            kmeas++;

        }

    }

    *rmserror = sqrt( sumesq / (double)kmeas );


}

//############################################################################################

//======== This function computes a weighted LMS fit to velocity and deceleration terms from
//         sequences of position and time measurements obtained from several independent
//         tracks. Assumes all the tracks have the same velocity and deceleration and the
//         model chosen is EXPONENT. The input data is comprised of a set of concatenated
//         track measurements in vectors for position "pos", time "tim", and weight "wgt"
//         which is a normalized inverse of the variance per measurement.
//
//         The time MUST include timing offsets between cameras.
//
//         This is an approximation because the exponential has been expanded to just 3 terms
//         such that exp(ax) ~ 1 + ax + (ax)^2/2
//
//         If a deceleration coefficient that should be greater than zero is found
//         to be a negative value, it is set to zero rather than try to do a
//         constrained LMS for positive coefficients (since this is a first guess
//         function it should be OK to initialize the solution with zero deceleration).
//
//         Thus the user inputs the number of sequences or tracks "ncameras" and the number of
//         measurements for each camera "nmeas_per_camera" which does NOT have to be the same
//         measurement count per camera.
//============================================================================================

void  ExponentVelocity_MultiTrackApprox( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt, double *velocityLMS, double *decel1LMS, double *decel2LMS, double *rmserror, double *err )
{
int     kcamera, kmeas, k;
double  sumwt0, sumwt1, sumwt2, sumwt3, sumwt4, sumwxt0, sumwxt1, sumwxt2, sumesq;
double  matinv[3][3], denom, alpha, beta, eta;
double  vel_const, xo_const, rmserr_const;


//======== Compute sums for the LMS solution

    sumwt0         = 0.0;
    sumwt1         = 0.0;
    sumwt2         = 0.0;
    sumwt3         = 0.0;
    sumwt4         = 0.0;
    sumwxt0        = 0.0;
    sumwxt1        = 0.0;
    sumwxt2        = 0.0;

    kmeas = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            sumwt0  += wgt[kmeas];
            sumwt1  -= wgt[kmeas] * tim[kmeas];
            sumwt2  += wgt[kmeas] * tim[kmeas] * tim[kmeas];
            sumwt3  -= wgt[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];
            sumwt4  += wgt[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas] * tim[kmeas];

            sumwxt0 -= wgt[kmeas] * pos[kmeas];
            sumwxt1 += wgt[kmeas] * pos[kmeas] * tim[kmeas];
            sumwxt2 -= wgt[kmeas] * pos[kmeas] * tim[kmeas] * tim[kmeas];

            kmeas++;

        }  // end of measurement loop for the given camera

    }  //... end of camera loop


    //======== LMS solve for the velocity and deceleration terms via 3x3 matrix inverse

	matinv[0][0] = sumwt2 * sumwt4 - sumwt3 * sumwt3;
	matinv[1][0] = sumwt3 * sumwt2 - sumwt1 * sumwt4;
	matinv[2][0] = sumwt1 * sumwt3 - sumwt2 * sumwt2;

	matinv[0][1] = sumwt2 * sumwt3 - sumwt1 * sumwt4;
	matinv[1][1] = sumwt0 * sumwt4 - sumwt2 * sumwt2;
	matinv[2][1] = sumwt1 * sumwt2 - sumwt0 * sumwt3;

	matinv[0][2] = sumwt1 * sumwt3 - sumwt2 * sumwt2;
	matinv[1][2] = sumwt2 * sumwt1 - sumwt0 * sumwt3;
	matinv[2][2] = sumwt0 * sumwt2 - sumwt1 * sumwt1;

	denom = sumwt0 * matinv[0][0] + sumwt1 * matinv[1][0] + sumwt2 * matinv[2][0];


    alpha = ( sumwxt0 * matinv[0][0] + sumwxt1 * matinv[0][1] + sumwxt2 * matinv[0][2] ) / denom;

    eta   = ( sumwxt0 * matinv[1][0] + sumwxt1 * matinv[1][1] + sumwxt2 * matinv[1][2] ) / denom;

    beta  = ( sumwxt0 * matinv[2][0] + sumwxt1 * matinv[2][1] + sumwxt2 * matinv[2][2] ) / denom;


	if( alpha > 0.0  &&  beta >= 0.0 )  {

	    *decel1LMS = alpha;

	    *decel2LMS = sqrt( 2.0 * beta / alpha );

        *velocityLMS = eta - *decel1LMS * *decel2LMS;

	}
	else  {

		ConstantVelocity_MultiTrackFit( ncameras, nmeas_per_camera, pos, tim, wgt, &vel_const, &xo_const, &rmserr_const, err );

        *velocityLMS = vel_const;

	    *decel1LMS = 0.0;

	    *decel2LMS = 0.0;

	}


    //======== Determine the root-mean-square error

    sumesq = 0.0;

    kmeas  = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            err[kmeas] = pos[kmeas] - *velocityLMS * tim[kmeas]  
			                        +   *decel1LMS * exp( *decel2LMS * tim[kmeas] );


            sumesq += err[kmeas] * err[kmeas];

            kmeas++;

        }

    }

    *rmserror = sqrt( sumesq / (double)kmeas );

}


//############################################################################################

//======== This function computes a weighted LMS fit to only the deceleration terms from
//         sequences of position and time measurements obtained from several independent
//         tracks. Assumes all the tracks have the same velocity and deceleration and the
//         model chosen is EXPONENT. The input data is comprised of a set of concatenated
//         track measurements in vectors for position "pos", time "tim", and weight "wgt"
//         which is a normalized inverse of the variance per measurement.
//
//         The time MUST include timing offsets between cameras.
//
//         The velocity is presumed known a priori.
//
//         If a deceleration coefficient that should be greater than zero is found
//         to be a negative value, it is set to zero rather than try to do a
//         constrained LMS for positive coefficients (since this is a first guess
//         function it should be OK to initialize the solution with zero deceleration).
//
//         Thus the user inputs the number of sequences or tracks "ncameras" and the number of
//         measurements for each camera "nmeas_per_camera" which does NOT have to be the same
//         measurement count per camera.
//============================================================================================

void  ExponentVelocity_MultiTrackFit( int ncameras, int *nmeas_per_camera, double *pos, double *tim, double *wgt, double velocity_input, double *velocityLMS, double *decel1LMS, double *decel2LMS, double *rmserror, double *err )
{
int     kcamera, kmeas, k;
double  sumw, sumwe, sumwex, sumwet, sumwext, sumesq, expa2t, posvel; 
double  diff, diff_last, diff_best;
double  decel2, decel2_start, decel2_step, decel2_stop, decel2_best, decel1_best;


	//======== Loop over positive only decel2 values

    decel2_start =  0.000;
	decel2_step  =  0.001;
	decel2_stop  = 30.0001;

    diff_best = 1.0e+30;

    for( decel2=decel2_start; decel2<=decel2_stop; decel2+=decel2_step )  {

        //======== Compute sums given the decel2 coefficient

		sumw     = 0.0;
        sumwe    = 0.0;
        sumwet   = 0.0;
        sumwex   = 0.0;
        sumwext  = 0.0;

        kmeas = 0;

        for( kcamera=0; kcamera<ncameras; kcamera++ ) {

            for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

				expa2t = exp( decel2 * tim[kmeas] );

				posvel = pos[kmeas] - velocity_input * tim[kmeas];

		        sumw    += wgt[kmeas];
                sumwe   += wgt[kmeas] * expa2t * expa2t;
                sumwet  += wgt[kmeas] * expa2t * expa2t * tim[kmeas];
                sumwex  -= wgt[kmeas] * expa2t * posvel;
                sumwext -= wgt[kmeas] * expa2t * posvel * tim[kmeas];

                kmeas++;

            }  // end of measurement loop for the given camera

        }  //... end of camera loop

		sumwe   /= sumw;
		sumwet  /= sumw;
		sumwex  /= sumw;
		sumwext /= sumw;


		diff = sumwext * sumwe - sumwex * sumwet;

		if( decel2 == decel2_start )  diff_last = diff;

		if( fabs(diff) < diff_best )  {
			diff_best   = fabs(diff);
			decel1_best = sumwex / sumwe;
			decel2_best = decel2;
			////printf(" %lf  %le  %le  %le  %le\n", decel1_best, decel2, diff, sumwe);
		}

		if( ( diff <= 0.0  &&  diff_last >= 0.0 )  ||
			( diff >= 0.0  &&  diff_last <= 0.0 )      )  break;

		diff_last = diff;

	}  //... end of decel2 loop


	//======== Set the returns values for the decelerations

	if( decel1_best < 0.0 )  decel1_best = 0.0;
	if( decel2_best < 0.0 )  decel2_best = 0.0;

	*decel1LMS = decel1_best;
	*decel2LMS = decel2_best;

	*velocityLMS = velocity_input;


    //======== Determine the root-mean-square error

    sumesq = 0.0;

    kmeas  = 0;

    for( kcamera=0; kcamera<ncameras; kcamera++ ) {

        for( k=0; k<nmeas_per_camera[kcamera]; k++ )  {

            err[kmeas] = pos[kmeas] - velocity_input * tim[kmeas]  
			                        + *decel1LMS * exp( *decel2LMS * tim[kmeas] );


            sumesq += err[kmeas] * err[kmeas];

            kmeas++;

        }

    }

    *rmserror = sqrt( sumesq / (double)kmeas );


}


//##################################################################################################
//##################################################################################
//
//======== Function used to refine the estimation parameters by calling the PSO
//         minimization module. Four levels of accuracy are available based on need:
//         1) a low quality "quick fit" for performing many Monte Carlo solutions 
//            given the no added noise best solution as starting guess each time. 
//         2) a "good" quality solution for bootstrapping to the final best solution
//            without a large runtime cost.
//         3) a high quality "accurate" solution for best solution iteration using
//            large number of particles and tighter convergence criteria
//         4) an "extremely" heavy particle count in the PSO and even tighter 
//            constraints on convergence
//         Input is the starting guess of parameters passed in via traj->xguess 
//         with range limits in traj->xshift, with the output solution saved off
//         to traj->params.
//=================================================================================


void   ParameterRefinementViaPSO( struct trajectory_info *traj, int cost_function, int velocity_model, int pso_fit_control )
{
int     k;
double  cost_func_value;

struct particleswarming  pso;

struct PSO_info          trajpsofit;


    //======== Particle Swarm Optimizer parameters settings

    trajpsofit = traj->PSOfit[pso_fit_control];


    //======== Pre-populate the PSO and initialize with guesses and shifts (zero shifts for fixed parameters)

    ParticleSwarm_PrePopulate( trajpsofit.number_particles, 
		                       traj->numparams, 
							   trajpsofit.maximum_iterations,
                               trajpsofit.boundary_flag,  
							   trajpsofit.limits_flag,
							   trajpsofit.particle_distribution_flag, 
							   trajpsofit.epsilon_convergence,
                               trajpsofit.weight_inertia, 
							   trajpsofit.weight_stubborness, 
							   trajpsofit.weight_grouppressure,
                               &pso );

    ParticleSwarm_Initialize( traj->xguess, traj->xshift, &pso );


    //======== PSO minimization loop

    while( pso.processing == CONTINUE_PROCESSING )  {

        if(      cost_function == MEAS2LINE  )  cost_func_value = Anglesum_Measurements2Line(  &pso.xtest[0],
                                                                                               &pso.xtest[3],
                                                                                                traj );

        else if( cost_function == MEAS2MODEL )  cost_func_value = Anglesum_Measurements2Model(  velocity_model,
                                                                                               &pso.xtest[0],
                                                                                               &pso.xtest[3],
                                                                                                pso.xtest[6],
                                                                                                pso.xtest[7],
																							    pso.xtest[8],
                                                                                               &pso.xtest[9],
                                                                                                traj );

        else  {
            printf("ERROR--> cost_function %d not implemented in ParameterRefinementViaPSO\n", cost_function );
			Delay_msec(15000);
            exit(1);
        }


        ParticleSwarm_Update( cost_func_value, &pso );

    }

    ////printf(" Halt Condition %d,  Number of iterations %d\n", pso.processing, pso.niteration );


    //======== Assign refined parameters, cleanup memory, and return

    for( k=0; k<traj->numparams; k++ )  traj->params[k] = pso.gbest[k];

    ParticleSwarm_PostCleanup( &pso );

}


//##################################################################################
//
//======== Function that slides the timing offsets to be sure we did not get caught
//         in a local minimum. Looks at the cost function result by working on each
//         camera independently.
//=================================================================================


int   TimeOffsetRefinement( struct trajectory_info *traj )
{
int     offsets_shifted, k, kt, ktbest, kcamera, nmeas;
double  dt, tbase, tshift, cfv, cfvbest;


    offsets_shifted = 0;

	for( k=0; k<traj->numparams; k++ )  traj->xguess[k] = traj->params[k];


	//======== Check each camera independently (skip the first since it is always zero offset)

    for( kcamera=1; kcamera<traj->numcameras; kcamera++ )  {

		//-------- Get the average time spacing between measurements for this camera

		nmeas = traj->nummeas[kcamera];

		dt = ( traj->meashat_ECI[kcamera][nmeas-1][T] - traj->meashat_ECI[kcamera][0][T] ) / (double)(nmeas-1);


		//-------- Find the minimum of the cost function

		tbase = traj->xguess[9+kcamera];

		ktbest  = 0;
		cfvbest = 1.0e+30;

		for( kt=-3500; kt<=+3500; kt++ )  {  //... range is -3.5*dt to +3.5*dt

			tshift = (double)kt * dt / 1000.0;

			traj->xguess[9+kcamera] = tbase + tshift;

			cfv = Anglesum_Measurements2Model( traj->velmodel, &traj->xguess[0],
                                                               &traj->xguess[3],
                                                                traj->xguess[6],
                                                                traj->xguess[7],
																traj->xguess[8],
                                                               &traj->xguess[9],
                                                                traj );

			if( cfv < cfvbest )  {
				cfvbest = cfv;
				ktbest  = kt;
			}

		} // end of time shift test loop


		//-------- reset to original value before trying other cameras

		traj->xguess[9+kcamera] = tbase;


		//-------- Did we find a better time offset for this camera ?

		if( ktbest != 0 )  {  

			tshift = (double)ktbest * dt / 100.0;

			traj->params[9+kcamera] = tbase + tshift;

			offsets_shifted = 1;

		}

	}  //... end of camera loop


	return( offsets_shifted );

}

	
//##################################################################################
//
//======== Function to sum the angles between the radiant line and the measurement
//         rays to provide a cost function value to minimize against. Used for
//         Borovicka's least squares radiant solution, but replaced Jiri's distance
//         sum with an angle sum, for the cost function that is being minimized.
//         The cost function is weighted by the inverse of the noise variance.
//==================================================================================


double  Anglesum_Measurements2Line( double *ro_ECI, double *vo_ECI, struct trajectory_info *traj )
{
int     kcamera, kmeas;
double  fsum, cosangle, r[3], rmeas[3], radiant_hat_ECI[3], rmagnitude, cpa_distance, wsum, weight;



    fsum = 0.0;
    wsum = 0.0;

    VectorNorm( vo_ECI, radiant_hat_ECI, &rmagnitude );  //... Unit vector for the radiant direction

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  {

            //-------- Find the ECI vector to point "r" on the radiant line closest to the the measurement ray

            TwoLineCPA( traj->rcamera_ECI[kcamera][kmeas], traj->meashat_ECI[kcamera][kmeas], rmeas,
                        ro_ECI, radiant_hat_ECI, r,
                        &cpa_distance );

            //-------- The weighting of 1/variance is based on the per measurment noise standard deviation

            weight = traj->weight[kcamera][kmeas];

            wsum += weight;

            //-------- CPA ray's unit vector to the point on the line from the camera position

            r[X] = r[X] - traj->rcamera_ECI[kcamera][kmeas][X];
            r[Y] = r[Y] - traj->rcamera_ECI[kcamera][kmeas][Y];
            r[Z] = r[Z] - traj->rcamera_ECI[kcamera][kmeas][Z];

            VectorNorm( r, r, &rmagnitude );

            //-------- Angle between measurement ray and the CPA ray as seen from the camera

            cosangle = VectorDotProduct( traj->meashat_ECI[kcamera][kmeas], r );

            if( cosangle > +1.0 )  cosangle = +1.0;
            if( cosangle < -1.0 )  cosangle = -1.0;

            fsum += weight * acos( cosangle );
        }

    }

    fsum /= wsum;

    return( fsum );

}

//##################################################################################
//
//======== Function to sum the angles between the motion MODEL positions and the
//         MEASUREMENT rays to provide a cost function value to minimize against.
//         The cost function is weighted by the inverse of the noise variance.
//==================================================================================

double  Anglesum_Measurements2Model( int velmodel,  double *ro_ECI, double *vo_ECI,
                                     double decel1, double decel2,  double ttzero,  double *toffset,
                                     struct trajectory_info *traj )
{
int     kcamera, kmeas;
double  fsum, tt, length_km, gravity_km;
double  g, ro_lat, ro_LST, ro_hkm;
double  r[3], radiant_hat_ECI[3], gravity_hat_ECI[3];
double  vbegin, rmagnitude, cosangle, weight, wsum;



    fsum = 0.0;
    wsum = 0.0;

    VectorNorm( vo_ECI, radiant_hat_ECI, &vbegin );         //... radiant direction

    VectorNorm( ro_ECI, gravity_hat_ECI, &rmagnitude );     //... anti-gravity direction

	ECEF2LatLonAlt( ro_ECI, &ro_lat, &ro_LST, &ro_hkm );    //... height of meteor above surface

	g = g_surface * ( 6378.137 / ( 6378.137 + ro_hkm ) );  //... gravity km/sec^2 corrected for height



    for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

        for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  {

            //-------- Compute the model time from the measurement time plus the camera time offset

            tt = traj->meashat_ECI[kcamera][kmeas][T] + toffset[kcamera];

            //-------- The weighting of 1/variance is based on the per measurment noise standard deviation

            weight = traj->weight[kcamera][kmeas];

            wsum += weight;

            //-------- Propagation distance for the given time (the propagation model must
			//             handle the tzero reference point)

            length_km = Propagation( POSITION, velmodel, tt, ttzero, vbegin, decel1, decel2 );


			//-------- Propagation towards earth center due to gravity acceleration

			gravity_km = 0.5 * g * tt * tt;


            //-------- Model ray's unit vector from the camera position

            r[X] = ro_ECI[X] - traj->rcamera_ECI[kcamera][kmeas][X] - radiant_hat_ECI[X] * length_km - gravity_hat_ECI[X] * gravity_km;
            r[Y] = ro_ECI[Y] - traj->rcamera_ECI[kcamera][kmeas][Y] - radiant_hat_ECI[Y] * length_km - gravity_hat_ECI[Y] * gravity_km;
            r[Z] = ro_ECI[Z] - traj->rcamera_ECI[kcamera][kmeas][Z] - radiant_hat_ECI[Z] * length_km - gravity_hat_ECI[Z] * gravity_km;

            VectorNorm( r, r, &rmagnitude );

            //-------- Angle between the measurement ray and model ray as seen from the camera

            cosangle = VectorDotProduct( traj->meashat_ECI[kcamera][kmeas], r );

            if( cosangle > +1.0 )  cosangle = +1.0;
            if( cosangle < -1.0 )  cosangle = -1.0;

            fsum += weight * acos( cosangle );

        }

    }

    fsum /= wsum;

	////printf(" Height = %lf   Gravity g = %lf   Lgrav = %lf m\n", ro_hkm, g, gravity_km*1000.0 );

	return( fsum );

}


//##################################################################################

//======== Assign return LLA parameters for measurement directions at their closest
//         point of approach (CPA) to the radiant line, plus the associated range.
//         Also get the model position along the radiant line and the associated
//         modeled velocity computed ON the radiant line at the TIMES input and
//         adjusted for the timing offsets and NOT directly associated with the
//         measurements. Note that LLA is latitude, longitude, altitude in
//         geodetic WGS84 coordinates for the reference Julian date/time.
//==================================================================================

void   ReportFill_LLAVT_Meas_Model( struct trajectory_info *traj )
{
int     kcamera, kmeas;
double  radiant_hat[3], rdummy[3], r[3], rfit[3], r_lat, r_lon, r_LST, r_hkm;
double  dist, vbegin, tt, length_km, rafit, decfit;
double  azim, elev, sazim, eazim, zenangle, pi;
double  g, gravity_km, gravity_hat[3], ro_lat, ro_LST, ro_hkm, romag;
double  RApersecond, LSTshift;


    pi = 4.0 * atan(1.0);

	RApersecond = 2.0 * pi * 1.00273785 / 86400.0;

    VectorNorm( &traj->solution[3], radiant_hat, &vbegin );

    VectorNorm( &traj->solution[0], gravity_hat, &romag );

	ECEF2LatLonAlt( &traj->solution[0], &ro_lat, &ro_LST, &ro_hkm );  //... height of meteor begin above surface

	g = g_surface * ( 6378.137 / ( 6378.137 + ro_hkm ) );  //... gravity km/sec^2 corrected for height


    //======== Loop over all cameras and their associated measurements

    for( kcamera=0; kcamera<traj->numcameras; kcamera++ ) {

        for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ ) {

             //-------- Get measurement ray positional info at the closest point of approach to the radiant line
			 //            DOES NOT INCLUDE GRAVITATIONAL ACCELERATION CHANGE IN TRACK

             TwoLineCPA( traj->rcamera_ECI[kcamera][kmeas], traj->meashat_ECI[kcamera][kmeas], r,
                         &traj->solution[0], radiant_hat, rdummy, 
						 &dist );

             ECEF2LatLonAlt( r, &r_lat, &r_LST, &r_hkm );

             r_lon = LST2LongitudeEast( traj->jdt_ref, r_LST );

             traj->meas_lat[kcamera][kmeas] = r_lat;
             traj->meas_lon[kcamera][kmeas] = r_lon;
             traj->meas_hkm[kcamera][kmeas] = r_hkm;

             traj->meas_range[kcamera][kmeas] = sqrt( (r[X] - traj->rcamera_ECI[kcamera][kmeas][X]) * (r[X] - traj->rcamera_ECI[kcamera][kmeas][X])
                                                    + (r[Y] - traj->rcamera_ECI[kcamera][kmeas][Y]) * (r[Y] - traj->rcamera_ECI[kcamera][kmeas][Y])
                                                    + (r[Z] - traj->rcamera_ECI[kcamera][kmeas][Z]) * (r[Z] - traj->rcamera_ECI[kcamera][kmeas][Z]) );


             //-------- Compute the measurement time with timing offsets

             tt = traj->meashat_ECI[kcamera][kmeas][T] + traj->solution[9+kcamera];  //... internal time relative to dtime_ref

             traj->model_time[kcamera][kmeas] = tt + traj->dtime_ref;                //... time relative to jdt_ref


             //-------- Get model positional info at each time position along the radiant line

             length_km = Propagation( POSITION, traj->velmodel, tt, traj->ttzero, traj->vbegin, traj->decel1, traj->decel2 );

				 
			 //-------- Propagation towards earth center due to gravity acceleration

	         gravity_km = 0.5 * g * tt * tt;

             r[X] = traj->solution[0] - radiant_hat[X] * length_km - gravity_hat[X] * gravity_km;
             r[Y] = traj->solution[1] - radiant_hat[Y] * length_km - gravity_hat[Y] * gravity_km;
             r[Z] = traj->solution[2] - radiant_hat[Z] * length_km - gravity_hat[Z] * gravity_km;

             ECEF2LatLonAlt( r, &r_lat, &r_LST, &r_hkm );

             r_lon = LST2LongitudeEast( traj->jdt_ref, r_LST );

             traj->model_lat[kcamera][kmeas] = r_lat;
             traj->model_lon[kcamera][kmeas] = r_lon;
             traj->model_hkm[kcamera][kmeas] = r_hkm;

             traj->model_range[kcamera][kmeas] = sqrt( (r[X] - traj->rcamera_ECI[kcamera][kmeas][X]) * (r[X] - traj->rcamera_ECI[kcamera][kmeas][X])
                                                     + (r[Y] - traj->rcamera_ECI[kcamera][kmeas][Y]) * (r[Y] - traj->rcamera_ECI[kcamera][kmeas][Y])
                                                     + (r[Z] - traj->rcamera_ECI[kcamera][kmeas][Z]) * (r[Z] - traj->rcamera_ECI[kcamera][kmeas][Z]) );


             //-------- Compute the model velocity at each time (duplicate this value for the measurement)
			 //               NO ADJUSTMENT MADE FOR GRAVITY INDUCED VELOCITY (~few m/sec)

             traj->model_vel[kcamera][kmeas] = Propagation( VELOCITY, traj->velmodel, tt, traj->ttzero, traj->vbegin, traj->decel1, traj->decel2 );

             traj->meas_vel[kcamera][kmeas] = traj->model_vel[kcamera][kmeas];


             //-------- Compute the model ray unit vector "fit" in ECI and convert to "meastype" for output reporting

             rfit[X] = r[X] - traj->rcamera_ECI[kcamera][kmeas][X];
             rfit[Y] = r[Y] - traj->rcamera_ECI[kcamera][kmeas][Y];
             rfit[Z] = r[Z] - traj->rcamera_ECI[kcamera][kmeas][Z];

             ECI2RahDec( rfit, &rafit, &decfit );

	         LSTshift = RApersecond * tt;

             if( traj->meastype == RODEC )  {
                 traj->model_fit1[kcamera][kmeas] = rafit;
                 traj->model_fit2[kcamera][kmeas] = decfit;
             }

             else if( traj->meastype == RADEC )  {
                 traj->model_fit1[kcamera][kmeas] = rafit - LSTshift; //.. bring back to fixed LST of input data
                 traj->model_fit2[kcamera][kmeas] = decfit;
             }

             else if( traj->meastype == NAZEL )  {
                 RADec2AzimuthElevation( rafit, decfit, traj->camera_lat[kcamera], traj->camera_LST[kcamera] + LSTshift, &azim, &elev );
                 traj->model_fit1[kcamera][kmeas] = azim;
                 traj->model_fit2[kcamera][kmeas] = elev;
             }

             else if( traj->meastype == SAZZA )  {
                 RADec2AzimuthElevation( rafit, decfit, traj->camera_lat[kcamera], traj->camera_LST[kcamera] + LSTshift, &azim, &elev );
                 sazim = azim + pi;
                 if( sazim <  0.0      )  sazim += 2.0 * pi;
                 if( sazim >= 2.0 * pi )  sazim -= 2.0 * pi;
                 zenangle = pi / 2.0 - elev;
                 traj->model_fit1[kcamera][kmeas] = sazim;
                 traj->model_fit2[kcamera][kmeas] = zenangle;
             }

             else if( traj->meastype == EAZZA )  {
                 RADec2AzimuthElevation( rafit, decfit, traj->camera_lat[kcamera], traj->camera_LST[kcamera] + LSTshift, &azim, &elev );
                 eazim = pi / 2.0 - azim;
                 if( eazim <  0.0      )  eazim += 2.0 * pi;
                 if( eazim >= 2.0 * pi )  eazim -= 2.0 * pi;
                 zenangle = pi / 2.0 - elev;
                 traj->model_fit1[kcamera][kmeas] = eazim;
                 traj->model_fit2[kcamera][kmeas] = zenangle;
             }

             else  {
                 printf(" ====> ERROR in ReportFill_LLAVT_Meas_Model: meastype %i not implemented for meas_fit*\n", traj->meastype );
                 Delay_msec(15000);
				 exit(1);
             }


        }  //... end of measurement loop per camera

    }  //... end of camera loop


}

//##################################################################################

//======== Assign return LLA parameters for the BEGIN or END position and the time
//         offset relative to the reference Julian date/time by finding the
//         earliest or latest measurement respectively. This point is on the radiant
//         line as specified by the adjusted time - thus is NOT tied to the
//         measurement unit vector which may be of poor quality because it has
//         low illumination level and inaccurate centroid.
//==================================================================================

void    ReportFill_LLA_Beg_End( int LLA_position, struct trajectory_info *traj )
{
int     kcamera, kcamera1, kmeas;
double  tt, length_km, vbegin;
double  r_lat, r_lon, r_LST, r_hkm, r[3], radiant_hat[3];
double  g, gravity_km, gravity_hat[3], ro_lat, ro_LST, ro_hkm, romag;


    //======== Get the earliest measurement time

    if( LLA_position == LLA_BEG )  {

        tt = +1.0e+20;

        for( kcamera1=0; kcamera1<traj->numcameras; kcamera1++ ) {

            kmeas = 0;

            if( tt > traj->meashat_ECI[kcamera1][kmeas][T] + traj->tref_offsets[kcamera1] )  {
                tt = traj->meashat_ECI[kcamera1][kmeas][T] + traj->tref_offsets[kcamera1];
                kcamera = kcamera1;
            }
        }

        traj->ttbeg = tt;

        kmeas = 0;
    }


    //======== Get the latest measurement time

    if( LLA_position == LLA_END )  {

        tt = -1.0e+20;

        for( kcamera1=0; kcamera1<traj->numcameras; kcamera1++ ) {

            kmeas = traj->nummeas[kcamera1] - 1;

            if( tt < traj->meashat_ECI[kcamera1][kmeas][T] + traj->tref_offsets[kcamera1] )  {
                tt = traj->meashat_ECI[kcamera1][kmeas][T] + traj->tref_offsets[kcamera1];
                kcamera = kcamera1;
            }
        }

        traj->ttend = tt;

        kmeas = traj->nummeas[kcamera] - 1;
    }


    //======== Given the time for the desired measurement point, compute its ECI coords, convert to LLA
	//             NOTE: tt is either the begin or end point time based on LLA_position input argument

    length_km = Propagation( POSITION, traj->velmodel, tt, traj->ttzero, traj->vbegin, traj->decel1, traj->decel2 );

    VectorNorm( &traj->solution[3], radiant_hat, &vbegin );

    VectorNorm( &traj->solution[0], gravity_hat, &romag );

	ECEF2LatLonAlt( &traj->solution[0], &ro_lat, &ro_LST, &ro_hkm );  //... height of meteor begin above surface

	g = g_surface * ( 6378.137 / ( 6378.137 + ro_hkm ) );  //... gravity km/sec^2 corrected for height

	gravity_km = 0.5 * g * tt * tt;

	r[X] = traj->solution[0] - radiant_hat[X] * length_km - gravity_hat[X] * gravity_km;
    r[Y] = traj->solution[1] - radiant_hat[Y] * length_km - gravity_hat[Y] * gravity_km;
    r[Z] = traj->solution[2] - radiant_hat[Z] * length_km - gravity_hat[Z] * gravity_km;

    ECEF2LatLonAlt( r, &r_lat, &r_LST, &r_hkm );

    r_lon = LST2LongitudeEast( traj->jdt_ref, r_LST );


    //======== Assign to output report structure elements

    if( LLA_position == LLA_BEG )  {
        traj->rbeg_lat = r_lat;
        traj->rbeg_lon = r_lon;
        traj->rbeg_hkm = r_hkm;
    }

    if( LLA_position == LLA_END )  {
        traj->rend_lat = r_lat;
        traj->rend_lon = r_lon;
        traj->rend_hkm = r_hkm;
    }


}

//##################################################################################

//======== Function to run a noisy measurement Monte Carlo set of trials to estimate
//         the error in each parameter. Adds noise to the original measurements and
//         starts the solver with the no noise "solution" information to be used as
//         initialization point for each minimization. Returns the standard
//         deviations of various critical parameters.
//===================================================================================

void  MonteCarlo_ErrorEstimate( struct trajectory_info *traj )
{
int     kmonte, kcamera, k;
double  ra, dec, vbegin, length_km, kount;
double  radiant_hat[3], r[3], r_lat, r_lon, r_hkm, r_LST;
double  g, gravity_km, gravity_hat[3], ro_lat, ro_LST, ro_hkm, romag;


    //======== Initialize the standard deviation (squared accumulation) to zero

    traj->ra_sigma       = 0.0;
    traj->dec_sigma      = 0.0;
    traj->vbegin_sigma   = 0.0;
    traj->decel1_sigma   = 0.0;
    traj->decel2_sigma   = 0.0;

    traj->rbeg_hkm_sigma = 0.0;
    traj->rbeg_lat_sigma = 0.0;
    traj->rbeg_lon_sigma = 0.0;

    traj->rend_hkm_sigma = 0.0;
    traj->rend_lat_sigma = 0.0;
    traj->rend_lon_sigma = 0.0;


    //======== Monte Carlo loop that adds measurement noise to find error estimate

    for( kmonte=0; kmonte<traj->nummonte; kmonte++ )  {

        //-------- Retrieve measurement unit vectors again but this time add noise

        for( kcamera=0; kcamera<traj->numcameras; kcamera++ )  {

             Angles2SiteMeasurements( traj->camera_lat[kcamera],
                                      traj->camera_lon[kcamera],
                                      traj->camera_hkm[kcamera],
                                      traj->camera_LST[kcamera],
                                      traj->nummeas[kcamera],
                                      traj->meastype,
                                      traj->meas1[kcamera],
                                      traj->meas2[kcamera],
                                      traj->dtime[kcamera],
                                      traj->noise[kcamera],
                                      ADD_NOISE,
									  traj->dtime_ref,
									  traj->tref_offsets[kcamera],
                                      traj->meashat_ECI[kcamera],
                                      traj->rcamera_ECI[kcamera]  );

        }  //... end of camera loop to add noise to the measurements


        //-------- Find noise added trajectory with the no noise solution as the starting guess

        for( k=0; k<traj->numparams; k++ )  traj->xguess[k] = traj->solution[k];
        for( k=0; k<traj->numparams; k++ )  traj->xshift[k] = traj->limits[k];

        if( traj->velmodel <= 0 )  traj->xshift[6] = 0.0;  //... Constrain decel1 due to velocity model
        if( traj->velmodel <= 1 )  traj->xshift[7] = 0.0;  //... Constrain decel2 due to velocity model
        if( traj->velmodel <= 0 )  traj->xshift[8] = 0.0;  //... Constrain tzero  due to velocity model

        ParameterRefinementViaPSO( traj, MEAS2MODEL, traj->velmodel, traj->PSO_fit_control[5] );   //... call index 5 
		
		                          // --> traj-params[*]


		//-------- Set deceleration terms to correct sign

		traj->params[6] = fabs( traj->params[6] );

		if( traj->velmodel == EXPONENT )  traj->params[7] = fabs( traj->params[7] );


        //-------- Form statistics on radiant, velocity, deceleration estimates

        ECI2RahDec( &traj->params[3], &ra, &dec );

        VectorNorm( &traj->params[3], radiant_hat, &vbegin );

        if( ra - traj->ra_radiant > +3.14159265359 )  ra -= 2.0 * 3.14159265359;
        if( ra - traj->ra_radiant < -3.14159265359 )  ra += 2.0 * 3.14159265359;

        traj->ra_sigma   += (ra  - traj->ra_radiant ) * (ra  - traj->ra_radiant );
        traj->dec_sigma  += (dec - traj->dec_radiant) * (dec - traj->dec_radiant);

        traj->vbegin_sigma += (               vbegin - traj->vbegin) * (               vbegin - traj->vbegin);
        traj->decel1_sigma += (fabs(traj->params[6]) - traj->decel1) * (fabs(traj->params[6]) - traj->decel1);
        traj->decel2_sigma += (fabs(traj->params[7]) - traj->decel2) * (fabs(traj->params[7]) - traj->decel2);

		//-------- Get the gravitation constant and direction of gravity

		VectorNorm( &traj->params[0], gravity_hat, &romag );

	    ECEF2LatLonAlt( &traj->params[0], &ro_lat, &ro_LST, &ro_hkm );  //... height of meteor begin above surface

	    g = g_surface * ( 6378.137 / ( 6378.137 + ro_hkm ) );  //... gravity km/sec^2 corrected for height


        //-------- Form statistics on begin LLA

        length_km = Propagation( POSITION, traj->velmodel, traj->ttbeg, traj->params[8], vbegin, fabs(traj->params[6]), fabs(traj->params[7]) );

	    gravity_km = 0.5 * g * traj->ttbeg * traj->ttbeg;

        r[X] = traj->params[0] - radiant_hat[X] * length_km - gravity_hat[X] * gravity_km;
        r[Y] = traj->params[1] - radiant_hat[Y] * length_km - gravity_hat[Y] * gravity_km;
        r[Z] = traj->params[2] - radiant_hat[Z] * length_km - gravity_hat[Z] * gravity_km;

        ECEF2LatLonAlt( r, &r_lat, &r_LST, &r_hkm );

        r_lon = LST2LongitudeEast( traj->jdt_ref, r_LST );

        if( r_lon - traj->rbeg_lon > +3.14159265359 )  r_lon -= 2.0 * 3.14159265359;
        if( r_lon - traj->rbeg_lon < -3.14159265359 )  r_lon += 2.0 * 3.14159265359;

        traj->rbeg_lon_sigma += (r_lon - traj->rbeg_lon) * (r_lon - traj->rbeg_lon);
        traj->rbeg_lat_sigma += (r_lat - traj->rbeg_lat) * (r_lat - traj->rbeg_lat);

        traj->rbeg_hkm_sigma += (r_hkm - traj->rbeg_hkm) * (r_hkm - traj->rbeg_hkm);


        //..... Form statistics on end LLA

        length_km = Propagation( POSITION, traj->velmodel, traj->ttend, traj->params[8], vbegin, fabs(traj->params[6]), fabs(traj->params[7]) );

	    gravity_km = 0.5 * g * traj->ttend * traj->ttend;

        r[X] = traj->params[0] - radiant_hat[X] * length_km - gravity_hat[X] * gravity_km;
        r[Y] = traj->params[1] - radiant_hat[Y] * length_km - gravity_hat[Y] * gravity_km;
        r[Z] = traj->params[2] - radiant_hat[Z] * length_km - gravity_hat[Z] * gravity_km;

        ECEF2LatLonAlt( r, &r_lat, &r_LST, &r_hkm );

        r_lon = LST2LongitudeEast( traj->jdt_ref, r_LST );

        if( r_lon - traj->rend_lon > +3.14159265359 )  r_lon -= 2.0 * 3.14159265359;
        if( r_lon - traj->rend_lon < -3.14159265359 )  r_lon += 2.0 * 3.14159265359;

        traj->rend_lon_sigma += (r_lon - traj->rend_lon) * (r_lon - traj->rend_lon);
        traj->rend_lat_sigma += (r_lat - traj->rend_lat) * (r_lat - traj->rend_lat);

        traj->rend_hkm_sigma += (r_hkm - traj->rend_hkm) * (r_hkm - traj->rend_hkm);


    }  //... end of Monte Carlo loop


    //======== Compute standard deviations

    if( traj->nummonte > 1 ) kount = (double)(traj->nummonte - 1);
    else                     kount = 1.0;

    traj->ra_sigma       = sqrt( traj->ra_sigma       / kount );
    traj->dec_sigma      = sqrt( traj->dec_sigma      / kount );
    traj->vbegin_sigma   = sqrt( traj->vbegin_sigma   / kount );
    traj->decel1_sigma   = sqrt( traj->decel1_sigma   / kount );
    traj->decel2_sigma   = sqrt( traj->decel2_sigma   / kount );

    traj->rbeg_lon_sigma = sqrt( traj->rbeg_lon_sigma / kount );
    traj->rend_lon_sigma = sqrt( traj->rend_lon_sigma / kount );

    traj->rbeg_lat_sigma = sqrt( traj->rbeg_lat_sigma / kount );
    traj->rend_lat_sigma = sqrt( traj->rend_lat_sigma / kount );

    traj->rbeg_hkm_sigma = sqrt( traj->rbeg_hkm_sigma / kount );
    traj->rend_hkm_sigma = sqrt( traj->rend_hkm_sigma / kount );

}


//##################################################################################

double  VectorDotProduct( double *a, double *b )
{
   double c = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
   return(c);
}

//==============================================================

void  VectorCrossProduct( double *a, double *b, double *c )
{
   c[0] = a[1]*b[2] - a[2]*b[1];
   c[1] = a[2]*b[0] - a[0]*b[2];
   c[2] = a[0]*b[1] - a[1]*b[0];
}

//==============================================================

void  VectorNorm( double *a, double *b, double *c )
{
   *c = sqrt( a[0]*a[0] + a[1]*a[1] + a[2]*a[2] );
   if( *c != 0 ) {
       b[0] = a[0] / *c;
       b[1] = a[1] / *c;
       b[2] = a[2] / *c;
   }
}

//##################################################################################

double  RandomGauss( double sigma, double maxvalue )
{
static short   kthcall = 0;
static double  rangauss1, rangauss2;
double         ran1, ran2, v1, v2, fac, vsq;


     if( kthcall == 0 )  {
         do {

             do {
                  ran1 = (double)rand() / (double)RAND_MAX;
                  ran2 = (double)rand() / (double)RAND_MAX;
                  v1   = 2.0 * ran1 - 1.0;
                  v2   = 2.0 * ran2 - 1.0;
                  vsq  = v1*v1 + v2*v2;
             }
             while( vsq >= 1.0 );

             fac = sqrt( -2.0*log(vsq)/vsq );

             rangauss1 = sigma * v1 * fac;
             rangauss2 = sigma * v2 * fac;
         }
         while( rangauss1 > maxvalue || rangauss2 > maxvalue );

         kthcall = 1;
         return( rangauss1 );
     }
     else {
         kthcall = 0;
         return( rangauss2 );
     }

}

//##################################################################################

//=======================================================================================
// Conversion from geodetic lat, lon, alt to ECEF assuming WGS84 height in km and radians
//=======================================================================================

void  LatLonAlt2ECEF( double lat, double lon, double alt_km, double *ecef_km  )
{
double  a, e, N;

  a = 6378.137;   //... WGS84 constants
  e = 0.081819190842621;

  N = a / sqrt( 1.0 - e*e * sin(lat)*sin(lat) );

  ecef_km[0] = (N + alt_km) * cos(lat) * cos(lon);
  ecef_km[1] = (N + alt_km) * cos(lat) * sin(lon);
  ecef_km[2] =  ((1-e*e) * N + alt_km) * sin(lat);

}

//============================================================================
// Conversion from ECEF to geodetic lat, lon, alt using WGS84 (radians and km)
//============================================================================

void  ECEF2LatLonAlt( double *ecef_km, double *lat, double *lon, double *alt_km )
{
double a, b, e, ep, N, p, theta;


  a = 6378.137;   //... WGS84 constants
  e = 0.081819190842621;

  b = sqrt( a*a * (1.0 - e*e) );
  ep = sqrt( (a*a - b*b) / (b*b) );

  *lon = atan2( ecef_km[1], ecef_km[0] );

  p = sqrt( ecef_km[0] * ecef_km[0]  +  ecef_km[1] * ecef_km[1] );
  theta = atan2( ecef_km[2] * a, p * b );

  *lat = atan2( ecef_km[2] + ep*ep*b*sin(theta)*sin(theta)*sin(theta), p - e*e*a*cos(theta)*cos(theta)*cos(theta) );

  N = a / sqrt( 1.0 - e*e * sin(*lat)*sin(*lat) );

  *alt_km = p / cos(*lat) - N;

  if( fabs(ecef_km[0]) < 0.001  &&  fabs(ecef_km[1]) < 0.001 )  *alt_km = fabs(ecef_km[2]) - b;

}

//============================================================================
// Conversion from ECI to Right Ascension and Declination
//============================================================================

void  ECI2RahDec( double *eci, double *rah, double *dec )
{
double  pi, mag_dummy, rad[3];


    pi = 3.14159265359;

    VectorNorm( eci, rad, &mag_dummy );

    *dec = asin( rad[Z] );

    *rah = atan2( rad[Y], rad[X] );

    if( *rah < 0.0 )  *rah += 2.0 * pi;

}

//============================================================================
// Conversion from Right Ascension and Declination to ECI Coordinates
//============================================================================

void  RahDec2ECI( double rah, double dec, double *eci )
{
    eci[X] = cos(dec) * cos(rah);
    eci[Y] = cos(dec) * sin(rah);
    eci[Z] = sin(dec);

}

//============================================================================
// Conversion from Azimuth (east of north) and Elevation (altitude angle) to
//     Right Ascension and Declination. All angles in radians.
//============================================================================

void   AzimuthElevation2RADec( double  azim,
                               double  elev,
                               double  geodetic_latitude,
                               double  LST,
                               double *RA,
                               double *DEC    )
{
double pi, hourangle, sinlat, coslat;


       pi = 3.14159265359;

       //... Note formulae signs assume azim measured positive east of north

       sinlat = sin( geodetic_latitude );
       coslat = cos( geodetic_latitude );

       *DEC = asin( sinlat * sin( elev )
                  + coslat * cos( elev ) * cos( azim ) );

       hourangle = atan2( -sin( azim ),
                           tan( elev ) * coslat - cos( azim ) * sinlat );

       *RA = LST - hourangle;

       while( *RA < 0.0      )  *RA += 2.0 * pi;
       while( *RA > 2.0 * pi )  *RA -= 2.0 * pi;

}

//============================================================================
// Conversion from Right Ascension and Declination to Azimuth (east of north)
//     and Elevation (altitude angle). All angles in radians.
//============================================================================

void     RADec2AzimuthElevation( double  RA,
                                 double  DEC,
                                 double  geodetic_latitude,
                                 double  LST,
                                 double *Azim,
                                 double *Elev  )
{
double pi, hourangle, sinelev, sinlat, coslat;


       pi = 3.14159265359;

       sinlat = sin( geodetic_latitude );
       coslat = cos( geodetic_latitude );

       hourangle = LST - RA;

       while( hourangle < -pi )  hourangle += 2.0 * pi;
       while( hourangle > +pi )  hourangle -= 2.0 * pi;

       *Azim = pi + atan2( sin( hourangle ),
                           cos( hourangle ) * sinlat
                         - tan( DEC ) * coslat );

       sinelev = sinlat * sin( DEC )
               + coslat * cos( DEC ) * cos( hourangle );

       if( sinelev > +1.0 )  sinelev = +1.0;
       if( sinelev < -1.0 )  sinelev = -1.0;

       *Elev = asin( sinelev );

}

//============================================================================
// Computation of Local Sidereal Time - Based on Meeus
//============================================================================

double  LocalSiderealTimeE( double jdt, double longitude_east )
{
double  pi, tim, stg, lst;  // longitude assumed +east and in radians

        //.................. sidereal time at Greenwich

        pi = 3.14159265359;

        tim = ( jdt - 2451545.0 ) / 36525.0;

        stg = 280.46061837
            + 360.98564736629 * ( jdt - 2451545.0 )
            + tim * tim * 0.000387933
            - tim * tim * tim / 38710000.0;

        //.................. local sidereal time

        lst = stg * pi / 180.0  +  longitude_east;

        //.................. set value between 0 and 2pi

        while( lst >= 2.0 * pi )  lst -= 2.0 * pi;
        while( lst <       0.0 )  lst += 2.0 * pi;

        return( lst );
}

//============================================================================
// Conversion from LST to Longitude using Julian date/time
//============================================================================

double  LST2LongitudeEast( double jdt, double LST )
{
double  pi, tim, stg, eastlongitude;  // longitude assumed +east

        //.................. sidereal time at Greenwich

        pi = 3.14159265359;

        tim = ( jdt - 2451545.0 ) / 36525.0;

        stg = 280.46061837
            + 360.98564736629 * ( jdt - 2451545.0 )
            + tim * tim * 0.000387933
            - tim * tim * tim / 38710000.0;

        //.................. local sidereal time

        eastlongitude = LST - stg * pi / 180.0;

        //.................. set value between 0 and 360 degrees

        while( eastlongitude >= +pi )  eastlongitude -= 2.0 * pi;
        while( eastlongitude <  -pi )  eastlongitude += 2.0 * pi;

        return( eastlongitude );

}


//##################################################################################
//
//======== Function to initially allocate memory for 1D and 2D arrays based on the
//         maximum number of camera measurement sets expected. This function should
//         be called only once at program start-up.
//==================================================================================

void    InitTrajectoryStructure( int maxcameras, struct trajectory_info *traj )
{
int  kcamera, maxparams;


    maxparams = 9 + maxcameras;   //... Multi-parameter fitting
                                  //    position vector, radiant unit vector, velocity, decel1, decel2, dtime(#cameras)

    traj->maxcameras = maxcameras;
    traj->numcameras = 0;

    traj->camera_lat     =   (double*) malloc( maxcameras * sizeof( double  ) );
    traj->camera_lon     =   (double*) malloc( maxcameras * sizeof( double  ) );
    traj->camera_hkm     =   (double*) malloc( maxcameras * sizeof( double  ) );
    traj->camera_LST     =   (double*) malloc( maxcameras * sizeof( double  ) );
    traj->tref_offsets   =   (double*) malloc( maxcameras * sizeof( double  ) );
    traj->nummeas        =      (int*) malloc( maxcameras * sizeof(    int  ) );
    traj->malloced       =      (int*) malloc( maxcameras * sizeof(    int  ) );

    traj->params         =   (double*) malloc( maxparams  * sizeof( double  ) );
    traj->solution       =   (double*) malloc( maxparams  * sizeof( double  ) );
    traj->limits         =   (double*) malloc( maxparams  * sizeof( double  ) );
    traj->xguess         =   (double*) malloc( maxparams  * sizeof( double  ) );
    traj->xshift         =   (double*) malloc( maxparams  * sizeof( double  ) );

    traj->meas1          =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->meas2          =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->dtime          =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->noise          =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->weight         =  (double**) malloc( maxcameras * sizeof( double* ) );

    traj->meas_lat       =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->meas_lon       =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->meas_hkm       =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->meas_range     =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->meas_vel       =  (double**) malloc( maxcameras * sizeof( double* ) );

    traj->model_lat      =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->model_lon      =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->model_hkm      =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->model_range    =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->model_vel      =  (double**) malloc( maxcameras * sizeof( double* ) );

    traj->model_fit1     =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->model_fit2     =  (double**) malloc( maxcameras * sizeof( double* ) );
    traj->model_time     =  (double**) malloc( maxcameras * sizeof( double* ) );

    traj->meashat_ECI    = (double***) malloc( maxcameras * sizeof( double**) );
    traj->rcamera_ECI    = (double***) malloc( maxcameras * sizeof( double**) );



    if( traj->camera_lat     == NULL  ||
        traj->camera_lon     == NULL  ||
        traj->camera_hkm     == NULL  ||
        traj->camera_LST     == NULL  ||
        traj->tref_offsets   == NULL  ||
        traj->nummeas        == NULL  ||
        traj->params         == NULL  ||
        traj->solution       == NULL  ||
        traj->limits         == NULL  ||
        traj->xguess         == NULL  ||
        traj->xshift         == NULL  ||
        traj->malloced       == NULL  ||
        traj->meas1          == NULL  ||
        traj->meas2          == NULL  ||
        traj->dtime          == NULL  ||
        traj->noise          == NULL  ||
        traj->weight         == NULL  ||
        traj->meas_lat       == NULL  ||
        traj->meas_lon       == NULL  ||
        traj->meas_hkm       == NULL  ||
        traj->meas_range     == NULL  ||
        traj->meas_vel       == NULL  ||
        traj->model_lat      == NULL  ||
        traj->model_lon      == NULL  ||
        traj->model_hkm      == NULL  ||
        traj->model_range    == NULL  ||
        traj->model_vel      == NULL  ||
        traj->model_fit1     == NULL  ||
        traj->model_fit2     == NULL  ||
        traj->model_time     == NULL  ||
        traj->rcamera_ECI    == NULL  ||
        traj->meashat_ECI    == NULL      )  {

        printf("ERROR--> Memory not allocated for vectors and arrays in InitTrajectoryStructure\n");
        Delay_msec(15000);
		exit(1);

    }


    //======== Set the memory allocated flag to "freed"

    for( kcamera=0; kcamera<traj->maxcameras; kcamera++ )  traj->malloced[kcamera] = 0;


}


//##################################################################################
//
//======== Function to free up ALL allocated memory for 1D, 2D and 3D arrays. This
//         should be called only once at program completion
//==================================================================================

void    FreeTrajectoryStructure( struct trajectory_info *traj )
{

    //... First free up the column dimensions of the 2D arrays

    ResetTrajectoryStructure( 0.0, 0.0, 0, 0, 0, 0, traj );

    free( traj->meas1          );     //... Now free up the row dimension of the 2D arrays
    free( traj->meas2          );
    free( traj->dtime          );
    free( traj->noise          );
    free( traj->weight         );

    free( traj->meas_lat       );
    free( traj->meas_lon       );
    free( traj->meas_hkm       );
    free( traj->meas_range     );
    free( traj->meas_vel       );

    free( traj->model_lat      );
    free( traj->model_lon      );
    free( traj->model_hkm      );
    free( traj->model_range    );
    free( traj->model_vel      );

    free( traj->model_fit1     );
    free( traj->model_fit2     );
    free( traj->model_time     );

    free( traj->rcamera_ECI    );
    free( traj->meashat_ECI    );

    free( traj->malloced       );


    free( traj->camera_lat     );     //... Free up the vectors
    free( traj->camera_lon     );
    free( traj->camera_hkm     );
    free( traj->camera_LST     );
    free( traj->tref_offsets   );
    free( traj->nummeas        );
    free( traj->params         );
    free( traj->solution       );
    free( traj->limits         );
    free( traj->xguess         );
    free( traj->xshift         );


}

//##################################################################################
//
//======== Function to reset the trajectory structure by freeing up the column
//         dimension (measurement dimension) of the 2D arrays.
//==================================================================================

void    ResetTrajectoryStructure( double jdt_ref, double max_toffset,
                                  int velmodel, int nummonte, int meastype, int verbose,
                                  struct trajectory_info *traj )
{
int  kcamera, kmeas;


    //======== Set up some initial parameters for this trajectory solution

    traj->jdt_ref     = jdt_ref;
    traj->max_toffset = max_toffset;
    traj->velmodel    = velmodel;
    traj->nummonte    = nummonte;
    traj->meastype    = meastype;
    traj->verbose     = verbose;

    traj->numcameras = 0;

    for( kcamera=0; kcamera<traj->maxcameras; kcamera++ )  traj->tref_offsets[kcamera] = 0.0;

	traj->dtime_tzero = 0.0;


    //======== Free memory through all possible camera arrays of 2D or 3D dimensions

    for( kcamera=0; kcamera<traj->maxcameras; kcamera++ )  {

        if( traj->malloced[kcamera] == 1 )  {

            free( traj->meas1[kcamera]          );
            free( traj->meas2[kcamera]          );
            free( traj->dtime[kcamera]          );
            free( traj->noise[kcamera]          );
            free( traj->weight[kcamera]         );

            free( traj->meas_lat[kcamera]       );
            free( traj->meas_lon[kcamera]       );
            free( traj->meas_hkm[kcamera]       );
            free( traj->meas_range[kcamera]     );
            free( traj->meas_vel[kcamera]       );

            free( traj->model_lat[kcamera]      );
            free( traj->model_lon[kcamera]      );
            free( traj->model_hkm[kcamera]      );
            free( traj->model_range[kcamera]    );
            free( traj->model_vel[kcamera]      );

            free( traj->model_fit1[kcamera]     );
            free( traj->model_fit2[kcamera]     );
            free( traj->model_time[kcamera]     );

            //... free first the 3rd dimension (XYZ or XYZT) of rcamera_ECI and meashat_ECI 
			//       then free the 2nd measurement dimension

            for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  free( traj->rcamera_ECI[kcamera][kmeas] );
            for( kmeas=0; kmeas<traj->nummeas[kcamera]; kmeas++ )  free( traj->meashat_ECI[kcamera][kmeas] );

            free( traj->rcamera_ECI[kcamera] );
            free( traj->meashat_ECI[kcamera] );

            traj->malloced[kcamera] = 0;  //... memory freed

        } //... end of IF preiously malloc'ed

    } //... end of "kcamera" loop


}

//##################################################################################
//
//======== Function to infill the trajectory structure with a new set of measurements
//         from a single camera. Multiple calls to this function builds up the full
//         measurement set for multiple sites and multiple cameras.
//==================================================================================

void    InfillTrajectoryStructure( int nummeas, double *meas1, double *meas2, double *dtime, double *noise,
                                   double site_latitude, double site_longitude, double site_height,
                                   struct trajectory_info *traj )
{
int  kcamera, kmeas;


    kcamera = traj->numcameras;

    AllocateTrajectoryMemory4Infill( kcamera, nummeas, traj );

    traj->nummeas[kcamera] = nummeas;


    //======== Assign the measurements to the working arrays

    for( kmeas=0; kmeas<nummeas; kmeas++ )  {

        traj->meas1[kcamera][kmeas] = meas1[kmeas];
        traj->meas2[kcamera][kmeas] = meas2[kmeas];
        traj->dtime[kcamera][kmeas] = dtime[kmeas];
        traj->noise[kcamera][kmeas] = noise[kmeas];

    }

    traj->camera_lat[kcamera]   = site_latitude;
    traj->camera_lon[kcamera]   = site_longitude;
    traj->camera_hkm[kcamera]   = site_height;
    traj->camera_LST[kcamera]   = LocalSiderealTimeE( traj->jdt_ref, site_longitude );

    traj->tref_offsets[kcamera] = 0.0;


    traj->numcameras = kcamera + 1;  //... Increment the active camera counter for next camera's measurments


}


//##################################################################################
//
//======== Function to allocate memory for the column dimension of the 2D arrays
//==================================================================================

void    AllocateTrajectoryMemory4Infill( int kcamera, int nummeas, struct trajectory_info *traj )
{
int  kmeas;

    //======== Check to make sure memory was previously freed for each array column

    if( traj->malloced[kcamera] == 1 )  {

        printf("ERROR--> You must call ResetTrajectoryStructure prior to the first InfillTrajectoryStructure call\n");
        Delay_msec(15000);
        exit(1);

    }


    //======== Allocate memory through all the working arrays of this camera index

    traj->meas1[kcamera]          = (double*) malloc( nummeas * sizeof(double) );
    traj->meas2[kcamera]          = (double*) malloc( nummeas * sizeof(double) );
    traj->dtime[kcamera]          = (double*) malloc( nummeas * sizeof(double) );
    traj->noise[kcamera]          = (double*) malloc( nummeas * sizeof(double) );
    traj->weight[kcamera]         = (double*) malloc( nummeas * sizeof(double) );

    traj->meas_lat[kcamera]       = (double*) malloc( nummeas * sizeof(double) );
    traj->meas_lon[kcamera]       = (double*) malloc( nummeas * sizeof(double) );
    traj->meas_hkm[kcamera]       = (double*) malloc( nummeas * sizeof(double) );
    traj->meas_range[kcamera]     = (double*) malloc( nummeas * sizeof(double) );
    traj->meas_vel[kcamera]       = (double*) malloc( nummeas * sizeof(double) );

    traj->model_lat[kcamera]      = (double*) malloc( nummeas * sizeof(double) );
    traj->model_lon[kcamera]      = (double*) malloc( nummeas * sizeof(double) );
    traj->model_hkm[kcamera]      = (double*) malloc( nummeas * sizeof(double) );
    traj->model_range[kcamera]    = (double*) malloc( nummeas * sizeof(double) );
    traj->model_vel[kcamera]      = (double*) malloc( nummeas * sizeof(double) );

    traj->model_fit1[kcamera]     = (double*) malloc( nummeas * sizeof(double) );
    traj->model_fit2[kcamera]     = (double*) malloc( nummeas * sizeof(double) );
    traj->model_time[kcamera]     = (double*) malloc( nummeas * sizeof(double) );

    traj->rcamera_ECI[kcamera]    = (double**) malloc( nummeas * sizeof(double*) );
    traj->meashat_ECI[kcamera]    = (double**) malloc( nummeas * sizeof(double*) );


    //======== Check the 2D memory was allocated

    if( traj->meas1[kcamera]          == NULL  ||
        traj->meas2[kcamera]          == NULL  ||
        traj->dtime[kcamera]          == NULL  ||
        traj->noise[kcamera]          == NULL  ||
        traj->weight[kcamera]         == NULL  ||
        traj->meas_lat[kcamera]       == NULL  ||
        traj->meas_lon[kcamera]       == NULL  ||
        traj->meas_hkm[kcamera]       == NULL  ||
        traj->meas_range[kcamera]     == NULL  ||
        traj->meas_vel[kcamera]       == NULL  ||
        traj->model_lat[kcamera]      == NULL  ||
        traj->model_lon[kcamera]      == NULL  ||
        traj->model_hkm[kcamera]      == NULL  ||
        traj->model_range[kcamera]    == NULL  ||
        traj->model_vel[kcamera]      == NULL  ||
        traj->model_fit1[kcamera]     == NULL  ||
        traj->model_fit2[kcamera]     == NULL  ||
        traj->model_time[kcamera]     == NULL  ||
        traj->meashat_ECI[kcamera]        == NULL      )  {

        printf("ERROR--> Memory not allocated for 2D array columns in AllocateTrajectoryMemory4Infill\n");
		Delay_msec(15000);
        exit(1);

    }

    //======== Allocate the 3rd dimension for components XYZT of the measurement unit vectors
    //              and the 3rd dimension for components XYZ  of the camera site vectors

    for( kmeas=0; kmeas<nummeas; kmeas++ )  {

        traj->rcamera_ECI[kcamera][kmeas] = (double*) malloc( 3 * sizeof(double) );
        traj->meashat_ECI[kcamera][kmeas] = (double*) malloc( 4 * sizeof(double) );

        if( traj->meashat_ECI[kcamera][kmeas] == NULL  ||  traj->rcamera_ECI[kcamera][kmeas] == NULL )  {
            printf("ERROR--> Memory not allocated for meashat_ECI or rcamera_ECIin AllocateTrajectoryMemory4Infill\n");
			Delay_msec(15000);
            exit(1);
        }

    }


    //======== Set the memory allocated flag for this camera

    traj->malloced[kcamera] = 1;


}

//################################################################################
//
//======== Function to read the Particle Swarm Optimizer fit control and settings
//         specific to the trajectory solver
//==================================================================================

void    ReadTrajectoryPSOconfig( char* PSOconfig_pathname, struct trajectory_info *traj )
{
int     kcall;
char    text[128], desc[128];
FILE   *PSOfile;


   //------------------ Open the particle swarm optimizer configuration file

   if( ( PSOfile = fopen( PSOconfig_pathname, "r" )) == NULL )  {
     	printf(" Cannot open PSO config file %s for reading \n", PSOconfig_pathname );
        Delay_msec(15000);
        exit(1);
   }
      

   //------------------ Skip past header lines
      
   fscanf( PSOfile, "%[^\n]\n", text );
   fscanf( PSOfile, "%[^\n]\n", text );
   fscanf( PSOfile, "%[^\n]\n", text );



   //------------------ Check the mnemonic identifiers are in correct order

   fscanf(PSOfile,"%[^=]= %s", desc, text );

   if( strstr( text, "QUICK_FIT" ) == NULL )  {
	   printf("PSO config file error, 1st column entry should be QUICK_FIT\n");
	   Delay_msec(15000);
	   exit(1);
   }
      
   fscanf(PSOfile,"%s", text );

   if( strstr( text, "GOOD_FIT" ) == NULL )  {
	   printf("PSO config file error, 2nd column entry should be GOOD_FIT\n");
	   Delay_msec(15000);
	   exit(1);
   }
      
   fscanf(PSOfile,"%s", text );

   if( strstr( text, "ACCURATE_FIT" ) == NULL )  {
	   printf("PSO config file error, 3rd column entry should be ACCURATE_FIT\n");
	   Delay_msec(15000);
	   exit(1);
   }
      
   fscanf(PSOfile,"%[^\n]\n", text );

   if( strstr( text, "EXTREME_FIT" ) == NULL )  {
	   printf("PSO config file error, 4th column entry should be EXTREME_FIT\n");
	   Delay_msec(15000);
	   exit(1);
   }
      

   //------------------ Read each of the config parameters for all fit options

   fscanf( PSOfile, "%[^=]= %d %d %d %d\n", desc, 
	                                        &traj->PSOfit[0].number_particles,
	                                        &traj->PSOfit[1].number_particles,
								            &traj->PSOfit[2].number_particles,
									        &traj->PSOfit[3].number_particles );

   fscanf( PSOfile, "%[^=]= %d %d %d %d\n", desc, 
	                                        &traj->PSOfit[0].maximum_iterations,
	                                        &traj->PSOfit[1].maximum_iterations,
									        &traj->PSOfit[2].maximum_iterations,
									        &traj->PSOfit[3].maximum_iterations );

   fscanf( PSOfile, "%[^=]= %d %d %d %d\n", desc, 
	                                        &traj->PSOfit[0].boundary_flag,
	                                        &traj->PSOfit[1].boundary_flag,
									        &traj->PSOfit[2].boundary_flag,
									        &traj->PSOfit[3].boundary_flag );

   fscanf( PSOfile, "%[^=]= %d %d %d %d\n", desc, 
	                                        &traj->PSOfit[0].limits_flag,
	                                        &traj->PSOfit[1].limits_flag,
									        &traj->PSOfit[2].limits_flag,
									        &traj->PSOfit[3].limits_flag );

   fscanf( PSOfile, "%[^=]= %d %d %d %d\n", desc, 
	                                        &traj->PSOfit[0].particle_distribution_flag,
	                                        &traj->PSOfit[1].particle_distribution_flag,
									        &traj->PSOfit[2].particle_distribution_flag,
									        &traj->PSOfit[3].particle_distribution_flag );

   fscanf( PSOfile, "%[^=]= %lf %lf %lf %lf\n", desc, 
	                                            &traj->PSOfit[0].epsilon_convergence,
	                                            &traj->PSOfit[1].epsilon_convergence,
									  	        &traj->PSOfit[2].epsilon_convergence,
										        &traj->PSOfit[3].epsilon_convergence );

   fscanf( PSOfile, "%[^=]= %lf %lf %lf %lf\n", desc, 
	                                            &traj->PSOfit[0].weight_inertia,
	                                            &traj->PSOfit[1].weight_inertia,
									  	        &traj->PSOfit[2].weight_inertia,
										        &traj->PSOfit[3].weight_inertia );

   fscanf( PSOfile, "%[^=]= %lf %lf %lf %lf\n", desc, 
	                                            &traj->PSOfit[0].weight_stubborness,
	                                            &traj->PSOfit[1].weight_stubborness,
									  	        &traj->PSOfit[2].weight_stubborness,
										        &traj->PSOfit[3].weight_stubborness );

   fscanf( PSOfile, "%[^=]= %lf %lf %lf %lf\n", desc, 
	                                            &traj->PSOfit[0].weight_grouppressure,
	                                            &traj->PSOfit[1].weight_grouppressure,
									  	        &traj->PSOfit[2].weight_grouppressure,
										        &traj->PSOfit[3].weight_grouppressure );


   //------------------ Skip past some more header lines
      
   fscanf( PSOfile, "%[^\n]\n", text );
   fscanf( PSOfile, "%[^\n]\n", text );
   fscanf( PSOfile, "%[^\n]\n", text );


   //------------------ Read the fit type for each stage (call) of the trajectory bootstrapping process

   for( kcall=0; kcall<NPSO_CALLS; kcall++ )  {

       fscanf( PSOfile, "%[^=]= %[^\n]\n", desc, text );

       if(      strstr( text, "QUICK_FIT"    ) != NULL )  traj->PSO_fit_control[kcall] = QUICK_FIT;
       else if( strstr( text, "GOOD_FIT"     ) != NULL )  traj->PSO_fit_control[kcall] = GOOD_FIT;
       else if( strstr( text, "ACCURATE_FIT" ) != NULL )  traj->PSO_fit_control[kcall] = ACCURATE_FIT;
       else if( strstr( text, "EXTREME_FIT"  ) != NULL )  traj->PSO_fit_control[kcall] = EXTREME_FIT;
       else {
		   printf("Incorrect fit name in PSO config file for line %d of the control settings\n", kcall+1 );
		   Delay_msec(15000);
		   exit(1);
	   }

   } //... end of stage/call loop
                     
   fclose( PSOfile );
   

}

//################################################################################
