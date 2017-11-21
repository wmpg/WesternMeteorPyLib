
//###################################################################################
//              Functions to perform Particle Swarm Optimization (PSO)
//###################################################################################
//
//  Based on:  Y. Shi and R. Eberhart, 1998
//             "A Modified Particle Swarm Optimizer"  
//             1998 Proceedings of Congress on Evolutionary Computation, 79-73
//
//            I.G. Tsoulos and A. Stavrakoudis, 2010
//            "Enhancing PSO Methods for Global Optimization"
//            Applied Mathemathical Compututation, 216, 2988-3001
//
//            M. Eslami, H. Shareef, M. Khajehzadeh, and A. Mohamed, 2012
//            "A Survey of the State of the Art in Particle Swarm Optimization"
//            Research Journal of Applied Sciences, Engineering and 
//                Technology 4(9): 1181-1197
//
//###################################################################################s
//  Implementation of an asynchronous particle swarm optimizer (PSO) to perform multi-
//  dimensional minimization of a user computed cost function. This PSO function set 
//  operates differently than most typical minimizer functions. Rather than pass down
//  a pointer to a cost function along with all the parameters that are both varying  
//  and fixed, this set of functions wraps around a user's cost function computation.
//  The user first calls a pre-population function at program start to set up any 
//  control flags and memory allocations needed for the the PSO (if those change
//  during the program, then the memory cleanup function needs to be called and a 
//  new pre-population call made with updated arguments).  
//
//  For each minimization problem using the same pre-population parameters, the first
//  step is an initialization based on a user's best guess of the varying parameters
//  and their plus/minus range of search (shift) from those nominal guess values. The
//  process then proceeds iteration by iteration, by looping over all the particles 
//  in an iteration. Embedded in a while loop, the user computes the cost function for
//  a test vector pso.xtest[*] that gets updated internally in the PSO structure. For
//  each processing pass, the while loop runs sequentially over iterations and particles.  
//  Convergence criteria are based on either acheiving a consistently small change in
//  the cost function, a low variance in the best parameter set, or a maximum iteration
//  limit is reached. Any of the three conditions halts the processing by terminating 
//  the while loop. A pseudo-code example is shown below.
//
//  For convenience, one can have some parameters fixed (constrained not to change)
//  rather than pass down a subset of only varying parameters. To constrain a 
//  parameter, set its shift to zero. The "limits_flag" of STRICT or LOOSE
//  is ignored for that parameter and the parameter velocity is set to zero.
//
//  Note that this PSO is based on an asynchronous global best parameter update 
//  such that each particle is propagated with a velocity that was derived using
//  the most recent best swarm solution. The code incorporates a cost function 
//  computation avoidance if the parameters of any particle does not change between
//  iterations to one part in a million (relative). The code also allows for the 
//  search limit boundaries (lower or upper) to grow on a per particle basis if the
//  best solution approaches a specific boundary.
//
//###################################################################################
//
//  double   costfunctionvalue;
//  double  *guess_parameters, *shift_limits;  //... allocate memory pso.narams
//  struct   particleswarming  pso;
//  ...
//
//  ParticleSwarm_PrePopulate( ..., &pso );   Startup call for a given set of control parameters
//  ...
//
//  ============ Ready to solve a minimization problem where each parameter's range is given by:  
//                                                       lower limit = guess - shift
//                                                       upper limit = guess + shift
//
//  Assign guess_parameters[*] and shift_limits[*]  for * = 0 to pso.nparams-1
//
//
//  ParticleSwarm_Initialize( &guess_parameters[0], &shift_limits[0], &pso ); --> fills 1st  pso.test  parameter vector
//
//  while( pso.processing == CONTINUE_PROCESSING )  {
//         
//         User computes cost function given the vector  pso.xtest[*]  --> costfunctionvalue
//
//         ParticleSwarm_Update( costfunctionvalue, &pso );  --> fills new  pso.xtest  parameter vector
//      
//  }  
//
//
//  Do something with minimized solution stored in vector  pso.gbest[*]
//
//  Number of cost fucntion evaluations = pso.ncostevals
//
//  Halting condition = pso.processing
//
//  ============ End solving this minimization problem
//
//
//  Repeat for other minimization problems with same pre-populated control parameters ...
//
//  ...
//
//  ParticleSwarm_PostCleanup( &pso );   Frees memory associated with current pre-populated setup
//
//
//###################################################################################
//  Date        Author      Description
//  ----------  ----------  ---------------------------------------------------------
//  2016-10-12  Pete Gural  Initial implementation
//  2017-05-17  Pete Gural  Added PSO parameter read function
//
//###################################################################################


//###################################################################################
//#############################   Structure Definition    ###########################
//###################################################################################

struct  PSOparameters            // Parameters read from a PSO config file
{
	int         nparticles;      // Number of particles in the swarm
	int         maxiter;         // Maximum number of iterations for the optimization
	int         boundary_flag;   // Boundary handlng flag:  0 = reflective, 1 = periodic
	int         limits_flag;     // Boundary limits changing flag:  0 = fixed, 1 = modifiable
	int         distro_flag;     // Initial particle distribution flag: 0 = random, 1 = gaussian

	double      winertia_init;   // PSO weight for inertia - initially (typically 0.8)
	double      wstubborness;    // PSO weight for stubborness (typically 1.0)
	double      wgrouppressure;  // PSO weight for grouppressure (typically 2.0)
	double      eps_convergence; // Convergence criteria of the cost function (typically 1.0e-10)
};



struct  particleswarming
{
	//  Notations:  x* v* = particle level info,    g* = global swarm info
	//
	int         nparticles;      // Number of particles in the swarm
	int         maxiter;         // Maximum number of iterations for the optimization
	int         nparams;         // Number of parameters that we are trying to optimize
	int         boundary_flag;   // Boundary handlng flag:  0 = reflective, 1 = periodic
	int         limits_flag;     // Boundary limits changing flag:  0 = fixed, 1 = modifiable
	int         distro_flag;     // Initial particle distribution flag: 0 = random, 1 = gaussian

	double      winertia;        // PSO weight for inertia - adjusted down per iteration
	double      winertia_init;   // PSO weight for inertia - initially (typically 0.8)
	double      wstubborness;    // PSO weight for stubborness (typically 1.0)
	double      wgrouppressure;  // PSO weight for grouppressure (typically 2.0)
	double      eps_convergence; // Convergence criteria of the cost function (typically 1.0e-10)

	double     *xtest;           // Test vector of length nparams, passed to user for cost function eval

	double     *xshift;          // Shift limit of parameter values, vector of length nparams
	double     *xlower;          // Lower limit of parameter values, vector of length nparams
	double     *xupper;          // Upper limit of parameter values, vector of length nparams
	double    **xcurr;           // Current  parameter values, array of nparticles x nparams
	double    **vcurr;           // Current  parameter velocities, array of nparticles x nparams
	double     *vmaxx;           // Max allowed parameter velocities, vector of length nparams

	double    **xbest;           // Per particle best parameter values, array of nparticles x nparams
	double     *xbestcost;       // The best particle cost function values, vector of length nparticles

	double     *gbest;           // Best global swarm parameter values, vector of length nparams
	double     *gprev;           // Last global swarm parameter values, vector of length nparams
	double      gbestcost;       // Best global swarm cost function value across all particles
	double      gprevcost;       // Last global swarm cost function value across all particles

	double      vsum1;           // Accumulator #1 for variance stopping criteria
	double      vsum2;           // Accumulator #2 for variance stopping criteria
	double      savevar;         // The saved variance threshold stopping criteria

	int         niteration;      // Actual number of iterations executed
	int         nsamecount;      // Counter for when the global best cost is repeatedly the same
	int         ncostevals;      // Counter for number of cost function evaluations
	int         kparticle;       // Index of particle currently being worked on
	int         processing;      // Continue the PSO processing or the resultant halting condition           
                                 //    Iteration CONTINUE_PROCESSING = 1
                                 //    Halted on ACCURACY_LIMIT      = 2
                                 //    Halted on ITERATION LIMIT     = 3
                                 //    Halted on VARIANCE_LIMIT      = 4

};


//###################################################################################
//#############################         Mnemonics         ###########################
//###################################################################################

//------ "boundary_flag" options
#define   BOUNDARY_PERIODIC        0  // Parameter values wrap around upper/lower limits
#define   BOUNDARY_REFLECTIVE      1  // Parameter values reverse velocity at limits

//------ "limits_flag" options
#define   LIMITS_ARE_STRICT        0  // No changes allowed to the boundary limits
#define   LIMITS_ARE_LOOSE         1  // Limits can change based on "best" parameter's proximity to boundary

//------ "distro_flag" options
#define   PARTICLEDISTRO_RANDOM    0  // Uniformly random distribution of particles around initial guess
#define   PARTICLEDISTRO_GAUSS     1  // Normal (Gaussian) distribution of particles around initial guess

//------ "processing" and resultant halting conditions
#define   CONTINUE_PROCESSING      1  // Indicator that particle looping and iterations should continue
#define   ACCURACY_LIMIT           2  // Processing halted due to accuracy convergence acheived
#define   ITERATION_LIMIT          3  // Processing halted due to max iteration count exceeded
#define   VARIANCE_LIMIT           4  // Processing halted due to variance dropped below threshold


//###################################################################################
//#############################   Function Prototypes   #############################
//###################################################################################

int    ParticleSwarm_ReadParameters( char *PSOpathname, struct PSOparameters *psodata );

int    ParticleSwarm_PrePopulate( int number_particles, int number_parameters, int maximum_iterations, 
	                              int boundary_flag, int limits_flag, int particle_distribution_flag, 
								  double epsilon_convergence,
	                              double weight_inertia, double weight_stubborness, double weight_grouppressure,  								   
								  struct particleswarming *pso );

int    ParticleSwarm_Initialize( double *param_guess, double *param_shiftlimit, struct particleswarming *pso );

void   ParticleSwarm_Update( double cost_func_value, struct particleswarming *pso );

int    ParticleSwarm_Propagate( int jparticle, struct particleswarming *pso );

int    ParticleSwarm_ParameterSimilarity( double *curr, double *prev, int nparams );

void   ParticleSwarm_PostCleanup( struct particleswarming *pso );

