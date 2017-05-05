
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
//
//###################################################################################


//###################################################################################
//#############################   Structure Definition    ###########################
//###################################################################################

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


//###################################################################################
//#############################   PSO Functions   ###################################
//###################################################################################

int    ParticleSwarm_PrePopulate( int number_particles, int number_parameters, int maximum_iterations, 
	                              int boundary_flag, int limits_flag, int particle_distribution_flag, 
								  double epsilon_convergence,
	                              double weight_inertia, double weight_stubborness, double weight_grouppressure,  								   
								  struct particleswarming *pso )
{
int  kparticle;


	//======== Fill structure with static values

	pso->nparticles      = number_particles;
	pso->maxiter         = maximum_iterations;
	pso->nparams         = number_parameters;
	pso->boundary_flag   = boundary_flag;
	pso->limits_flag     = limits_flag;
	pso->distro_flag     = particle_distribution_flag;
	pso->winertia        = weight_inertia;
	pso->winertia_init   = weight_inertia;
	pso->wstubborness    = weight_stubborness;
	pso->wgrouppressure  = weight_grouppressure;
	pso->eps_convergence = epsilon_convergence;


	//======== Allocate memory for structure's "1D vectors"

	pso->xshift    = (double*) malloc( pso->nparams    * sizeof(double) );
	pso->xlower    = (double*) malloc( pso->nparams    * sizeof(double) );
	pso->xupper    = (double*) malloc( pso->nparams    * sizeof(double) );
	pso->xtest     = (double*) malloc( pso->nparams    * sizeof(double) );
	pso->vmaxx     = (double*) malloc( pso->nparams    * sizeof(double) );
	pso->gbest     = (double*) malloc( pso->nparams    * sizeof(double) );
	pso->gprev     = (double*) malloc( pso->nparams    * sizeof(double) );
	pso->xbestcost = (double*) malloc( pso->nparticles * sizeof(double) );

	if( pso->xlower == NULL  ||  pso->xtest == NULL  ||  pso->gbest == NULL  ||  pso->xshift    == NULL  ||  
		pso->xupper == NULL  ||  pso->vmaxx == NULL  ||  pso->gprev == NULL  ||  pso->xbestcost == NULL )  {
		printf("ERROR ===> Memory not allocated for 1D vectors in ParticleSwarm_PrePopulate\n");
		return( 1 );
	}


	//======== Allocate memory for structure's "2D arrays" dependent on number of particles
	//            and the number of parameters to optimize

	pso->xbest = (double**) malloc( pso->nparticles * sizeof(double*) );
	pso->xcurr = (double**) malloc( pso->nparticles * sizeof(double*) );
	pso->vcurr = (double**) malloc( pso->nparticles * sizeof(double*) );

	if( pso->xbest == NULL   ||  pso->xcurr == NULL   ||  pso->vcurr == NULL )  {
		printf("ERROR ===> Memory not allocated for 2D array **pointers in ParticleSwarm_PrePopulate\n");
		return( 1 );
	}

	for( kparticle=0; kparticle<pso->nparticles; kparticle++ )  {
		pso->xbest[kparticle] = (double*) malloc( pso->nparams * sizeof(double) );
		pso->xcurr[kparticle] = (double*) malloc( pso->nparams * sizeof(double) );
		pso->vcurr[kparticle] = (double*) malloc( pso->nparams * sizeof(double) );

	    if( pso->xbest[kparticle] == NULL  ||  pso->xcurr[kparticle] == NULL  ||  pso->vcurr[kparticle] == NULL )  {
		    printf("ERROR ===> Memory not allocated for 2D array pointers in ParticleSwarm_PrePopulate\n");
		    return( 1 );
	    }
	}


	//======== Normal return

	return( 0 );

}

								  

//###################################################################################

int    ParticleSwarm_Initialize( double *param_guess, double *param_shiftlimit, struct particleswarming *pso )
{
int     kparam, kparticle;
double  random1, random2, grandom1, grandom2;



	//======== Fill in the lower and upper limits and set the maximum velocity 
    //            at half the search range per parameter

	for( kparam=0; kparam<pso->nparams; kparam++ )  {

		pso->xshift[kparam] = param_shiftlimit[kparam];

		pso->xlower[kparam] = param_guess[kparam] - param_shiftlimit[kparam];
		pso->xupper[kparam] = param_guess[kparam] + param_shiftlimit[kparam];

		pso->vmaxx[kparam]  = 0.5 * ( pso->xupper[kparam] - pso->xlower[kparam] );

	}


	//======== Assign the first particle with the guess values and no initial velocity

	kparticle = 0;
        
	for( kparam=0; kparam<pso->nparams; kparam++ )  {
			
		pso->xcurr[kparticle][kparam] = param_guess[kparam];
	    pso->vcurr[kparticle][kparam] = 0.0;

	}


	//======== Now "position" the remaining particles and assign "velocities" according to distribution type

	if( pso->distro_flag == PARTICLEDISTRO_RANDOM )  {

		//-------- These particles are assigned uniformly randomized parameter "positions" and "velocities" 
		//            that fall within their low/high limits and velocity maximum

		for( kparticle=1; kparticle<pso->nparticles; kparticle++ )  {

			for( kparam=0; kparam<pso->nparams; kparam++ )  {

				random1 = (double)rand() / (double)RAND_MAX;  //.. uniform pseudo-random # = [0,1]
				random2 = (double)rand() / (double)RAND_MAX;

			    pso->xcurr[kparticle][kparam] = pso->xlower[kparam] + random1 * ( pso->xupper[kparam] - pso->xlower[kparam] );

				pso->vcurr[kparticle][kparam] = pso->vmaxx[kparam] * ( 1.0 - random2 ) * ( random1 - 0.5 ) / fabs( random1 - 0.5 );

			}  //... end of parameter loop

		}  //... end of particle loop
	
	} //... end PARTICLEDISTRO_RANDOM


	else if( pso->distro_flag == PARTICLEDISTRO_GAUSS )  {

		//-------- These particles are assigned random gaussian distributed parameter "positions" centered on the guess with
		//            standard deviation equal to half the shift limit for the parameter. Gaussian distributed
		//            "velocities" are centered on zero with standard deviation equal to half the max velocity, 
		//            such that they must fall within the low/high limits and velocity maximum.

		for( kparticle=1; kparticle<pso->nparticles; kparticle++ )  {

			for( kparam=0; kparam<pso->nparams; kparam++ )  {

				random1 = (double)rand() / (double)RAND_MAX;  //.. uniform pseudo-random # = [0,1]
				random2 = (double)rand() / (double)RAND_MAX;

				random1 = sqrt( -2.0 * log( random1 ) );

				grandom1 = random1 * cos( 2.0 * 3.141592654 * random2 );   //.. normal pseudo-random # with mean=0, sigma=1
				grandom2 = random1 * sin( 2.0 * 3.141592654 * random2 );


			    pso->xcurr[kparticle][kparam] = param_guess[kparam] + 0.5 * param_shiftlimit[kparam] * grandom1;

				if( pso->xcurr[kparticle][kparam] > pso->xupper[kparam] )  pso->xcurr[kparticle][kparam] = pso->xupper[kparam];
				if( pso->xcurr[kparticle][kparam] < pso->xlower[kparam] )  pso->xcurr[kparticle][kparam] = pso->xlower[kparam];

				pso->vcurr[kparticle][kparam] = 0.5 * pso->vmaxx[kparam] * grandom2;

				if( pso->vcurr[kparticle][kparam] > +pso->vmaxx[kparam] )  pso->vcurr[kparticle][kparam] = +pso->vmaxx[kparam];
				if( pso->vcurr[kparticle][kparam] < -pso->vmaxx[kparam] )  pso->vcurr[kparticle][kparam] = -pso->vmaxx[kparam];

			}  //... end of parameter loop

		}  //... end of particle loop
	
	} //... end of PARTICLEDISTRO_GAUSS


	else  { 
		printf("ERROR ===> Particle distribution mode %d not implemented in ParticleSwarm_Initialize\n");
		return( 1 );
	}


	//======== Fill the parameter test vector for the first particle to be processed on the first iteration

	pso->kparticle = 0;

	for( kparam=0; kparam<pso->nparams; kparam++ )  pso->xtest[kparam] = pso->xcurr[0][kparam];


	//======== Set a high global best cost function value, zero the variance accumulator sums for one of
	//            the stopping criteria, set the continue processing flag, zero the various counters

	pso->gbestcost  = 1.0e+30;

	pso->vsum1      =  0.0;
	pso->vsum2      =  0.0;
	pso->savevar    = -1.0;

	pso->processing = CONTINUE_PROCESSING;

	pso->nsamecount = 0;

	pso->ncostevals = 0;

    pso->niteration = 0;


	//======== Normal return

	return( 0 );

}

//###################################################################################

void    ParticleSwarm_Update( double cost_func_value, struct particleswarming *pso )
{
int     kparam, curr_particle, next_particle, parameters_changed;
double  sumsq, var;


    //======== Assign indices for current and next particles

    curr_particle = pso->kparticle;

	next_particle = ( curr_particle + 1 ) % pso->nparticles;


    //=================================================================================================
    //========  If this is the first iteration, then we need to wait for all the particle's cost 
    //             function values to get passed through this function on multiple calls as we search 
	//             for the global best value. And only after that do we update the first particle.
    //=================================================================================================

    if( pso->niteration == 0 )  {

		//-------- First pass through, so the current particle's new parameters always results in the best cost function 
		
		pso->xbestcost[curr_particle] = cost_func_value;

		for( kparam=0; kparam<pso->nparams; kparam++ )  pso->xbest[curr_particle][kparam] = pso->xcurr[curr_particle][kparam];


		//-------- Select the global swarm's best particle

		if( cost_func_value < pso->gbestcost )  { 

			pso->gbestcost = cost_func_value;

			for( kparam=0; kparam<pso->nparams; kparam++ )  pso->gbest[kparam] = pso->xbest[curr_particle][kparam];

		}


		//-------- If we are ready to start the second iteration, propagate the parameters of the first particle and 
		//         duplicate the global best parameter set to the previous global best vector.
		//         Otherwise there is no propagation during the first iteration.

		if( next_particle == 0 )  {
			
			ParticleSwarm_Propagate( next_particle, pso ); 

			pso->gprevcost = pso->gbestcost;

			for( kparam=0; kparam<pso->nparams; kparam++ )  pso->gprev[kparam] = pso->gbest[kparam];

		}


	}  //... end of first iteration processing code block



    //=================================================================================================
	//========  All subsequent iterations follow an "asynchronous" global revision where if the global
	//              best parameters change, they immediately feed the NEXT particle's parameter update
    //=================================================================================================

	else  { 

		//-------- Check if the current particle's cost function is better than the best it obtained previously 
		
		if( cost_func_value < pso->xbestcost[curr_particle] )  {
			
			pso->xbestcost[curr_particle] = cost_func_value;

		    for( kparam=0; kparam<pso->nparams; kparam++ )  pso->xbest[curr_particle][kparam] = pso->xcurr[curr_particle][kparam];

		}


		//-------- Check if this is now the global swarm's best particle

		if( cost_func_value < pso->gbestcost )  { 

			pso->gbestcost = cost_func_value;

			for( kparam=0; kparam<pso->nparams; kparam++ )  pso->gbest[kparam] = pso->xbest[curr_particle][kparam];

		}

		
		//-------- Propagate the NEXT particle to be processed for cost function evaluation. If the parameters
		//            for the particle did not change, skip to the next particle to save on cost function evals.

		parameters_changed = ParticleSwarm_Propagate( next_particle, pso );

		while( parameters_changed == 0  &&  next_particle > 0  &&  next_particle < pso->nparticles - 1 )  {

            next_particle++;

			parameters_changed = ParticleSwarm_Propagate( next_particle, pso );

		}

		
	}  //... end of subsequent iterations processing code block



    //=================================================================================================
    //========  If we are going to start a new iteration: adjust inertia weight, check various 
	//              convergence criteria, modify the limits, and increment the iteration count.
    //=================================================================================================

    if( next_particle == 0 )  {

		//-------- Adjust the inertia weight slowly down to half the initial value

		pso->winertia = pso->winertia_init * ( 1.0 - 0.5 * (double)pso->niteration / (double)pso->maxiter );


		//-------- Check the cost function has not changed for 20 sequential iterations

		if( fabs( pso->gbestcost - pso->gprevcost ) < pso->eps_convergence )  pso->nsamecount++;
		else                                                                  pso->nsamecount = 0;

		if( pso->nsamecount > 20 )  pso->processing = ACCURACY_LIMIT;  //... Converged, stop iteration


		//-------- Check if the variance of the best found parameters has shrunken significantly

		sumsq = 0.0;

		for( kparam=0; kparam<pso->nparams; kparam++ )  sumsq += pso->gbest[kparam] * pso->gbest[kparam];

		pso->vsum1 += sqrt( sumsq );
		pso->vsum2 += sumsq;

		var = pso->vsum2 / (double)( pso->niteration + 1 )
			- pso->vsum1 / (double)( pso->niteration + 1 ) 
			* pso->vsum1 / (double)( pso->niteration + 1 );

		parameters_changed = ParticleSwarm_ParameterSimilarity( pso->gbest, pso->gprev, pso->nparams );

		if( parameters_changed == 1 )  pso->savevar = var / 2.0;

		if( var <= pso->savevar )  pso->processing = VARIANCE_LIMIT;  //... Variance criteria met, stop iteration


		//-------- Save the best parameter set and best cost function value to the "previous" best 
		//            storage variables for use at the end of the next iteration's convergence testing.

		pso->gprevcost = pso->gbestcost;

		for( kparam=0; kparam<pso->nparams; kparam++ )  pso->gprev[kparam] = pso->gbest[kparam];


		//-------- Check the best values of the parameters to see if they are close to the limits
		//            and adjust either the lower or upper limits if necessary

		if( pso->limits_flag == LIMITS_ARE_LOOSE )  {

		    for( kparam=0; kparam<pso->nparams; kparam++ )  {

				if( pso->xshift[kparam] != 0.0 )  {  //... Only non-fixed (unconstrained) parameters can be adjusted

				    if( fabs( pso->gbest[kparam] - pso->xlower[kparam] ) < 0.1 * pso->xshift[kparam] )  pso->xlower[kparam] -= pso->xshift[kparam];
				    if( fabs( pso->gbest[kparam] - pso->xupper[kparam] ) < 0.1 * pso->xshift[kparam] )  pso->xupper[kparam] += pso->xshift[kparam];

				}

			}

		}


		//-------- Increment the iteration counter and check we have not exceeded the max iterations

		pso->niteration++;

	    if( pso->niteration > pso->maxiter )  pso->processing = ITERATION_LIMIT;  //... Max exceeded, stop iteration 

	}


    //=================================================================================================
    //======== Assign the next particle's parameters to the "xtest" vector for external user computation
	//            of the cost function, increment the particle index and cost function eval counter.
    //===================================s==============================================================

	for( kparam=0; kparam<pso->nparams; kparam++ )  pso->xtest[kparam] = pso->xcurr[next_particle][kparam];
			
	pso->kparticle = next_particle;

	pso->ncostevals++;


}


//###################################################################################

int     ParticleSwarm_Propagate( int jparticle, struct particleswarming *pso )
{
int     kparam, parameters_changed;
double  random1, random2, random3, vel, xprev;


    //======== Propagate each parameter for this particle, check the value is within bounds, and identify if it changed

    parameters_changed = 0;

    for( kparam=0; kparam<pso->nparams; kparam++ )  {

		random1 = (double)rand() / (double)RAND_MAX;  //.. uniform pseudo-random # = [0,1]
		random2 = (double)rand() / (double)RAND_MAX;
		random3 = (double)rand() / (double)RAND_MAX;

		xprev = pso->xcurr[jparticle][kparam];

		vel = pso->winertia       * random1 *   pso->vcurr[jparticle][kparam]
		    + pso->wstubborness   * random2 * ( pso->xbest[jparticle][kparam] - pso->xcurr[jparticle][kparam] )
		    + pso->wgrouppressure * random3 * ( pso->gbest           [kparam] - pso->xcurr[jparticle][kparam] );

		if( fabs(vel) > pso->vmaxx[kparam] )  vel = pso->vmaxx[kparam] * vel / fabs(vel);  //... limit the speed, keep the sign

		pso->vcurr[jparticle][kparam] = vel;

		pso->xcurr[jparticle][kparam] = pso->xcurr[jparticle][kparam] + vel;


		//-------- Check bounds of the propagated parameter

		if(      pso->boundary_flag == BOUNDARY_PERIODIC )  {  //... Periodic boundary condition

			while( pso->xcurr[jparticle][kparam] > pso->xupper[kparam] )  pso->xcurr[jparticle][kparam] -= pso->xupper[kparam] - pso->xlower[kparam];
			while( pso->xcurr[jparticle][kparam] < pso->xlower[kparam] )  pso->xcurr[jparticle][kparam] += pso->xupper[kparam] - pso->xlower[kparam];

		}


		else if( pso->boundary_flag == BOUNDARY_REFLECTIVE )  {  //... Reflective boundary condition

			if( pso->xcurr[jparticle][kparam] >  pso->xupper[kparam] )  {
				pso->xcurr[jparticle][kparam] = +pso->xupper[kparam];
				pso->vcurr[jparticle][kparam] = -pso->vcurr[jparticle][kparam];
			}

			if( pso->xcurr[jparticle][kparam] <  pso->xlower[kparam] )  {
				pso->xcurr[jparticle][kparam] = +pso->xlower[kparam];
				pso->vcurr[jparticle][kparam] = -pso->vcurr[jparticle][kparam];
			}

		}


	    else  { 
		    printf("ERROR ===> Boundary flag %d not implemented in ParticleSwarm_Propagate\n");
		    exit( 1 );
	    }


		//-------- Identify if any one of the parameters have changed using a relative shift test

		if( ParticleSwarm_ParameterSimilarity( &pso->xcurr[jparticle][kparam], &xprev, 1 ) == 1 )  parameters_changed = 1;


    }  //... end of parameter update loop


	//======== Let the Update routine know that any of the parameters have changed (1) or all remained 
	//            the same (0) to within a certain precision. This allows the Update function to skip  
	//            over cost function evaluations of a particle that has not moved.

	return( parameters_changed );

}


//###################################################################################

int   ParticleSwarm_ParameterSimilarity( double *curr, double *prev, int nparams )
{
int     kparam, parameters_changed;
double  aver, diff;


    //-------- Identify if any one of the parameters have changed using a relative shift test

    parameters_changed = 0;

    for( kparam=0; kparam<nparams; kparam++ )  {

		aver = fabs( curr[kparam] + prev[kparam] ) * 0.5;
		diff = fabs( curr[kparam] - prev[kparam] );

	    if( aver == 0.0  &&  diff        > 1.0e-8 )  parameters_changed = 1;

	    if( aver != 0.0  &&  diff / aver > 1.0e-6 )  parameters_changed = 1;

	}

	return( parameters_changed );

}


//###################################################################################

void   ParticleSwarm_PostCleanup( struct particleswarming *pso )
{
int   jparticle;


	//======== Free vector memory

	free( pso->xshift    );  
	free( pso->xlower    );  
	free( pso->xupper    );
	free( pso->xtest     );
	free( pso->vmaxx     );
	free( pso->gbest     );
	free( pso->gprev     );
	free( pso->xbestcost );


	//======== Free array memory

	for( jparticle=0; jparticle<pso->nparticles; jparticle++ )  {
		free( pso->xbest[jparticle] );
		free( pso->xcurr[jparticle] );
		free( pso->vcurr[jparticle] );
	}

	free( pso->xbest );
	free( pso->xcurr );
	free( pso->vcurr );

}


//###################################################################################

