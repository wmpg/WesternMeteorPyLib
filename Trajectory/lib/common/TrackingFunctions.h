
//#######################################################################################
//  The following functions are designed to provide a multi-frame tracking capability
//  across multiple active tracks that interfaces to an array of measurements/attributes 
//  given at one specific time. It was adapted from the AIM-IT algorithmic functions for 
//  extremely short latency detection and tracking of meteors. The tracking functions have
//  been upgraded to handle more generic input and they rely on an alpha-beta tracker that 
//  assumes nearly constant velocity motion (little to no acceleration). 
//
//  These tracking functions expect that the user has obtained measurements/attributes
//  on a single frame or interleave field, and will call TrackingAllMeasurements only
//  after a COMPLETE set of measurements is available for that given frame or field 
//  time (one cannot call the function twice for the SAME time stamp, because ALL the 
//  active tracks are propagated forward in time before return from the function
//  TrackingAllMeasurements). For interleaved sensors, the TrackingAllMeasurements
//  function should be called for EACH field separately in the odd/even or even/odd 
//  field's time causal order (monotonically increasing in time) and should be properly 
//  time tagged as appropriate to the "field" timing.
//
//  Note that the TrackingAllMeasurements function should be called even if there
//  were ZERO measurements for the frame or fields to allow the M of N tracker status 
//  updating to occur as it processes both detections and missed detections. This
//  allows immature tracks to be dropped or firm tracks to eventually get closed. 
//
//  Once the tracks are updated for a given time stamp at the per-frame level, the
//  user should call the TrackingSaver function to extract any CLOSED tracks and free 
//  up trackers slots for new track generation.
//  
//  TrackingAllMeasurements ingests a set of computed attributes for a number of
//  spatial measurements. However only a few attributes are actually used in this
//  set of tracking functions (row and column centroids). The other attributes are 
//  available to be passed through and populate the track sequence historical vectors
//  for a CLOSED track output product. This will include the corresponding historical
//  sequence of time, row, and column measurements of the track.
//
//  The output product is a track whose temporally historical values have been retained 
//  for each "detection" passed to the tracker. The information can be retreived for 
//  the k-th temporal value as follows:
//
//     struct  trackerinfo  track;
//
//     track.detected[k].attribute      for k = 0, 1, ... track.nummeas
//
//  Where "attribute" is defined per the second column entries below taken
//     from the file AttributeStructure.h
//
//	   double    time;          //  Time in user defined units (sec or frame# or...)
//	   double    rowcentroid;   //  Row centroid in pixels
//	   double    colcentroid;   //  Col centroid in pixels
//	   long      nexceed;       //  Number of exceedance pixels in region
//	   long      nsaturated;    //  Number of saturated pixels in region
//	   long      npixels;       //  Number of total pixels in region
//     double    sumsignal;     //  Summed signal over image chip region
//     double    sumnoise;      //  Summed noise over image chip region
//	   double    SsqNdB;        //  Signal squared over noise detection metric in dB
//	   double    entropy;       //  Region entropy
//	   double    PCD_angle;     //  Phase coded disk angle estimate in deg
//	   double    PCD_width;     //  Phase coded disk line width estimate in pixels
//	   double    Hueckle_angle; //  Hueckle kernel angle estimate in deg
//	   double    Hueckle_width; //  Hueckle kernel line width estimate in pixels
//
//  There are also multiple frame aggregated output products for total SNR, BNR (binary
//  exceedance count to total pixel count), average PCD angle, average entropy, track
//  linearity, and velocity.
//  Others can be easily added to the trackerinfo and computed in TrackingSaver.
//
//
//=======================================================================================
//  The processing calls invoked by a developer are as follows:
//
//  struct  trackerinfo  *trackers;
//
//  TrackingInitialization     - Called once near program start to set up the trackerinfo
//                                  structure which returns a vector of "trackers" and to
//                                  infill the associated run parameter contents.
//
//  ---Frame by frame loop
//
//     "Pre-Processing"        - Front end processes to form the measurement row and
//                                  column centroids and other computed attributes
//                                  for multi-frame tracking.
//
//     TrackingAllMeasurements - Called for each frame/field's measurements collection for
//                                  the given frame/field time to associate measurements
//                                  with tracks and propagate ALL active tracks forward
//                                  to the current time. For interleaved collected frames
//                                  this would entail two calls. One for the odd field
//                                  and a second for the even field measurements (or
//                                  vice versa of even then odd, depending on sensor and
//                                  capture card interleave mode/timing). The field  centroid
//                                  measurements will be time tagged by the actual respective
//                                  "field" times passed as a separate argument.
//
//     while( TrackingSaver( trackers, &track ) == 1 )  {  do stuff with "track"  }
//
//                             - Called to identify any CLOSED tracks, one at a time,
//                                  extract them from the trackers structure, and give
//                                  the user an opportunity to perform final track
//                                  refinement, false alarm reduction, and reporting.
//                                  Each track that is extracted is released back  
//                                  into the new track pool (with a NOTRACK status).
//
//  ---End of frame loop
//
//  TrackingFreeMemory         - Call once at the end of image processing to free memory
//
//
//=======================================================================================
//
//  Date        Author      Change Description
//  ----------  ----------  -------------------------------------------------------------
//  2015-12-21  Gural       Original code based on AIMIT tracking algorithms
//  2016-01-26  Gural       Revised base code for arbitary # of attributes passed
//  2016-07-30  Gural       Added track linearity and velocity
//  2016-10-04  Gural       Numerous improvements to the matched filter processing
//
//#######################################################################################




//#######################################################################################
//                         Mnemonics and Structure Definitions
//#######################################################################################

                             //--- Track status mnemonics
#define  NOTRACK          0  //    No track started, no measurements assigned
#define  SPINUP           1  //    Track started but has not met N of M detections
#define  FIRM             2  //    Solid track maintaining M of N detections
#define  TROUBLED         3  //    Warning that track has missed ndown detections in a row
#define  CLOSED           4  //    Track halted/closed as M of N drop criteria reached

#define  NOT_DETECTED     0  //--- Detection flag mnemonics
#define  WAS_DETECTED     1   

#define  NOT_ASSOCIATED   0  //--- Measurement usage flag mnemonics
#define  WAS_ASSOCIATED   1


//======================================================================================

struct   trackerinfo
{
	//        Tracker control and status parameters
	//-------------------------------------------------------------------------------
	long      maxtracks;    // User specified total number of trackers at initialization
	long      maxhistory;   // User specified max history length at initialization
    long      status;       // Track status (NOTRACK, SPINUP, FIRM, TROUBLED, CLOSED)
	long      numhist;      // # of tracker updates (yes+no detection measurement times)
    long      nummeas;      // # of detected measurements fed the tracker thus far
    long      mfirm;        // "M" out of N  to declare firm track
    long      nfirm;        //  M out of "N" to declare firm track
    long      mdrop;        // "M" out of N  to declare dropped track
    long      ndrop;        //  M out of "N" to declare dropped track
	long      ndown;        //  "N" non-detections in a row to downgrade a track


	//        Alpha-Beta tracker components
	//-------------------------------------------------------------------------------
    double    alpha;        // Alpha coefficient for linear motion
    double    beta;         // Beta coefficent for linear motion

	double    timeupdate;   // Last tracker update time
	double    timemeas;     // Last tracker "detected" measurement time

    double    rowmeas;      // Row measurement last fed the tracker
    double    rowfilt;      // Filtered row from tracker
    double    rowrate;      // Row rate in pixels per timeupdate units
    double    rowpred;      // Predicted row position

    double    colmeas;      // Column measurement last fed the tracker
    double    colfilt;      // Filtered column from tracker
    double    colrate;      // Column rate in pixel per timeupdate units
    double    colpred;      // Predicted column position


	//        Output track products for multiple frames (once status = CLOSED)
	//-------------------------------------------------------------------------------
    double    multi_SNRdB;     // Multi-frame SNR in dB 10*log10(signal^2/noise)
    double    multi_BNRdB;     // Multi-frame BNR in dB 10*log10(exceedances/total_pixels)
	double    multi_entropy;   // Multi-frame average entropy
	double    multi_angle;     // Multi-frame average angle in degrees

	double    multi_rowstart;  // Multi-frame best linear fit model row starting position
	double    multi_colstart;  // Multi-frame best linear fit model col starting position
	double    multi_rowspeed;  // Multi-frame best linear fit model row velocity
	double    multi_colspeed;  // Multi-frame best linear fit model col velocity
	double    multi_linearity; // Multi-frame distance deviation from a straight line
	double    multi_modelerr;  // Multi-frame distance deviation from the model fit


	//        Historical list of detection flags and "detected" measurements
	//-------------------------------------------------------------------------------
	long     *detectflags_history; // Pointer to the detection flag history vector for ALL 
	                               // updates which includes YES and/or NO detections

	struct    attributeinfo  *detected; // pointer to the detections historical storage vector
	                                    // made up of measurement attributes

};



//###################################################################################
// The TrackingSetup function initializes a specific track to a starting state.
//
//===================================================================================


void    TrackingSetup( struct trackerinfo *trak )
{
long    k;

    //======== Initialize the track with starting values

    trak->status  = NOTRACK;
    trak->numhist = 0;
    trak->nummeas = 0;
                  
    trak->rowmeas = 0.0;
    trak->rowpred = 0.0;
    trak->rowrate = 0.0;
    trak->rowfilt = 0.0;
                      
    trak->colmeas = 0.0;
    trak->colpred = 0.0;
    trak->colrate = 0.0;
    trak->colfilt = 0.0;
       
    //======== Clear the detection history
       
    for( k=0; k<trak->maxhistory; k++ )  trak->detectflags_history[k] = NOT_DETECTED;

}


//###################################################################################
// The TrackingInitialization function is called near program startup to allocate
// memory space for the maximum number of tracks needed by the user and the maximum 
// historical flag length and data structure vector length. The function accepts 
// inputs for M of N required detections to establish a firm track, inputs for M of N
// missed detections to drop and close a track, and the number of missed detections 
// in a row to downgrade a firm track to troubled.
//
// INPUTS:
//     maxtracks     Declared maximum number of trackers at initialization
//     maxhistory    Declared maximum history length at initialization
//     mfirm        "M" out of N  to declare firm track
//     nfirm         M out of "N" to declare firm track
//     mdrop        "M" out of N  to declare dropped track as CLOSED
//     ndrop         M out of "N" to declare dropped track as CLOSED
//     ndown        Number of non-detections in a row to downgrade to TROUBLED
//
// OUTPUT:
//    *trackers      Pointer to maxtracks vector of trackerinfo structures
//
//  DEPENDENCIES:
//     TrackingSetup
//
//===================================================================================


struct trackerinfo*  TrackingInitialization( long maxtracks, long maxhistory, long mfirm, long nfirm, long mdrop, long ndrop, long ndown )
{
long    ktrack;
struct  trackerinfo  *trak;
struct  trackerinfo  *trackers;

       
    //======== Allocate memory for the trackerinfo structures

    trackers = (struct trackerinfo* ) malloc( maxtracks * sizeof( struct trackerinfo ) );

	if( trackers == NULL )  {
	    printf("ERROR allocating memory for trackerinfo structure in TrackingInitialization\n");
		Delay_msec(10000);
		exit(1);
	}


    //======== Loop over each track allocated and initialize parameters

	for( ktrack=0; ktrack<maxtracks; ktrack++ )  {

 	     trak = &trackers[ktrack];  //... pointer to a specific "trackers" vector element

		 //-------- Allocate memory for trak detection flags history

         trak->detectflags_history = ( long* ) malloc( maxhistory * sizeof( long ) );

	     if( trak->detectflags_history == NULL )  {
	         printf("ERROR allocating memory on track %d for detectflags_history in TrackingInitialization\n", ktrack );
		     Delay_msec(10000);
		     exit(1);
	     }

		 //-------- Allocate memory for trak detection attributes

         trak->detected = ( struct  attributeinfo* ) malloc( maxhistory * sizeof( struct  attributeinfo ) );

	     if( trak->detected == NULL )  {
	         printf("ERROR allocating memory on track %d for attributeinfo detected in TrackingInitialization\n", ktrack );
		     Delay_msec(10000);
		     exit(1);
	     }

		 //--------- Infill track status constraints

         trak->mfirm      = mfirm;
         trak->nfirm      = nfirm;
         trak->mdrop      = mdrop;
         trak->ndrop      = ndrop;
		 trak->ndown      = ndown;

		 trak->maxtracks  = maxtracks;
		 trak->maxhistory = maxhistory;

		 TrackingSetup( trak );  //... reset each track to startup (NOTRACK)

	}

	return( trackers );
       
}


//###################################################################################
// The TrackingFreeMemory function frees the tracker memory and is usually called
// once near the end of the program after image processing has been completed.
//
//===================================================================================

void    TrackingFreeMemory( struct trackerinfo *trackers )
{
long    ktrack;
struct  trackerinfo  *trak;

    for( ktrack=0; ktrack<trackers[0].maxtracks; ktrack++ )  {

 	     trak = &trackers[ktrack];  //... pointer to a specific "trackers" vector element

		 free( trak->detectflags_history );
		 free( trak->detected );
	}

	free( trackers );
}



//###################################################################################
// The TrackingProjection function propagates (coasts via constant velocity) the
// track row and column positions based on the difference in time since the last
// track update and the projected time forward.
//
//===================================================================================

void   TrackingProjection( double               time_projected, 
                           struct trackerinfo  *trak,
                           double              *row_projected, 
                           double              *col_projected )
{
double dtime;


       //======== Delta time between future projected time and the time since
       //            the last tracker update is used to coast the tracker
       //            forward and predict the next row and col position.

       dtime = time_projected - trak->timeupdate;
              
       *row_projected = trak->rowfilt + dtime * trak->rowrate;
       
       *col_projected = trak->colfilt + dtime * trak->colrate;
       

}

//###################################################################################
// The TrackingAssociation function checks ALL provided measurements against a single
// track that is propagated forward to the measurement time. It will first identify  
// that measurement spatially closest to the predicted track position. If the spatial
// distance of that closest measurement is within dtolerance pixels of the predicted
// track position, then that measurement index is returned to the calling function.
// Otherwise a -1 index is returned indicating no measurement was found meeting the
// tolerance criteria. Note that all the measurements in the row and column vectors
// are assumed "detected".
//
// INPUTS:
//     nmeas        Number of measurements in rowmeas and colmeas
//    *rowmeas      Pointer to the vector of row position measurements 
//    *colmeas      Pointer to the vector of col position measurements 
//     time         Time stamp of the measurments
//     dtolerance   Spatial proximty to associate meas to track (pixels)
//    *trak         Pointer to the trackerinfo structure containing the track
//
// OUTPUT:
//     kmeas        Index into the measurement vector element that was associated
//                     such that -1 indicates no measurements were associated 
//                     with this track
//
//  DEPENDENCIES:
//     TrackingProjection
//
//===================================================================================

long    TrackingAssociation( long                 nmeas,
                             double              *rowmeas,
                             double              *colmeas,
                             double               time,
							 double               dtolerance,
                             struct trackerinfo  *trak )
{
long    kmeas, kmeas_closest;
double  dsq, dsqmin, rowproj, colproj, scaled_dtolerance;


    //======== Check for any measurements to associate to

	kmeas_closest = -1;

    if( nmeas == 0 )  return( kmeas_closest );


	//======== Project the given track forward to the measurement time

    TrackingProjection( time, trak, &rowproj, &colproj );


	//======== Loop through all measurements to find closest association to the given track.

	dsqmin = 1.0e+30;

	for( kmeas=0; kmeas<nmeas; kmeas++ )  {

		 //-------- Find smallest pixel distance between the projected track position and measurement

		 dsq = ( rowproj - rowmeas[kmeas] ) * ( rowproj - rowmeas[kmeas] )
			 + ( colproj - colmeas[kmeas] ) * ( colproj - colmeas[kmeas] );

		 if( dsqmin > dsq )  {
			 dsqmin = dsq;
			 kmeas_closest = kmeas;
		 }

	} //... end of loop over all tracks


	//======== Determine if the closest projected track is outside reasonable bounds.

	if( trak->status < FIRM  ||  trak->nummeas < 2 )  scaled_dtolerance = dtolerance;
	else                                              scaled_dtolerance = dtolerance / ( (double)trak->nummeas - 1.0 );
	
	if( scaled_dtolerance < 5.0 )  scaled_dtolerance = 5.0;

	if( dsqmin > scaled_dtolerance * scaled_dtolerance )  kmeas_closest = -1;


	return( kmeas_closest );

}


//###################################################################################
// The TrackingStatus function looks at the recent historical sequence of detection
// flags to determine the current state of the track. 
//
//===================================================================================

void    TrackingStatus( struct trackerinfo *trak )
{
long    k, klimit, hits, misses;
              
       
    //======== Check the most recent nfirm historical detection flags
	//             to determine promotion of a SPINUP or TROUBLED track
	//             to a FIRM track.
       
    if( trak->status == SPINUP  ||  trak->status == TROUBLED )  {

		klimit = trak->numhist - trak->nfirm;
		if( klimit < 0 )  klimit = 0;
       
        hits = 0;
        for( k=trak->numhist-1; k>=klimit; k-- )  {
			 if( trak->detectflags_history[k] == WAS_DETECTED )  hits++;
        }
                      
        if( hits >= trak->mfirm )  trak->status = FIRM;

    }
       
       
    //======== If the last "ndown" measurements were non-detections and
	//            the track is firmly established, then downgrade
	//            to a TROUBLED track but maintain its existance.
       
    if( trak->status == FIRM  &&  trak->numhist > 1 )  {
       
		klimit = trak->numhist - trak->ndown;
		if( klimit < 0 )  klimit = 0;

		misses = 0;
        for( k=trak->numhist-1; k>=klimit; k-- )  {
             if( trak->detectflags_history[k] == NOT_DETECTED )  misses++;
        }
           
        if( misses == trak->ndown )  trak->status = TROUBLED;

    }
       
       
    //======== Check the most recent "ndrop" historical detection flags
	//            to determine demotion of a FIRM or TROUBLED track
	//            to a CLOSED track.
       
    if( trak->status == FIRM  ||  trak->status == TROUBLED )  {
       
		klimit = trak->numhist - trak->ndrop;
		if( klimit < 0 )  klimit = 0;

		misses = 0;
        for( k=trak->numhist-1; k>=klimit; k-- )  {
             if( trak->detectflags_history[k] == NOT_DETECTED )  misses++;
        }
           
        if( misses >= trak->mdrop )  trak->status = CLOSED;

    }
       
          
    //======== If this is the first detection, upgrade this new track to SPINUP status 
       
    if( trak->status == NOTRACK  &&  trak->detectflags_history[trak->numhist-1] == WAS_DETECTED )  {
		
		trak->status = SPINUP;

	}
       

	//======== If the last "ndown" measurements were non-detections
	//            and there was never a firm track established,
	//            then downgrade to NOTRACK.
       
    if( trak->status == SPINUP  &&  trak->numhist > 1 )  {
       
		klimit = trak->numhist - trak->ndown;
		if( klimit < 0 )  klimit = 0;

		misses = 0;
        for( k=trak->numhist-1; k>=klimit; k-- )  {
             if( trak->detectflags_history[k] == NOT_DETECTED )  misses++;
        }
           
        if( misses == trak->ndown )  trak->status = NOTRACK;

    }
      
       

}

						  
//###################################################################################
// The AlphaBeta function builds tracker coeficients based on a constant
// linear motion assumption with no acceleration.
//
//===================================================================================

void    AlphaBeta( long n, double *alpha_ptr, double *beta_ptr )
{
double  en, denom;

       //======== Compute alpha-beta tracker coefficients for a constant velocity target

       if( n < 15 )  en = (double)n;  //... Number of measurements
       else          en = 15.0;
       
       denom = en * ( en + 1.0 );
       
       *alpha_ptr = 2.0 * ( 2.0 * en - 1.0 ) / denom;
       
       *beta_ptr  = 6.0 / denom;

}


//###################################################################################
// The TrackingUpdate function takes an individual measurement and associated track
// and propagates the track forward in time using a coefficient set tied to a
// constant velocity alpha-beta tracker (the coefs are based on the number of 
// "detected" measurements in track thus far). If the detection flag is a 
// non-detection, then the track is "coasted" forward to the specified time. A 
// historical record of detection flags (YES/NO) is maintained and the track's 
// status is updated as a final step based on that history.
//
// Note time is passed in in case a tracking update needs to be performed and 
// there are no associated measurements (thus time would not get passed through
// the measurement attributeinfo structure).
//
//  DEPENDENCIES:
//     TrackingSetup,  AlphaBeta,  TrackingStatus
//
//===================================================================================

void   TrackingUpdate( long    detectflag,
	                   struct  attributeinfo  *measurement,
					   double  time,
                       struct  trackerinfo    *trak )
{
double   dtimeupdate, dtimemeas, alpha, beta;


       //======== Close the trak if we are going to exceed historical vector storage length
       //            Note that nummeas is always <= numhist

       if( trak->numhist >= trak->maxhistory )  trak->status = CLOSED;

	   if( trak->status == CLOSED )  return;


       //======== Reset track if a new track is being started with a detection. 
       //         Cannot start up a new track when there is no measurement.

       if( trak->status == NOTRACK  &&  detectflag == WAS_DETECTED )  TrackingSetup( trak );

       if( trak->status == NOTRACK  &&  detectflag == NOT_DETECTED )  return;



	   //======== Compute time differences between the current time and the last tracker update, 
	   //           as well as the last measurement fed the tracker that was "detected"
       
       dtimeupdate = time - trak->timeupdate;  // Delta time since last tracker update 
       dtimemeas   = time - trak->timemeas;    // Delta time since last "detected" measurement


	   //======== Save the detection flag history and increment counter

	   trak->detectflags_history[trak->numhist] = detectflag; 

       trak->numhist++;

       
	   //======== If a detection, save all the measurement attribute information, 
	   //           increment the detected measurement counter fed to the tracker, 
	   //           set the alpha-beta tracker coefficients, 
	   //           and update the track's filters based on number of 
	   //           detected measurements thus far.

	   if( detectflag == WAS_DETECTED )  {

		   trak->detected[ trak->nummeas ] = *measurement; //... should include time

           trak->nummeas++;

	       AlphaBeta( trak->nummeas, &alpha, &beta );


		   //-------- First measurement for this track
       
           if( trak->nummeas == 1 )  {  
           
               trak->rowmeas = measurement->rowcentroid;
               trak->rowpred = measurement->rowcentroid;
               trak->rowrate =  0.0f;
               trak->rowfilt = measurement->rowcentroid;
                         
               trak->colmeas = measurement->colcentroid;
               trak->colpred = measurement->colcentroid;
               trak->colrate =  0.0f;
               trak->colfilt = measurement->colcentroid;
           }
           

		   //-------- Second measurement for this track
       
           else if( trak->nummeas == 2 )  { 
                         
               trak->rowmeas = measurement->rowcentroid;
               trak->rowpred = trak->rowfilt + dtimeupdate * trak->rowrate;
               trak->rowrate =                         ( measurement->rowcentroid - trak->rowfilt ) / dtimemeas;
               trak->rowfilt = trak->rowpred + alpha * ( measurement->rowcentroid - trak->rowpred );
                         
               trak->colmeas = measurement->colcentroid;
               trak->colpred = trak->colfilt + dtimeupdate * trak->colrate;
               trak->colrate =                         ( measurement->colcentroid - trak->colfilt ) / dtimemeas;
               trak->colfilt = trak->colpred + alpha * ( measurement->colcentroid - trak->colpred );
           }

		   
		   //-------- Third or later measurement count for this track

           else  { 
                         
               trak->rowmeas = measurement->rowcentroid;
               trak->rowpred = trak->rowfilt + dtimeupdate * trak->rowrate;
               trak->rowrate = trak->rowrate + beta  * ( measurement->rowcentroid - trak->rowpred ) / dtimemeas;
               trak->rowfilt = trak->rowpred + alpha * ( measurement->rowcentroid - trak->rowpred );
                         
               trak->colmeas = measurement->colcentroid;
               trak->colpred = trak->colfilt + dtimeupdate * trak->colrate;
               trak->colrate = trak->colrate + beta  * ( measurement->colcentroid - trak->colpred ) / dtimemeas;
               trak->colfilt = trak->colpred + alpha * ( measurement->colcentroid - trak->colpred );
           }

		   //-------- Update the tracker time and the last "detected" measurment time

		   trak->timeupdate = time;
		   trak->timemeas   = time;

       }


	   //======== Not a detection for this track, so propagate the track in a  
	   //            "coasting" mode to the specified new time. Note that this
	   //            function's entry tests already narrowed the possibilities 
	   //            of this track being only SPINUP, FIRM, or TROUBLED.

	   else  {

           trak->rowpred = trak->rowfilt + dtimeupdate * trak->rowrate;
           trak->rowfilt = trak->rowpred;
                         
           trak->colpred = trak->colfilt + dtimeupdate * trak->colrate;
           trak->colfilt = trak->colpred;


	       //-------- Update only the tracker time (not the last measurement time)

	       trak->timeupdate = time;

	   }


	   //======== Update the track status flag based on the detection flag sequence

	   TrackingStatus( trak );

             
}


//###################################################################################
// The TrackingAllMeasurements function is the main interface for the user to feed
// new measurements to the trackers. It is required that ALL measurements for the
// specified "time" be sent to this function in a single call. The reason is because 
// this function propagates ALL active tracks forward to "time" and thus later calls 
// to this function for the same "time" will not track correctly. 
// All the attributes for every associated measurement is passed through for each
// measurement to the output track sequence in the trackerinfo structure. Association 
// of measurements to active tracks is based on a spatial distance tolerance relative
// to each track's forward propagation since its track update.
//
// This function would need to be called twice for a given frame if the collection 
// mode is interleaved. Once for the odd field and then again for the even field with
// time stamp appropriate to the field time (it may be even followed by odd depending
// on the collection sensor system).
//
// Note the assigned time for this set of measurements is passed as a separate argument
// rather than through the attributeinfo structure element "time" to allow proper tracker
// updates if the number of measurements passed was zero.
//
// INPUTS:
//     nmeas        Number of measurements in rowmeas and colmeas
//    *measurements Pointer to the attribute structure containing the measurements
//     time         Time of this set of measurements
//     dtolerance   Spatial proximty to associate measurements to tracks (pixels)
//    *trackers     Pointer to the maxtracks length vector of all tracks
//
// OUTPUT:
//    *trackers     Pointer to all tracks (all updated to "time")
//
//  DEPENDENCIES:
//     TrackingAssociation,  TrackingUpdate
//
//===================================================================================

void    TrackingAllMeasurements( long   nmeas,
	                             struct attributeinfo  *measurements,
							     double time_of_measurements,
	                             double dtolerance,
	                             struct trackerinfo  *trackers )
{
long                 kmeas, ktrack, maxtracks, num_assoc, num_notrack;
unsigned char       *measurement_assoc;
double              *rowmeas, *colmeas;
struct trackerinfo *trak;


    //======== Allocate temporary memory for this function's internal measurement "associated" flag.
    //            Initialize all flags to zero (not associated thus far).

    num_assoc = nmeas;
	if( num_assoc < 2 )  num_assoc = 2; //... min of 2 to ensure malloc allocates some memory

	measurement_assoc = (unsigned char*) malloc( num_assoc * sizeof( unsigned char ) );
	rowmeas           =        (double*) malloc( num_assoc * sizeof( double ) );
	colmeas           =        (double*) malloc( num_assoc * sizeof( double ) );

	memset( measurement_assoc, NOT_ASSOCIATED, num_assoc );

	//-------- Fill the centroid measurement vectors. The calls to TrackingUpdate
	//            will pass the time in case there are no measuremenst associated
	//            to a track update, but normally the measurements attributeinfo
	//            structure passes the time to the detected history and the
	//            assignment of the time must be made here.

	for( kmeas=0; kmeas<nmeas; kmeas++ )  {
	     rowmeas[kmeas] = measurements[kmeas].rowcentroid;
		 colmeas[kmeas] = measurements[kmeas].colcentroid;

		 measurements[kmeas].time = time_of_measurements;
	}

    //======== Check each track against the suite of measurements for this time, and associate
    //           the closest measurement to a given active track. Note any given measurement can 
	//           associate to more than one active track to handle close neighbor tracks.

	trak = &trackers[0];

	maxtracks = trak->maxtracks;

	for( ktrack=0; ktrack<maxtracks; ktrack++ )  {

		 trak = &trackers[ktrack];

		 //-------- Skip if this track is NOT an active track

		 if( trak->status == NOTRACK  ||  trak->status == CLOSED )  continue;

		 //-------- Check all the measurements and extract the measurement closest to the 
		 //            track's projection to "time" and within the user specified 
		 //            tolerance distance.

		 kmeas = TrackingAssociation( nmeas, rowmeas, colmeas, time_of_measurements, dtolerance, trak );

		 //-------- Skip to next active track if there was no measurement associated to this track

		 if( kmeas < 0 )  continue;

		 //-------- Measurement associated, so update tracking filter with a detection

         TrackingUpdate( WAS_DETECTED, &measurements[kmeas], time_of_measurements, trak );

		 //-------- Keep track of those measurements associated to tracks

		 if( trak->status >= FIRM )  measurement_assoc[kmeas] = WAS_ASSOCIATED; 


	} //... end of loop over over all tracks


    //======== Check all the measurements (assumed detections) but have NOT been 
    //            associated to any existing tracks, to start new track(s).

    for( kmeas=0; kmeas<nmeas; kmeas++ )  {

		 //-------- Skip if the measurement has been used (track associated)

		 if( measurement_assoc[kmeas] == WAS_ASSOCIATED    )  continue; 

		 //-------- Find an available tracker that has a status of NOTRACK

		 for( ktrack=0; ktrack<maxtracks; ktrack++ )  {

			  trak = &trackers[ktrack];

			  if( trak->status == NOTRACK )  {

				  //........ Initiate a new track with this unassociated measurement and flag the history
				  //             as a detection.

                  TrackingUpdate( WAS_DETECTED, &measurements[kmeas], time_of_measurements, trak );

				  break;  //... move on to the next measurement

			  }

		 }

	} //... end of loop over measurements that need new tracks


	//======== Update all the tracks to the same 1st measurement time 
	//             (essentially all those tracks that do not have an 
	//             associated detection at this time indicated by their
	//             last "timeupdate" being earlier than "time")

	trak = &trackers[0];

	maxtracks = trak->maxtracks;

	num_notrack = 0;

	for( ktrack=0; ktrack<maxtracks; ktrack++ )  {

	     trak = &trackers[ktrack];

		 if( trak->status == NOTRACK )  num_notrack++;

		 if( trak->status == NOTRACK  ||  trak->status == CLOSED )  continue;

		 if( trak->timeupdate < time_of_measurements )  TrackingUpdate( NOT_DETECTED, &measurements[0], time_of_measurements, trak );

	} //... end of loop over all tracks


	if( num_notrack == 0 )  printf(" WARNING: Run out of available trackers in TrackingFunctions.h \n");


	//======== Free the memory for measurement associated flags

	free( measurement_assoc );

}

/*
//###################################################################################
//  The TrackingLinearity function estimates the linear fit and distance
//  error between a 2D line and a series of "n" measurements in vectors "x"
//  and "y". This is based on minimizing the "perpendicular" distance
//  between the measurements and a straight line. The output product is
//  "multi-linearity" which is the average perpendicular distance from the
//  measurements to the line parameterization defined by the y-intercept
//  "ycept" and slope "mslope" equal to the change in y over change in x.
//
//  Model  y = ycept + mslope * x
//
// INPUTS:
//     track        Pointer to the trackinfo structure
//       .nmeas         Number of measurements in this track
//       .measurements  Measurements of row and column centroids
//       .time          Time of this set of measurements
//
// OUTPUT:
//     track.multi_linearity    
//
//===================================================================================

double   TrackingLinearity( struct  trackerinfo  *trak )
{
long      kmeas;
double    en, sumc, sumr, sumcc, sumcr, sumrr, meanc, meanr, denom;
double    q, bp, bm, dp, dm, ap, am, mep, mem, linearity;
  

    //======== Sufficnet points to estimate a line fit ?

    if( trak->nummeas < 2 )  return( 0.0 );


    //======== Compute the running sums for the linearity fit

	en = (double)trak->nummeas;

    sumc  = 0.0;
    sumr  = 0.0;
	sumcc = 0.0;
	sumcr = 0.0;
	sumrr = 0.0;

	for( kmeas=0; kmeas<trak->nummeas; kmeas++ )  {
		sumc  += trak->detected[kmeas].colcentroid;
		sumr  += trak->detected[kmeas].rowcentroid;
		sumcc += trak->detected[kmeas].colcentroid * trak->detected[kmeas].colcentroid;
		sumcr += trak->detected[kmeas].colcentroid * trak->detected[kmeas].rowcentroid;
		sumrr += trak->detected[kmeas].rowcentroid * trak->detected[kmeas].rowcentroid;
    }


    meanc = sumc / en;
    meanr = sumr / en;

    denom = en * meanc * meanr - sumcr;

    if( denom == 0.0 )  return( 0.0 );

    q = 0.5 * ( (sumrr - en * meanr * meanr) - (sumcc - en * meanc * meanc) ) / denom;

    bp = -q + sqrt( q*q + 1.0 );
    bm = -q - sqrt( q*q + 1.0 );

    dp = sqrt( 1.0 + bp * bp );
    dm = sqrt( 1.0 + bm * bm );

    ap = meanr - bp * meanc;
    am = meanr - bm * meanc;

	mep = 0.0;
	mem = 0.0;

	for( kmeas=0; kmeas<trak->nummeas; kmeas++ )  {
		mep += fabs( trak->detected[kmeas].rowcentroid - ap - bp * trak->detected[kmeas].colcentroid ) / dp;
		mem += fabs( trak->detected[kmeas].rowcentroid - am - bm * trak->detected[kmeas].colcentroid ) / dm;
	}
		
    if( mep < mem )  linearity = mep / en;
    else             linearity = mem / en;

	return( linearity ); 

}
						   
						   
//###################################################################################
//  The TrackingVelocity function estimates the linear motion fit of
//  a series of "n" measurements given by vectors of times "t", and the
//  positions "x" and "y". The fit is found by minimizing the distance
//  between all measurements and a model of points spaced proportionally to
//  time (constant velocity assumption). The input data does NOT have to be
//  collected at a uniform time spacing, as this function reports on the
//  uniformity of speed across either evenly or unevenly spaced
//  measurements in time.
//
//  Model  c(k) = colpos + t(k) * cvel    k = 1, 2, ... n
//         r(k) = rowpos + t(k) * rvel    n = # measurement points
//
// INPUTS:
//     track        Pointer to the trackinfo structure containing:
//       .nummeas         Number of measurements in this track
//       ->*.time         Time
//       ->*.rowcentroid  Row centroids
//       ->*.colcentroid  Col centroids
//
// OUTPUT:
//     velocity    Best fit straight line model velocity    
//
//===================================================================================


double   TrackingVelocity( struct  trackerinfo  *trak )
{
long      kmeas;
double    en, sumt, sumc, sumr, sumtt, sumtc, sumtr, denom, rvel, cvel, velocity;
  

    //======== Sufficnet points to estimate a line fit ?

    if( trak->nummeas < 2 )  return( 0.0 );


    //======== Compute the running sums for the linearity fit

	en = (double)trak->nummeas;

    sumt  = 0.0;
    sumc  = 0.0;
    sumr  = 0.0;
	sumtt = 0.0;
	sumtc = 0.0;
	sumtr = 0.0;

	for( kmeas=0; kmeas<trak->nummeas; kmeas++ )  {
		sumt  += trak->detected[kmeas].time;
		sumc  += trak->detected[kmeas].colcentroid;
		sumr  += trak->detected[kmeas].rowcentroid;
		sumtt += trak->detected[kmeas].time * trak->detected[kmeas].time;
		sumtc += trak->detected[kmeas].time * trak->detected[kmeas].colcentroid;
		sumtr += trak->detected[kmeas].time * trak->detected[kmeas].rowcentroid;
    }

	denom = sumt * sumt - en * sumtt;

    if( denom == 0.0 )  return( 0.0 );

    cvel = ( sumc * sumt - en * sumtc ) / denom;
    rvel = ( sumr * sumt - en * sumtr ) / denom;

	velocity = sqrt( cvel * cvel + rvel * rvel );

	return( velocity ); 

}
*/

//###################################################################################
//  The TrackingLinearModelFit function estimates the linear motion fit of a
//  series of "n" measurements given by vectors of "time", and the centroid
//  positions for "row" and "column". The fit is found by minimizing the distance
//  between all the given measurements and a model of points spaced proportionally
//  to the time spacing (assuming a constant velocity assumption). The input data 
//  does NOT have to be collected at a uniform time spacing. This function reports
//  on the uniformity of speed across either evenly or unevenly spaced temporal
//  measurements. It also provides the model starting position and model velocities
//  in row and column relative to the first time point, along with a measure of 
//  non-linearity to a straight line.
//
//  Model:  colpos(k) = colstart + [ t(k) - t(1) ] * colvel    k = 1, 2, ... n
//          rowpos(k) = rowstart + [ t(k) - t(1) ] * rowvel    n = # measurement points
//
// INPUTS:
//     track        Pointer to the trackinfo structure containing:
//       .nummeas         Number of measurements in this track
//       ->*.time         Time stamp of each measurement
//       ->*.rowcentroid  Row centroid of each measurement
//       ->*.colcentroid  Col centroid of each measurement
//
// OUTPUT:
//     track        Pointer to the trackinfo structure containing:
//       .rowstart        Starting row position of the best fit model 
//       .colstart        Starting col position of the best fit model 
//       .rowspeed        Row velocity of the best fit model 
//       .colspeed        Col velocity of the best fit model
//       .linearity       Mean distance error between model and straight line
//       .modelerr        Mean distance error between model points and measurements
//
//===================================================================================


void     TrackingLinearModelFit( struct  trackerinfo  *trak )
{
long      kmeas;
double    en, sumt, sumc, sumr, sumtt, sumtc, sumtr, denom, drow, dcol, dtime;
double    row1, row2, col1, col2, crossterm;
  

    //======== Sufficient points to estimate a line fit ?

	trak->multi_rowstart  = trak->detected[0].rowcentroid;
	trak->multi_colstart  = trak->detected[0].colcentroid;
	trak->multi_rowspeed  = 0.0;
	trak->multi_colspeed  = 0.0;
	trak->multi_linearity = 99999.0;
	trak->multi_modelerr  = 99999.0;

    if( trak->nummeas < 2 )  return;


    //======== Compute the running sums for the linear motion fit

	en = (double)trak->nummeas;

    sumt  = 0.0;
    sumc  = 0.0;
    sumr  = 0.0;
	sumtt = 0.0;
	sumtc = 0.0;
	sumtr = 0.0;

	for( kmeas=0; kmeas<trak->nummeas; kmeas++ )  {
		dtime  = trak->detected[kmeas].time - trak->detected[0].time;
		sumt  += dtime;
		sumr  += trak->detected[kmeas].rowcentroid;
		sumc  += trak->detected[kmeas].colcentroid;
		sumtt += dtime * dtime;
		sumtr += dtime * trak->detected[kmeas].rowcentroid;
		sumtc += dtime * trak->detected[kmeas].colcentroid;
    }

	denom = sumt * sumt - en * sumtt;

    if( denom == 0.0 )  return;

    trak->multi_rowspeed = ( sumr * sumt - en * sumtr ) / denom;
    trak->multi_colspeed = ( sumc * sumt - en * sumtc ) / denom;

    trak->multi_rowstart = ( sumr - trak->multi_rowspeed * sumt ) / en;
    trak->multi_colstart = ( sumc - trak->multi_colspeed * sumt ) / en;


	//-------- Compute the average distance of the measurements from the model fit

	sumt = 0.0;

	for( kmeas=0; kmeas<trak->nummeas; kmeas++ )  {
		 dtime = trak->detected[kmeas].time - trak->detected[0].time;
		 drow  = trak->detected[kmeas].rowcentroid - trak->multi_rowstart - dtime * trak->multi_rowspeed;
		 dcol  = trak->detected[kmeas].colcentroid - trak->multi_colstart - dtime * trak->multi_colspeed;
		 sumt += sqrt( drow * drow + dcol * dcol );
    }	

	trak->multi_modelerr = sumt / en;


	//-------- Compute the perpendicular distance of the measurements from the best line fit

 	sumt = 0.0;

	row1 = trak->multi_rowstart;
	col1 = trak->multi_colstart;

	dtime = trak->detected[trak->nummeas-1].time - trak->detected[0].time;

	row2 = trak->multi_rowstart + dtime * trak->multi_rowspeed;
	col2 = trak->multi_colstart + dtime * trak->multi_colspeed;

	drow = row2 - row1;
	dcol = col2 - col1;

	crossterm = row2 * col1 - col2 * row1;

	denom = sqrt( drow * drow + dcol * dcol );

	for( kmeas=0; kmeas<trak->nummeas; kmeas++ )  {
		 sumt += fabs( dcol * trak->detected[kmeas].rowcentroid - drow * trak->detected[kmeas].colcentroid + crossterm ) / denom;
    }	
	
	trak->multi_linearity = sumt / en;

}


//###################################################################################
// The TrackingSaver function is called to identify any CLOSED tracks in the full
// trackers list and extract 1 track into the "savedtrak" structure for further user 
// post-processing. The extracted track is then released back to the trackers pool 
// of tracks with a NOTRACK status. This function extracts ONLY one track each
// call so must be called multiple times to clear out several CLOSED tracks as in
// the code example below. The returned value is zero for no tracks extracted and 
// unity for track found with CLOSED status, copied to savedtrak, and reset to a
// NOTRACK status.
//
//    struct trackerinfo  track;
//
//    while( TrackingSaver( trackers, &track ) == 1 )  {  do stuff with "track"  }
//
// Various multi-frame track parameters are also computed in this function for user
// post-processing and false alarm reduction.
//
//===================================================================================


long   TrackingSaver( struct  trackerinfo  *trackers,  
                      struct  trackerinfo  *savedtrak )
{
long                 ktrack, maxtracks, knumer, kdenom, kmeas;
double               signal, noise, xsum, ysum, esum, SNR;
struct trackerinfo  *trak;

    
    trak = &trackers[0];

    maxtracks = trak->maxtracks;

	//======== Loop over all tracks searching for the first CLOSED track to save

	for( ktrack=0; ktrack<maxtracks; ktrack++ )  {

		 trak = &trackers[ktrack];

		 //-------- If any track status is CLOSED, compute some multi-frame parameters,
		 //            and then copy the track to a saved track and return.
		 //            Note that a CLOSED status implies it was once a FIRM track.

		 if( trak->status == CLOSED )  {

			 //...... Compute multi-frame averaged parameters

			 signal = 0.0;
			 noise  = 0.0;
			 xsum   = 0.0;
			 ysum   = 0.0;
			 esum   = 0.0;
			 knumer = 0;
			 kdenom = 0;

			 for( kmeas=0; kmeas<trak->nummeas; kmeas++ )  {

				  signal += trak->detected[kmeas].sumsignal;
				  noise  += trak->detected[kmeas].sumnoise;

				  xsum   += cos( trak->detected[kmeas].PCD_angle / 57.29577951 );
				  ysum   += sin( trak->detected[kmeas].PCD_angle / 57.29577951 );

				  esum   += trak->detected[kmeas].entropy;

				  knumer += trak->detected[kmeas].nexceed;
				  kdenom += trak->detected[kmeas].npixels;

			 }

			 if( noise != 0.0 )  SNR = signal * signal / noise;
			 else                SNR = 0.0;

			 trak->multi_SNRdB = 10.0 * log10( SNR + 1.0e-30 );


			 if( kdenom != 0 )  SNR = (double)knumer / (double)kdenom;
			 else               SNR = 0.0;

			 trak->multi_BNRdB = 10.0 * log10( SNR + 1.0e-30 );


			 trak->multi_angle = 57.29577951 * atan2( ysum, xsum );  //...deg


			 trak->multi_entropy = esum / (double)trak->nummeas;


			 //...... Fill in some multi-frame linear-motion fitting parameters

			 TrackingLinearModelFit( trak );


			 //...... trak copied to output trackerinfo structure (savedtrak)

             *savedtrak = *trak;      

	         savedtrak->status = CLOSED;

	         trak->status = NOTRACK;  //... "trackers" trak released back to the pool of tracks

			 TrackingSetup( trak );

			 return( 1 );  //... THIS WILL EXTRACT ONLY ONE TRACK PER TrackingSaver CALL

		 }

	}


	//======== No further tracks found in a CLOSED status category that could be saved

	return( 0 );  

}
						   
						  
//###################################################################################
// The TrackingScale function is called to scale a track products by a decimation
// factor in the event the imagery was processed at a decimated resolution. All
// row and column positional and velocity information gets scaled.
//
//===================================================================================

void   TrackingScale( struct  trackerinfo  *trak, double decimation_factor )
{
int k;

	trak->colfilt *= decimation_factor;
	trak->colmeas *= decimation_factor;
	trak->colpred *= decimation_factor;
	trak->colrate *= decimation_factor;

	trak->rowfilt *= decimation_factor;
	trak->rowmeas *= decimation_factor;
	trak->rowpred *= decimation_factor;
	trak->rowrate *= decimation_factor;

	trak->multi_colstart *= decimation_factor;
	trak->multi_colspeed *= decimation_factor;

	trak->multi_rowstart *= decimation_factor;
	trak->multi_rowspeed *= decimation_factor;

	for( k=0; k<trak->nummeas; k++ )  {
	     trak->detected[k].colcentroid *= decimation_factor;
	     trak->detected[k].rowcentroid *= decimation_factor;
	}

}

//###################################################################################
// The TrackingCopy function moves the contents of one trackerinfo structure to
// a second. We cannot do a memcpy or trak1 = trak2 as these are "shallow" copies
// which do not handle embedded pointers correctly (which these structures contain).
//
//===================================================================================

void   TrackingCopy( struct  trackerinfo  *trak1, struct  trackerinfo  *trak2 )
{
int k;

    trak2->maxtracks       = trak1->maxtracks;
    trak2->maxhistory      = trak1->maxhistory;
    trak2->status          = trak1->status;
    trak2->numhist         = trak1->numhist;
    trak2->nummeas         = trak1->nummeas;
    trak2->mfirm           = trak1->mfirm;
    trak2->nfirm           = trak1->nfirm;
    trak2->mdrop           = trak1->mdrop;
    trak2->ndrop           = trak1->ndrop;
    trak2->ndown           = trak1->ndown;
    trak2->alpha           = trak1->alpha;
    trak2->beta            = trak1->beta;
    trak2->timeupdate      = trak1->timeupdate;
    trak2->timemeas        = trak1->timemeas;
    trak2->rowmeas         = trak1->rowmeas;
    trak2->rowfilt         = trak1->rowfilt;
    trak2->rowrate         = trak1->rowrate;
    trak2->rowpred         = trak1->rowpred;
    trak2->colmeas         = trak1->colmeas;
    trak2->colfilt         = trak1->colfilt;
    trak2->colrate         = trak1->colrate;
    trak2->colpred         = trak1->colpred;
    trak2->multi_SNRdB     = trak1->multi_SNRdB;
    trak2->multi_BNRdB     = trak1->multi_BNRdB;
    trak2->multi_entropy   = trak1->multi_entropy;
    trak2->multi_angle     = trak1->multi_angle;
    trak2->multi_rowstart  = trak1->multi_rowstart;
    trak2->multi_colstart  = trak1->multi_colstart;
    trak2->multi_rowspeed  = trak1->multi_rowspeed;
    trak2->multi_colspeed  = trak1->multi_colspeed;
    trak2->multi_linearity = trak1->multi_linearity;
    trak2->multi_modelerr  = trak1->multi_modelerr;


	for( k=0; k<trak1->maxhistory; k++ )  {
		
		trak2->detectflags_history[k]  = trak1->detectflags_history[k];

		trak2->detected[k].time          = trak1->detected[k].time;
		trak2->detected[k].rowcentroid   = trak1->detected[k].rowcentroid;
		trak2->detected[k].colcentroid   = trak1->detected[k].colcentroid;
		trak2->detected[k].nexceed       = trak1->detected[k].nexceed;
		trak2->detected[k].nsaturated    = trak1->detected[k].nsaturated;
		trak2->detected[k].npixels       = trak1->detected[k].npixels;
		trak2->detected[k].sumsignal     = trak1->detected[k].sumsignal;
		trak2->detected[k].sumnoise      = trak1->detected[k].sumnoise;
		trak2->detected[k].SsqNdB        = trak1->detected[k].SsqNdB;
		trak2->detected[k].entropy       = trak1->detected[k].entropy;
		trak2->detected[k].PCD_angle     = trak1->detected[k].PCD_angle;
		trak2->detected[k].PCD_width     = trak1->detected[k].PCD_width;
		trak2->detected[k].Hueckel_angle = trak1->detected[k].Hueckel_angle;

	}

}

//###################################################################################
// The TrackingSpeedLinearityTest function checks for speed range and linear motion.
//
// Returns zero outside the constraints, unity if the track meets all constraints.
//
//===================================================================================


int  TrackingSpeedLinearityTest( struct trackerinfo  *track,  
	                             double velocity_lowerlimit,  
						         double velocity_upperlimit,         
						         double linearity_threshold,  
						         double modelfit_threshold )
{
double  velocity;


    //======== Check the speed is within low and high velocity constraints

    velocity = sqrt( track->multi_colspeed * track->multi_colspeed + 
				     track->multi_rowspeed * track->multi_rowspeed );
			
	if( velocity < velocity_lowerlimit  ||  
        velocity > velocity_upperlimit     )  return( 0 );


    //======== Check the multi-frame track linearity and constant-speed model fit constraints

	if( track->multi_linearity > linearity_threshold  ||
		track->multi_modelerr  > modelfit_threshold       )  return( 0 );


	//======== For short tracks of 3 frames or less, use a tighter linearity and model fit constraint

	if( track->nummeas <= 3   &&  
		(  track->multi_linearity > linearity_threshold / 2.0  ||
		   track->multi_modelerr  > modelfit_threshold  / 2.0      ) )  return( 0 );


	//======== Track meets speed and linearity constraints
	
	return(1); 

}

