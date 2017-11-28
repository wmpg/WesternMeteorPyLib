//#########################################################################
//#########################################################################
//
//  EMCCD I/O Functions
//
//#########################################################################
//#########################################################################

#pragma warning(disable: 4996)  // disable warning on strcpy, fopen, ...

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


//*************************************************************************
//**************  EMCCD Detection Read/Write Functions  *******************
//*************************************************************************

//------------------------------- Global FILE descriptor

FILE    *CandidateDetectWriteFile;
FILE    *CandidateDetectReadFile;


//------------------------------- File name structure

struct  EMCCDfileinfo
{
	char     filename[256];
};


//------------------------------- Input parameter structure

#define  MAXPREV   9


struct  EMCCDparameters
{
	//--------------------------------- Background parameters -------------------------------------
	long       Plow;                 // Pixel gray level low  end cutoff percentage = typically 5%
	long       Phigh;                // Pixel gray level high end cutoff percentage = typically 50%

	//--------------------------------- Compression parameters ---------------------------------------
	long       nframes_compress;     // Number of frames per sequential compression block

	//--------------------------------- Clustering parameters ----------------------------------------
	long       maxclusters;          // Max number of clusters per frame
	long       interleave;           // Interleave flag (0=progressive scan, 1=odd/even, 2=even/odd)
    long       blocksize;            // Size of hierarchical cells for fast clustering
    long       ntuplet;              // Number of neighbors plus self in 5x5 region for a cluster
    double     tiny_variance;        // Smallest noise variance to avoid divide by zero
    double     saturation;           // Saturation level of the imaging system
    double     sfactor;              // Sigma multiplicative factor for pixel thresholds over mean
    double     SsqNdB_threshold;     // Detection level of cluster in dB (Signal^2/Variance)

	//--------------------------------- Tracking parameters ----------------------------------------
    long       maxtracks;            // Maximum number of tracks to monitor
    long       maxhistory;           // Maximum duration of tracks in #frames
    long       mfirm;                // "M" of N detected criteria to declare FIRM track
    long       nfirm;                // M of "N" detected criteria to declare FIRM track
    long       mdrop;                // "M" of N dropped criteria to declare CLOSED track
    long       ndrop;                // M of "N" dropped criteria to declare CLOSED track
    long       ndown;                // Number dropped in a row to declare TROUBLED track

	//--------------------------------- Detection culling parameters ------------------------------
	double     velocity_lowerlimit;  // Min pixels/frame threshold criteria on apparent angular speed
	double     velocity_upperlimit;  // Max pixels/frame threshold criteria on apparent angular speed
    double     linearity_threshold;  // Multi-measurement error offset from a straight line threshold
	double     modelfit_threshold;   // Multi-measurement error from propagating constant motion threshold

	//--------------------------------- Duplication constraint thresholds
	double     AngleLimitDegrees;    // Angle offset limit from parallel in degrees
	double     RelVelocityPercentage;// Relative velocity difference in percentage
	double     DistanceLimitPixels;  // Max distance of start point on line A from line B in pixels

	//--------------------------------- Non control (pass-thru) information
	char       imagery_pathname[256];     // Imagery full pathname for the multi-frame data
	char       flatfile_pathname[256];    // File pathname for the flat field data
	char       darkfile_pathname[256];    // File pathname for the flat field data
	long       cameranumber;              // Camera serial number = 256*site + cam#
	long       nrows;                     // Number of rows
	long       ncols;                     // Number of cols

	//--------------------------------- Site information
	char       sitename[128];             // Name of the camera site
	char       platefile[256];            // File name of the astrometric plate solution
	double     latitude_deg;              // Cameras site geodetic latitude in degrees (GPS latitude)
	double     longitude_deg;             // Cameras site longitude +east of Greenwich in degrees
	double     height_meters;             // Cameras site height above WGS84 Earth in meters (GPS height)

	double     version;                   // Version number of the software
	char       versiontext[64];           // Software descriptive text containing app name and version
	char       filetext[64];              // Imagery file descriptive text
	double     previous_time[MAXPREV];    // Previous times to nearest second for files written in ASGARD format
	double     sample_time_sec;           // Sampling time in seconds = 1 / fps
	long       psfminsize;                // PSF square kernel minimum size in pixels (odd#)
	double     psfhalfwidth;              // PSF Gaussian half-width for half-peak
	double     TRT_threshold;             // TRT threshold
	double     MLEdb_threshold;           // MLE threshold in dB
	long       totalmeteors;              // Number of candidate meteor tracks in the file
	long       reporting_option;          // Reporting option sum of:
	                                      //     1 = Cluster/tracker detections
	                                      //     2 = Matched filter  detections
	                                      //     4 = Matched filter  detections (de-duplicated)
	                                      //     8 = Generate BMP for any MF detection
	                                      //    16 = Generate ASGRAD file per detection
};


//------------------------------- Output parameters reporting structure

struct  EMCCDtrackdata
{
	double     time;          // Relative time of the measurement in seconds
	double     row;           // Row measurement position (leading edge)
	double     col;           // Row measurement position (leading edge)
	double     theta_deg;     // Zenith angle of measurement in deg
	double     phi_deg;       // Azimuth of measurement in deg (+north of east)
	double     RA_deg;        // Right ascension (deg)
	double     Dec_deg;       // Declination (deg)
	double     logsumsignal;  // -2.5 * log10 of the spatially-integrated, background-removed signal
	double     magnitude;     // Magnitude of the track segment
	double     background;    // Background pixel level (currently variance of background)
	double     maxpixel_aoi;  // Maximum pixel value in area of interest (integrated region)
};


//=========================================================================
//                    EMCCD_Read_Config_Parameters
//-------------------------------------------------------------------------
//
// Purpose:   Function to open and read the configuration parameters
//            file. All config parameters are passed out through the
//            EMCCDparameters structure.
//
// Inputs:    config_pathname    Pathname of configuration parameters file
//
// Outputs:   params             Pointer to the EMCCDparameters structure
//
//
//=========================================================================


void   EMCCD_Read_Config_Parameters( char* config_pathname,  struct EMCCDparameters *params )
{
int      k;
double   read_version;
char     text[256];
FILE    *ParametersReadFile;


	//======== Open the output detection log file

    if( ( ParametersReadFile = fopen( config_pathname, "r" ) ) == NULL )  {
        fprintf( stdout, "ERROR ===> Config file %s could not be opened\n\n", config_pathname );
        fprintf( stderr, "ERROR ===> Config file %s could not be opened\n\n", config_pathname );
		Delay_msec(30000);
        exit(1);
    }


	//======== Read the version and test for compatibility

	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &read_version );

    if( read_version != 0.10 )  {
        fprintf( stdout, "ERROR ===> Config file version %lf not valid for %s\n\n", read_version, config_pathname );
        fprintf( stderr, "ERROR ===> Config file version %lf not valid for %s\n\n", read_version, config_pathname );
		Delay_msec(30000);
        exit(1);
    }


	//======== Read the remainer of the configuration parametersr

	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->sample_time_sec );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->interleave );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->Plow );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->Phigh );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->nframes_compress );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->blocksize );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->maxclusters );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->ntuplet );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->tiny_variance );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->saturation );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->sfactor );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->SsqNdB_threshold );
	fscanf( ParametersReadFile, "%[^=]= %ld of %ld",  text, &params->mfirm, &params->nfirm );
	fscanf( ParametersReadFile, "%[^=]= %ld of %ld",  text, &params->mdrop, &params->ndrop );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->ndown );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->maxtracks );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->maxhistory );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->velocity_lowerlimit );  //... pixels/frame
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->velocity_upperlimit );  //... pixels/frame
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->linearity_threshold );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->modelfit_threshold );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->psfminsize );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->psfhalfwidth );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->AngleLimitDegrees );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->RelVelocityPercentage );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->DistanceLimitPixels );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->TRT_threshold );
	fscanf( ParametersReadFile, "%[^=]= %lf",         text, &params->MLEdb_threshold );
	fscanf( ParametersReadFile, "%[^=]= %ld",         text, &params->reporting_option );

	fclose( ParametersReadFile );

	for( k=0; k<MAXPREV; k++ )  params->previous_time[k] = 0.0;

}


//=========================================================================
//                    EMCCD_WriteDetect_OpenHeader
//-------------------------------------------------------------------------
//
// Purpose:   Function to open and write the header to a detection output
//            log file. All header parameters are passed through the
//            EMCCDparameters structure.
//
// Inputs:    logfile_pathname   Pathname of log file
//            params             Pointer to the EMCCDparameters structure
//
// Outputs:   Log file opened and header written
//
//
//=========================================================================


void   EMCCD_WriteDetect_OpenHeader( char* logfile_pathname,  struct EMCCDparameters *params )
{
int     sitenum, camnum;
time_t  processedtime;


	//======== Open the output detection log file

    if( ( CandidateDetectWriteFile = fopen( logfile_pathname, "wt" ) ) == NULL )  {
        fprintf( stdout, "ERROR ===> Log file %s not opened\n\n", logfile_pathname );
        fprintf( stderr, "ERROR ===> Log file %s not opened\n\n", logfile_pathname );
		Delay_msec(30000);
        exit(1);
    }


	//======== Do some calculations

	processedtime = time(NULL);

	sitenum = params->cameranumber / 256;
	camnum  = params->cameranumber % 256;


	//======== Write the header

    fprintf( CandidateDetectWriteFile, "Meteor Count = 000000\n" );
    fprintf( CandidateDetectWriteFile, "----------------------------------------------------------------------------------------------\n");
	fprintf( CandidateDetectWriteFile, "EMCCD Detection Processor  = %6.2lf run on %s", params->version, ctime( &processedtime ) );
    fprintf( CandidateDetectWriteFile, "----------------------------------------------------------------------------------------------\n");
	fprintf( CandidateDetectWriteFile, "Imagery pathname processed = %s\n",  params->imagery_pathname );
	fprintf( CandidateDetectWriteFile, "Flat field file applied    = %s\n",  params->flatfile_pathname );
	fprintf( CandidateDetectWriteFile, "Dark field file applied    = %s\n",  params->darkfile_pathname );
	fprintf( CandidateDetectWriteFile, "Camera serial number       = %02d%c\n", sitenum, camnum );
	fprintf( CandidateDetectWriteFile, "Number of rows             = %d\n",  params->nrows );
	fprintf( CandidateDetectWriteFile, "Number of cols             = %d\n",  params->ncols );
	fprintf( CandidateDetectWriteFile, "Sample time per frame      = %lf\n", params->sample_time_sec );
	fprintf( CandidateDetectWriteFile, "Interleave type            = %d\n",  params->interleave );
	fprintf( CandidateDetectWriteFile, "Backgnd low cutoff (%%)     = %d\n", params->Plow );
	fprintf( CandidateDetectWriteFile, "Backgnd high cutoff (%%)    = %d\n", params->Phigh );
	fprintf( CandidateDetectWriteFile, "Number compressed frames   = %d\n",  params->nframes_compress );
	fprintf( CandidateDetectWriteFile, "Cell block size (pixels)   = %d\n",  params->blocksize );
	fprintf( CandidateDetectWriteFile, "Max clusters per frame     = %d\n",  params->maxclusters );
	fprintf( CandidateDetectWriteFile, "Neighbors + self tuplet    = %d\n",  params->ntuplet );
	fprintf( CandidateDetectWriteFile, "Tiny variance              = %lf\n", params->tiny_variance );
	fprintf( CandidateDetectWriteFile, "Saturation gray level      = %lf\n", params->saturation );
	fprintf( CandidateDetectWriteFile, "Sigma factor               = %lf\n", params->sfactor );
	fprintf( CandidateDetectWriteFile, "Cluster detection (dB)     = %lf\n", params->SsqNdB_threshold );
	fprintf( CandidateDetectWriteFile, "Tracker FIRM               = %d of %d\n", params->mfirm, params->nfirm );
	fprintf( CandidateDetectWriteFile, "Tracker CLOSED             = %d of %d\n", params->mdrop, params->ndrop );
	fprintf( CandidateDetectWriteFile, "Tracker TROUBLED           = %d\n",  params->ndown );
	fprintf( CandidateDetectWriteFile, "Tracker max tracklets      = %d\n",  params->maxtracks );
	fprintf( CandidateDetectWriteFile, "Tracker max history        = %d\n",  params->maxhistory );
	fprintf( CandidateDetectWriteFile, "Detect Vmin (pix/frame)    = %lf\n", params->velocity_lowerlimit );
	fprintf( CandidateDetectWriteFile, "Detect Vmax (pix/frame)    = %lf\n", params->velocity_upperlimit );
	fprintf( CandidateDetectWriteFile, "Detect linearity           = %lf\n", params->linearity_threshold );
	fprintf( CandidateDetectWriteFile, "Detect model fit           = %lf\n", params->modelfit_threshold );
	fprintf( CandidateDetectWriteFile, "PSF min size (pixels)      = %d\n",  params->psfminsize );
	fprintf( CandidateDetectWriteFile, "PSF halfwidth (pixels)     = %lf\n", params->psfhalfwidth );
	fprintf( CandidateDetectWriteFile, "Dedupe angle limit (deg)   = %lf\n", params->AngleLimitDegrees );
	fprintf( CandidateDetectWriteFile, "Dedupe rel velocity (%%)    = %lf\n", params->RelVelocityPercentage );
	fprintf( CandidateDetectWriteFile, "Dedupe max distance (pix)  = %lf\n", params->DistanceLimitPixels );
	fprintf( CandidateDetectWriteFile, "TRT Threshold              = %lf\n", params->TRT_threshold );
	fprintf( CandidateDetectWriteFile, "MLE Threashold (dB)        = %lf\n", params->MLEdb_threshold );
    fprintf( CandidateDetectWriteFile, "----------------------------------------------------------------------------------------------\n");
    fprintf( CandidateDetectWriteFile, "Imagery_pathname   V=pixels/frame\n");
	fprintf( CandidateDetectWriteFile, "DetectID  Npts  BNRorTRT  SNRorMLE  Xcol  Xrow  Vcol  Vrow  Lin  Model  Ref#\n");
	fprintf( CandidateDetectWriteFile, "UnixSeconds  Julian  YYYYMMDD:HH:MM:SS.MSC:UTC\n");
	fprintf( CandidateDetectWriteFile, "Seq#  RelativeTimeSec  Column  Row  Theta  Phi  RA  Dec  logI  Mag\n");

}


//=========================================================================
//                    EMCCD_WriteDetect_Tracklet
//-------------------------------------------------------------------------
//
// Purpose:   Function to write a candidate meteor track to a detection
//            output log file. All track parameters are passed through the
//            trackinfo structure.
//
// Inputs:    detectID        Detection algorithm identifier
//                                "CLUS" cluster/tracker
//                                "MDUP" = not de-duplicated MF detection
//                                "MFIL" matched filter
//            pathname        File pathname of the imagery sequence
//            track           Pointer to the trackerinfo structure
//            seqnumber       Vector of sequence numbers per measurement
//            sample_time_sec Time spacing betwen adjacent-in-time frame (seconds)
//            trackmeasurements     Pointer to the EMCCDtrackdata structure
//            refnumber       Reference number to relate back to a BMP file
//
// Outputs:   Log file appended with track information
//
//
//=========================================================================


void   EMCCD_WriteDetect_Tracklet( char* detectID, char* pathname, struct trackerinfo *track, double *seqnumber, double sample_time_sec, struct EMCCDtrackdata *trackmeasurements, int refnumber )
{
int     k;
long    year, month, day, hour, minute, second, milliseconds;
double  jdt_unixstart, jdt_firsttrackmeas, time_offset_seconds;
double  rowdelta_leadingedge, coldelta_leadingedge;


    //======== Calendar date and time extraction

    jdt_unixstart = 2440587.5;  //... Julian date for January 1, 1970  0:00:00 UTC

	jdt_firsttrackmeas = jdt_unixstart  +  track->detected[0].time / 86400.0;  //  .time is seconds since unix start

    jdt_firsttrackmeas += sample_time_sec / 86400.0;  // "leading edge" biased forward by the frame time

	CalendarDateAndTime( jdt_firsttrackmeas, &year, &month, &day, &hour, &minute, &second, &milliseconds );


	//======== report on the "leading edge" for the row and column positions and associated coords metrics

	rowdelta_leadingedge = track->multi_rowspeed * sample_time_sec / 2.0;
	coldelta_leadingedge = track->multi_colspeed * sample_time_sec / 2.0;


    fprintf( CandidateDetectWriteFile, "----------------------------------------------------------------------------------------------\n");

	fprintf( CandidateDetectWriteFile,"%s\n%s %03li %6.2lf %6.2lf %8.2lf %8.2lf %6.2lf %6.2lf %7.4lf %7.4lf %04d\n",
			 pathname,
	         detectID,
			 track->nummeas,
			 track->multi_BNRdB,
			 track->multi_SNRdB,
			 track->multi_colstart + coldelta_leadingedge,
             track->multi_rowstart + rowdelta_leadingedge,
			 track->multi_colspeed * sample_time_sec,
			 track->multi_rowspeed * sample_time_sec,
			 track->multi_linearity,
			 track->multi_modelerr,
			 refnumber );

	fprintf( CandidateDetectWriteFile,"%.3lf  %15.6lf  %04d%02d%02d:%02d:%02d:%02d.%03d:UTC\n",
		                               track->detected[0].time + sample_time_sec,
									   jdt_firsttrackmeas,
		                               year, month, day, hour, minute, second, milliseconds );

    for( k=0; k<track->nummeas; k++ )  {

		 time_offset_seconds = track->detected[k].time - track->detected[0].time;

		 if( trackmeasurements == NULL )  {  //... CLUS detection report

             fprintf( CandidateDetectWriteFile, "%011.3lf  %07.3lf  %08.2lf  %08.2lf\n",
			          seqnumber[k],
				      time_offset_seconds,
	                  track->detected[k].colcentroid + coldelta_leadingedge,
	                  track->detected[k].rowcentroid + rowdelta_leadingedge  );

		 }
		 else  {     //... MDUP and MFIL detection report

             fprintf( CandidateDetectWriteFile, "%011.3lf  %07.3lf  %08.2lf  %08.2lf  %07.3lf  %07.3lf  %07.3lf  %07.3lf  %07.3lf  %07.3lf\n",
			          seqnumber[k],
				      time_offset_seconds,
	                  track->detected[k].colcentroid + coldelta_leadingedge,
	                  track->detected[k].rowcentroid + rowdelta_leadingedge,
				      trackmeasurements[k].theta_deg,
				      trackmeasurements[k].phi_deg,
				      trackmeasurements[k].RA_deg,
				      trackmeasurements[k].Dec_deg,
				      trackmeasurements[k].logsumsignal,
				      trackmeasurements[k].magnitude     );
		 }

	} //... end measurement loop

}

//=========================================================================
//                    EMCCD_WriteDetect_MeteorCount
//-------------------------------------------------------------------------

void   EMCCD_WriteDetect_MeteorCount( long totalmeteors )
{

	fseek( CandidateDetectWriteFile, 0L, SEEK_SET );  //...reset to beginning of file

	fprintf( CandidateDetectWriteFile, "Meteor Count = %06li\n", totalmeteors );

}

//=========================================================================
//                    EMCCD_WriteDetect_Close
//-------------------------------------------------------------------------

void   EMCCD_WriteDetect_Close( )
{
	fclose( CandidateDetectWriteFile );
}

//=========================================================================



//=========================================================================
//                    EMCCD_ReadDetect_OpenHeader
//-------------------------------------------------------------------------
//
// Purpose:   Function to open and read the header of a detection log
//            file. All header parameters are passed out through the
//            EMCCDparameters structure.
//
// Inputs:    logfile_pathname   Pathname of log file
//
// Outputs:   params             Pointer to the EMCCDparameters structure
//
//=========================================================================


void   EMCCD_ReadDetect_OpenHeader( char* logfile_pathname,  struct EMCCDparameters *params )
{
long   sitenum;
char   camchar, text[256], datetimetext[256];


	//======== Open the input detection log file

    if( ( CandidateDetectReadFile = fopen( logfile_pathname, "r" ) ) == NULL )  {
        fprintf( stdout, "ERROR ===> Log file %s could not be opened\n\n", logfile_pathname );
        fprintf( stderr, "ERROR ===> Log file %s could not be opened\n\n", logfile_pathname );
		Delay_msec(30000);
        exit(1);
    }


	//======== Write the header

    fscanf( CandidateDetectReadFile, "%[^=]= %6d\n",       text, &params->totalmeteors );
    fscanf( CandidateDetectReadFile, "%[^\n]\n",           text );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf%[^\n]\n", text, &params->version, datetimetext );
    fscanf( CandidateDetectReadFile, "%[^\n]\n",           text );
	fscanf( CandidateDetectReadFile, "%[^=]= %[^\n]\n",    text, params->imagery_pathname );
	fscanf( CandidateDetectReadFile, "%[^=]= %[^\n]\n",    text, params->flatfile_pathname );
	fscanf( CandidateDetectReadFile, "%[^=]= %[^\n]\n",    text, params->darkfile_pathname );
	fscanf( CandidateDetectReadFile, "%[^=]= %2ld%c",      text, &sitenum, &camchar );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->nrows );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->ncols );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->sample_time_sec );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->interleave );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->Plow );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->Phigh );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->nframes_compress );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->blocksize );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->maxclusters );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->ntuplet );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->tiny_variance );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->saturation );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->sfactor );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->SsqNdB_threshold );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld of %ld",  text, &params->mfirm, &params->nfirm );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld of %ld",  text, &params->mdrop, &params->ndrop );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->ndown );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->maxtracks );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->maxhistory );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->velocity_lowerlimit );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->velocity_upperlimit );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->linearity_threshold );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->modelfit_threshold );
	fscanf( CandidateDetectReadFile, "%[^=]= %ld",         text, &params->psfminsize );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->psfhalfwidth );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->AngleLimitDegrees );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->RelVelocityPercentage );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->DistanceLimitPixels );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->TRT_threshold );
	fscanf( CandidateDetectReadFile, "%[^=]= %lf",         text, &params->MLEdb_threshold );
    fscanf( CandidateDetectReadFile, "%[^\n]\n",           text );
    fscanf( CandidateDetectReadFile, "%[^\n]\n",           text );
    fscanf( CandidateDetectReadFile, "%[^\n]\n",           text );
    fscanf( CandidateDetectReadFile, "%[^\n]\n",           text );
    fscanf( CandidateDetectReadFile, "%[^\n]\n",           text );

	params->cameranumber = 256L * sitenum + (long)camchar;

}


//=========================================================================
//                    EMCCD_ReadDetect_Tracklet
//-------------------------------------------------------------------------
//
// Purpose:   Function to read one candidate meteor track from a detection
//            output log file. All track parameters are passed through the
//            trackinfo structure. File read pointer is positioned to the
//            next track in the file after exit from this function. If
//            track.nummeas is zero we have reached EOF.
//
// Inputs:    Previously opened log file with track information
//
// Outputs:   detectID      Detection algorithm identifier
//                               "CLUS" = cluster/tracker
//                               "MDUP" = not de-duplicated MF detection
//                               "MFIL" = matched filter detection
//            pathname      File pathname of the imagery sequence
//            track         Pointer to the trackerinfo structure
//            jdt_startmeas Julian date/time fo the 1st measurement
//            trackmeasurements   Pointer to the EMCCDtrackdata structure
//
//
//=========================================================================

/*
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// NEED TO WORK OUT HOW TO ALLOCATE MEMORY WHEN READING THE VECTORS AND ARRAYS BACK IN
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

void   EMCCD_ReadDetect_Tracklet( double sample_time_sec, char* detectID, char* pathname, struct trackerinfo *track, double *jdt_startmeas, struct EMCCDtrackdata **trackmeasurements )
{
int     k, refnumber;
double  time_offset_seconds, seqnumber;
char    text[256];


    if( fscanf( CandidateDetectReadFile, "%[^\n]\n", text ) == EOF )  {
		track->nummeas = 0;
		return;
	}

	fscanf( CandidateDetectReadFile, "%[^\n]\n",  pathname );

	fscanf( CandidateDetectReadFile,"%s %ld %lf %lf %lf %lf %lf %lf %lf %lf %d\n",
		     detectID,
			 &track->nummeas,
			 &track->multi_BNRdB,
			 &track->multi_SNRdB,
			 &track->multi_colstart,  // no longer the centroid, but the leading edge
             &track->multi_rowstart,  // no longer the centroid, but the leading edge
			 &track->multi_colspeed,
			 &track->multi_rowspeed,
			 &track->multi_linearity,
			 &track->multi_modelerr,
			 &refnumber );

	track->multi_colspeed /= sample_time_sec;  //... convert to pixels/second
	track->multi_rowspeed /= sample_time_sec;  //... convert to pixels/second

	??? adjust detected[0].time back by sample_time_sec ???
	fscanf( CandidateDetectReadFile, "%lf %lf %s\n",  &track->detected[0].time, jdt_startmeas, text );  // Unix seconds, JDT, YYYYMMDD:HH:MM:SS.MSC:UTC

    for( k=0; k<track->nummeas; k++ )  {

         fscanf( CandidateDetectReadFile, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
			      &seqnumber,
				  &time_offset_seconds,
	              &track->detected[k].colcentroid,  // no longer the centroid, but the leading edge
	              &track->detected[k].rowcentroid,  // no longer the centroid, but the leading edge
				  trackmeasurements[k]->theta_deg,
				  trackmeasurements[k]->phi_deg,
				  trackmeasurements[k]->RA_deg,
				  trackmeasurements[k]->Dec_deg,
				  trackmeasurements[k]->logsumsignal,
				  trackmeasurements[k]->magnitude     );


		 track->detected[k].time = track->detected[0].time + time_offset_seconds;

	}

}
*/ //@@@@@@@@@@@@@@@@@@@@@@@


//=========================================================================
//                    EMCCD_ReadDetect_Close
//-------------------------------------------------------------------------

void   EMCCD_ReadDetect_Close( )
{
	fclose( CandidateDetectReadFile );
}


//=========================================================================
//                    BmpFileWriteMaxpixelRedCyan
//-------------------------------------------------------------------------
//  Function to map an image sequence dimensioned nframes x npixels to an
//  RGB bit map file where the red pixels are set to the temporal max value
//  for the even frame numbers (0,2,4,...) and the cyan (green and blue)
//  pixels are set to the temporal max for the odd frames (1,3,5,...)
//  NoTe that npixels = nrows * ncols.
//
//=========================================================================

void   BmpFileWriteMaxpixelRedCyan( char *filename, long nframes, long nrows, long ncols, double **image_seq,
	                                long nfiducials, double *row_fiducial, double *col_fiducial )
{
long            kframe, krow, kcol, krc, kf;
unsigned short  ushort_datum;
double         *image_ptr, double_datum;
unsigned char  *red_ptr, *grn_ptr, *blu_ptr, uchar_datum;


   //======== Allocate memory for RGB image set

   red_ptr = (unsigned char*) malloc( nrows * ncols * sizeof( unsigned char ) );
   grn_ptr = (unsigned char*) malloc( nrows * ncols * sizeof( unsigned char ) );
   blu_ptr = (unsigned char*) malloc( nrows * ncols * sizeof( unsigned char ) );

   if( red_ptr == NULL  ||  grn_ptr == NULL  ||  blu_ptr == NULL )  {
        fprintf( stdout, "ERROR ===> Memory for RGB not allocated in BmpFileWriteMaxpixelRedCyan\n" );
        fprintf( stderr, "ERROR ===> Memory for RGB not allocated in BmpFileWriteMaxpixelRedCyan\n" );
		Delay_msec(30000);
        exit(1);
   }

   memset( red_ptr, 0, nrows * ncols );
   memset( grn_ptr, 0, nrows * ncols );
   memset( blu_ptr, 0, nrows * ncols );


   //======== Looking for temporal max pixel across frames per pixel

   for( kframe=0; kframe<nframes; kframe++ )  {

       image_ptr = &image_seq[kframe][0];  //... image_seq indices = frame# x row:column

       for( krow=0; krow<nrows; krow++ )  {

		    //krc = ( nrows - krow ) * ncols;  //... flips RGB up/down

		    krc = krow * ncols;  //... let the BMP writer flip RGB up/down later

            for( kcol=0; kcol<ncols; kcol++ )  {

				//-------- Scale the image down to a resonable range = [0,255]

				double_datum = *image_ptr++ / 2.0;

				if( double_datum > 0.0 )  ushort_datum = (unsigned short)double_datum;
				else                      ushort_datum = 0;

				if( ushort_datum < 256 )  uchar_datum = (unsigned char)ushort_datum;
				else                      uchar_datum = (unsigned char)255;

				//-------- Red maps to frame 0,2,4,...   Green maps to frame 1,3,5,...

				if( kframe % 2 == 0 )  {
					if( uchar_datum > red_ptr[krc] )  red_ptr[krc] = uchar_datum;
				}
				else  {
					if( uchar_datum > grn_ptr[krc] )  grn_ptr[krc] = uchar_datum;
				}

				krc++;

			} //... end of column loop

        } //... end of row loop

   } //... end of frame loop


   //======== Copy the green image to the blue image to create cyan

   memcpy( blu_ptr, grn_ptr, nrows * ncols );


   //-------- Add the white fiducial map to the RGB image the the nearest
   //             integer pixel. Note that any flip up/down is done later
   //             by the BMP writer.

   for( kf=0; kf<nfiducials; kf++ )  {

	    krow = (long)( 0.5 + row_fiducial[kf] );
	    kcol = (long)( 0.5 + col_fiducial[kf] );

		if( krow < 1  ||  krow > nrows-2 )  continue;
		if( kcol < 1  ||  kcol > ncols-2 )  continue;

		krc = krow * ncols + kcol;

		red_ptr[krc] = 255;
		grn_ptr[krc] = 255;
		blu_ptr[krc] = 255;

		red_ptr[krc-1] = 255;
		grn_ptr[krc-1] = 255;
		blu_ptr[krc-1] = 255;

		red_ptr[krc+1] = 255;
		grn_ptr[krc+1] = 255;
		blu_ptr[krc+1] = 255;

		red_ptr[krc-ncols] = 255;
		grn_ptr[krc-ncols] = 255;
		blu_ptr[krc-ncols] = 255;

		red_ptr[krc+ncols] = 255;
		grn_ptr[krc+ncols] = 255;
		blu_ptr[krc+ncols] = 255;

   } //... end of fiducial points loop


   //-------- Save image to a bit mapped file

   WriteBMPfile_RGB( filename, red_ptr, grn_ptr, blu_ptr, nrows, ncols );


   //-------- Free memory

   free( red_ptr );
   free( grn_ptr );
   free( blu_ptr );

}



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//=========================================================================
//                        EMCCD_WriteASGARD
//-------------------------------------------------------------------------
//
// Purpose:   Function to open, write the header, track info and close an
//            ASGARD formatted output file. All parameters are passed
//            through the EMCCDparameters, trackinfo, or EMCCDtrackdata structures.
//
// Inputs:    logfile_folder     Pathname of file folder
//            params             Pointer to the EMCCDparameters structure
//            nummeas            Number of measurements
//            seqnumber          Vector of sequence numbers per measurement
//            sample_time_sec    Time spacing betwen adjacent-in-time frame (seconds)
//            trackmeasurements  Pointer to the EMCCDtrackdata structure
//
// Outputs:   ASGARD detection file
//
//
//=========================================================================


void   EMCCD_WriteASGARD( char*                    logfile_folder,
	                      char*                    software_version,
                          struct EMCCDparameters  *params,
						  int                      nummeas,
					      double                  *seqnumber,
						  double                   sample_time_sec,
						  struct EMCCDtrackdata   *trackmeasurements )
{
int     sitenum, camnum, frame_flag, frame_bias;
int     k, k_brightest, nprev;
long    year, month, day, hour, minute, second, milliseconds;
double  jdt_unixstart, jdt_brightest_trackmeas, current_time;
double  halfpi;
char    ASGARD_file_pathname[256];

FILE   *ASGARDWriteFile;


	//======== Parse the camera number into site and camera

	sitenum = params->cameranumber / 256;
	camnum  = params->cameranumber % 256;


	//======== Find the frame with the brightest signal (-2.5 log10 means brightest is most negative)

	k_brightest = 0;

    for( k=1; k<nummeas; k++ )  {

		if( trackmeasurements[k].logsumsignal < trackmeasurements[k_brightest].logsumsignal )  k_brightest = k;

	}


	//======== Calendar date and time extraction for the brightest signal frame

    jdt_unixstart = 2440587.5;  //... Julian date for January 1, 1970  0:00:00 UTC

	jdt_brightest_trackmeas = jdt_unixstart  +  trackmeasurements[k_brightest].time / 86400.0;  //  .time is seconds since unix start

	jdt_brightest_trackmeas += sample_time_sec / 86400.0;  // "leading edge" biased forward by the frame time

	CalendarDateAndTime( jdt_brightest_trackmeas, &year, &month, &day, &hour, &minute, &second, &milliseconds );


	//======== Count the number of ASGARD previous file time duplicates by testing that the time stamp for the brightest
	//              frame (to the nearest second) is the same as any previous ASGARD file times.

	nprev = 0;

	current_time = (double)( (int)trackmeasurements[k_brightest].time );  //... drop milliseconds

    for( k=0; k<MAXPREV; k++ )  {

		if( params->previous_time[k] == current_time )  nprev++;

	}

    for( k=0; k<MAXPREV-1; k++ )  params->previous_time[k] = params->previous_time[k+1];

	params->previous_time[MAXPREV-1] = current_time;


	//======== Open the output ASGARD formatted file

    sprintf( ASGARD_file_pathname, "%sev_%04d%02d%02d_%02d%02d%02d%c_%02d%c.txt",
		                           logfile_folder,
								   year, month, day, hour, minute, second,
								   65 + nprev,
								   sitenum,
								   camnum );

    if( ( ASGARDWriteFile = fopen( ASGARD_file_pathname, "wt" ) ) == NULL )  {
        fprintf( stdout, "ERROR ===> ASGARD file %s not opened\n\n", ASGARD_file_pathname );
        fprintf( stderr, "ERROR ===> ASGARD file %s not opened\n\n", ASGARD_file_pathname );
		Delay_msec(30000);
        exit(1);
    }




	//======== Report on the "leading edge" for the row and column positions

	frame_bias = (int)( 0.999999 + 1.0 / sample_time_sec );

	halfpi = 3.14159265359 / 2.0;


	//======== Write the header

	fprintf( ASGARDWriteFile, "#\n" );
	fprintf( ASGARDWriteFile, "#   version : %s\n", params->versiontext );
	fprintf( ASGARDWriteFile, "#    num_fr : %d\n", nummeas );
	fprintf( ASGARDWriteFile, "#    num_tr : 0\n" );
	fprintf( ASGARDWriteFile, "#      time : %04d%02d%02d %02d:%02d:%02d.%03d UTC\n", year, month, day, hour, minute, second, milliseconds );
	fprintf( ASGARDWriteFile, "#      unix : %17.6lf\n", trackmeasurements[k_brightest].time + sample_time_sec );
	fprintf( ASGARDWriteFile, "#       ntp : LOCK 000000 000000 000\n" );
	fprintf( ASGARDWriteFile, "#       seq : %d\n", (int)seqnumber[k_brightest] );
	fprintf( ASGARDWriteFile, "#       mul : %d [%c]\n", nprev, 65 + nprev );
	fprintf( ASGARDWriteFile, "#      site : %02d\n", sitenum );
	fprintf( ASGARDWriteFile, "#    latlon : %.4lf %.4lf %.1lf\n", params->latitude_deg, params->longitude_deg, params->height_meters );
	fprintf( ASGARDWriteFile, "#      text : %s\n", params->sitename );
	fprintf( ASGARDWriteFile, "#    stream : %c\n", camnum );
	fprintf( ASGARDWriteFile, "#     plate : %s\n", params->platefile );
	fprintf( ASGARDWriteFile, "#      geom : %d %d\n", params->ncols, params->nrows );
	fprintf( ASGARDWriteFile, "#    filter : 0\n" );
	fprintf( ASGARDWriteFile, "#\n" );
	fprintf( ASGARDWriteFile, "#  fr    time    sum     seq       cx       cy      th      phi     lsp    mag  flag   bak    max\n" );


	//======== Write each frame's measurement

    for( k=0; k<nummeas; k++ )  {

		//... Time offset is relative the brightest frame time

		frame_flag = 0;

	    fprintf( ASGARDWriteFile, "%5d %7.3lf %6d %7d %8.3lf %8.3lf %7.3lf %8.3lf %7.3lf %6.2lf  %04d %5.1lf %6.1lf\n",
			     frame_bias + (int)( seqnumber[k] - seqnumber[0] ),                //... frame relative to first frame + 1/fs bias
				 trackmeasurements[k].time - trackmeasurements[k_brightest].time,  //... time relative to brightest frame
				 (int)pow( 10.0, trackmeasurements[k].logsumsignal / -2.5 ),       //... integrated demeaned signal
				 (int)seqnumber[k],
	             trackmeasurements[k].col,
	             trackmeasurements[k].row,
				 trackmeasurements[k].theta_deg,      //... zenith angle from zenith
				 trackmeasurements[k].phi_deg,        //... azimuth north of east
				 trackmeasurements[k].logsumsignal,   //... -2.5 * log10 of integrated demeaned signal
				 trackmeasurements[k].magnitude,      //... V magnitude corrected for airmass
				 frame_flag,
				 trackmeasurements[k].background,     //... std dev of background in aoi
				 trackmeasurements[k].maxpixel_aoi ); //... max demeaned pixel in AOI

	}


	//======== Close the ASGARD file

	fclose( ASGARDWriteFile );

}

//=========================================================================

