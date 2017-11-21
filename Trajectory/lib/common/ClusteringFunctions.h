
//#######################################################################################
//  The following functions are designed to provide a very fast cluster-based 
//  blob/streak detection that can be coupled to a multi-frame tracking capability. 
//  It was adapted from the AIM-IT algorithmic functions for extremely short latency 
//  detection and tracking of meteors. These clustering functions have been upgraded 
//  to handle more generic input. Note that other line detection algorithms may be more
//  effective at detecting narrow field-of-view, i.e. very long streaks that appear 
//  for <= 3 frames in the FOV.
//
//  These clustering functions assume that the user has pre-thresholded the imagery data
//  and is providing an exceedance count and the pointer offsets to the exceedance pixel
//  positions per frame. This can be obtained in many ways via pre-processing of the 
//  imagery sequence. Pre-processing opions one may consider are:
//    1) Track the temporally running mean (or median) and standard deviation images and
//       threshold each new frame such that the frame > mean + factor * sigma
//       to obtain all the pixel exceedance locations in the frame.
//    2) Difference a pair of frames while also tracking the running standard deviation 
//       of the difference (assume the mean is zero) and threshold each pixel such that
//       |diff| > factor * sigma to obtain pixel exceedances. Note there are 4 centroids
//       per frame pair (interleaved) that must be worked on a per frame pair basis.
//    3) Rely on the CAMS compression algorithm to segment maxpixel values per frame.
//
//  For option 3, the data content needed by the ClusteringExceedances function is the 
//  same as that which is populated by the FourProduct_PrepBuild function of the CAMS 
//  archiveFTP.cpp software. Thus if a CAMS FF file is read in, the populated 
//  frmcount_ptr (number of exceedances for given frame index), the cumcount_ptr (the 
//  starting element into the mpoffset vector for given frame index), and the 
//  mpoffset_ptr (defined as the rcindex = row * coldim + column), are all vectors 
//  filled with the necessary input data needed by the ClusteringExceedances function. 
//
//  Note that the CAMS function FourProduct_FileRead calls the FourProduct_PrepBuild 
//  automatically when reading an FF file and that the CAMS compression format 
//  effectively thresholds the data into highest pixel values per frame number. One
//  can simply use the maxpixel values available per frame directly from the CAMS format.
//
//  The output product is a set of spatial position measurements across the focal plane
//  for either a full frame or both fields depending on interleave/progressive scan mode.
//  The time attriibute is NOT set upon exit and is up to the user to set this or have 
//  post-processing functions handle the time assignment. 
//
//  The information can be retreived for the k-th spatial cluster as follows:
//
//     struct  clusterinfo  cluster;
//
//     cluster.measurements_odds[k].attribute     for k = 0, 1, ... track.nclusters_odds  
//     cluster.measurements_even[k].attribute     for k = 0, 1, ... track.nclusters_even  
//     cluster.measurements_full[k].attribute     for k = 0, 1, ... track.nclusters_full
//
//  Where "odds" = odd rows field; "even" = even rows field; "full" = all rows full frame
//
//  Where "attribute" is defined per the entries below (second column) taken
//     from the file AttributeStructure.h
//
//	   double    time; (NOT SET) //  Time in user defined units (sec or frame# or...)
//	   double    rowcentroid;    //  Row centroid in pixels
//	   double    colcentroid;    //  Col centroid in pixels
//	   long      nexceed;        //  Number of exceedance pixels in region
//	   long      nsaturated;     //  Number of saturated pixels in region
//	   long      npixels;        //  Number of total pixels in region
//     double    sumsignal;      //  Summed signal over image chip region
//     double    sumnoise;       //  Summed noise over image chip region
//	   double    SsqNdB;         //  Signal squared over noise detection metric in dB
//	   double    entropy;        //  Region entropy
//	   double    PCD_angle;      //  Phase coded disk angle estimate in deg
//	   double    PCD_width;      //  Phase coded disk line width estimate in pixels
//	   double    Hueckel_angle;  //  Hueckel kernel angle estimate in deg
//
//
//=======================================================================================
//  The processing calls invoked by a developer are as follows:
//
//  ClusteringBlocksize      - Called once near program start after imagery size/scale 
//                                information is available to obtain a cluster block
//                                size recommendation. The user may choose their own.
//  ClusteringInitialization - Called once near program start to set up the clusterinfo
//                                structure and its associated run parameter contents.
//
//  ---Frame by frame loop
//
//     "Pre-Processing"      - Front end processes to form a pixel exceedance list to
//                                pass into the clustering functions and also obtain
//                                or maintain/update a running image mean and sigma.
//     ClusteringExceedances - Called for each frame (or difference frame pair) given
//                                thresholded pixel exceedance locations and produces
//                                clusters based on macro-cell peak counts with an initial
//                                guess at row and column centroids (binary pixel based).
//     ClusteringCentroids   - Called to refine the centroids previously generated
//                                for all the cluster peaks found using ALL the pixel 
//                                values (gray level), means, and standard deviations
//                                (not just the exceedance pixels).
//     ClusteringDetector    - Called after the centroid refinement for all the cluster
//                                peaks to estimate various additional attributes such
//                                as the squared signal over variance detector, entropy, 
//                                orientation, and per frame/field threshold detections.
//     "Post-processing"     - Back end processes to perform multi-frame tracking,
//                                track refinement, false alarm reduction, and reporting.
//
//                                IMPORTANT NOTE: The clustering codes are working on a 
//                                            frame or field pair and are thus agnostic
//                                            to the absolute time of each measurement,
//                                            so the post-processing functions is where 
//                                            the assignment of a time is made to each
//                                            interleaved or progressive scan measurement.
//
//  ---End of frame loop
//
//  ClusteringFreeMemory     - Called once at the end of image processing to free memory
//
//
//=======================================================================================
//
//  Date        Author      Change Description
//  ----------  ----------  -------------------------------------------------------------
//  2015-12-17  Gural       Original code based on AIMIT clustering algorithms
//  2016-01-26  Gural       Modified to add attributeinfo structure for measurements
//
//#######################################################################################




//#######################################################################################
//                            Mnemonics and Structure Definitions
//#######################################################################################

                             //--- Enumeration of "datatype" for the various array options
#define IS_UNSIGNED_CHAR  1  //    Single byte integer = [ 0, 255 ] 
#define IS_UNSIGNED_SHORT 2  //    Two byte positive integers = [ 0, 65535 ]
#define IS_SHORT          3  //    Two byte signed integers = [-32768, +32767 ]
#define IS_LONG           4  //    Four byte signed integers = [ -2147483648, +2147483647 ]
#define IS_FLOAT          5  //    Four byte floats ~7 significant decimal diits
#define IS_DOUBLE         6  //    Eight byte doubles ~16 significant decimal digits


//======================================================================================
//      Signal independent squared-numerator detector { R^-1 [x-<x>] }^2 / R^-1
//      R = diagonal covariance matrix comprised of the variance per pixel
//======================================================================================

struct  clusterinfo
{
  long    maxclusters;       // User specified max number of clusters to find in a frame/field
  long    nrows;             // Number of rows in the image frame
  long    ncols;             // Number of columns in the image frame
  long    coldim;            // Column wrap dimensionality of the image frame; often = ncols
  long    datatype;          // Data type specification of x* arrays (all arrays must be the same type)
  long    interleave;        // Interleave mode (0=progressive, 1=even/odd, 2=odd/even)
  long    ntuplet;           // Number of neighbors + self to declare a tuplet cluster of pixels
  long    blocksize;         // Number of pixels comprising a cluster block side dimension
  long    nblkrows;          // Number of block rows in the block array (1 + nrows / blocksize)
  long    nblkcols;          // Number of block cols in the block array (1 + ncols / blocksize)

  double  tiny_variance;     // Smallest variance for data type used, to avoid divide by 0
  double  saturation;        // Largest value to be encountered for a given data type
  double  sfactor;           // Standard deviation factor to find exceedance count attribute
                             //     where the exceedance threshold = <x> + sfactor * sigma
  double  SsqNdB_threshold;  // Signal independent squared threshold for detection in dB

  long    nclusters_odds;    // Number of cluster peaks (measurements) for odd  rows (interleaved)
  long    nclusters_even;    // Number of cluster peaks (measurements) for even rows (interleaved)
  long    nclusters_full;    // Number of cluster peaks (measurements) for full rows (progressive scan)

  struct  attributeinfo  *measurements_odds;  // Pointer to attributes based on odd  rows pixels in patch
  struct  attributeinfo  *measurements_even;  // Pointer to attributes based on even rows pixels in patch
  struct  attributeinfo  *measurements_full;  // Pointer to attributes based on all  rows pixels in patch


                             // Work arrays of nblkrows x nblkcols blocks ref via block_***[i][j]

  long  **block_counts_odds; // Counts of # pixel exceedances per block that were clustered for odd rows
  long  **block_rowsum_odds; // Sum of row indices for all clustered exceedances in a block for odd rows
  long  **block_colsum_odds; // Sum of col indices for all clustered exceedances in a block for odd rows

  long  **block_counts_even; // Counts of # pixel exceedances per block that were clustered for even rows
  long  **block_rowsum_even; // Sum of row indices for all clustered exceedances in a block for even rows
  long  **block_colsum_even; // Sum of col indices for all clustered exceedances in a block for even rows


                             // Work arrays of blocksize x blocksize ref via hueckel_coef[row,col][basis]

  double **hueckel_coef;     // Hueckel coefficients indexed by unwrapped row,col versus basis


                             // Work vectors such that signal = [x-<x>]/sigma^2,  noise = 1/sigma^2

  double *sigsum_across_row; // Sum of signal across the columns for each row
  double *sigsum_downcolumn; // Sum of signal down the rows for each column
  double *noisum_downcolumn; // Sum of noise down the rows for each column

                             // Work vectors s.t. dim = MAXCLUSTERPEAKS + 1 for insertion sort indexing

  long    *counts_peak;  // Sorted peak exceedance counts in 2x2 block macro-cells
  long    *blkrow_peak;  // Upper left block row index of macro-cell peaks
  long    *blkcol_peak;  // Upper left block col index of macro-cell peaks

};


//#######################################################################################
//  The ClusteringBlocksize function computes the maximum cluster region size based on
//  the maximum distance a meteor may traverse across the focal plane in one full frame
//  time. This assumes a 80 km minimum range, 72 km/sec maximum entry velocity, the 
//  meteor at 90 deg from the radiant (--> maximum sine value), and user input values 
//  for the angular resolution of the FOV in arcmin/pixel and the full image frame rate
//  in Hz (frames per second). The returned region size value recommendation can be used
//  to feed the "blocksize" input argument to ClusteringInitialization. 
//
//  Note: Max apparent angular velocity = ( 180 / Pi ) * Vmax * sin(Dradiant) / Rmin
//                                      = 51.6 deg/sec for the given assumptions above 
//
//=======================================================================================
//
//  INPUTS:
//     arcminperpixel  Angular resolution of a pixel in arcminutes per pixel
//     framerateHz     Full frames per second
//
//  OUTPUT:
//     blocksize       Blocksize recommendation for ClusteringInitialization
//                         equivalent to max meteor motion in pixels per full-frame
//
//  DEPENDENCIES:
//     None
//
//=======================================================================================

long  ClusteringBlocksize( double arcminperpixel, double framerateHz )
{
long  pixelsperframe;

      pixelsperframe = (long)( 51.6 * 60.0 / framerateHz / arcminperpixel );

	  return( pixelsperframe );
}


//#######################################################################################
//  The ClusteringInitialization function allocates memory and sets up the "clusterinfo"
//  structure during this single initialization call. The user provides the image 
//  frame size, declared column dimension of the image array, interleave flag, cluster 
//  block size, and cluster tuplet count threshold. An estimate of the smallest variance
//  expected (positive non-zero entry) to avoid divide by zero when whitening the signal
//  is expected from the user based on data type (e.g. integer data types can use a value
//  unity over a hundred or 0.01 for "tiny_variance". The user also defines the detection
//  threshold of the signal independent matched filter response in dB. The clusterinfo 
//  structure's contents are used on all subsequent calls of ClusteringExceedances, 
//  ClusteringCentroids and ClusteringMFDetector. To free the memory allocated to the
//  arrays and vectors embedded in the clusterinfo structure once all the processing 
//  is completed, call the function ClusteringFreeMemory.
//
//=======================================================================================
//
//  INPUTS:
//     maxclusters     User specified max number of clusters to find in a frame/field
//     nrows           Number of rows in the image frame
//     ncols           Number of columns in the image frame
//     coldim          Column wrap dimensionality of the image frame; often = ncols
//     datatype        Data type specification of x* arrays (all arrays must = same type)
//                        IS_UNSIGNED_CHAR, IS_UNSIGNED_SHORT, IS_SHORT, IS_LONG, 
//                        IS_FLOAT, IS_DOUBLE
//     interleave      Interleave mode (0=progressive, 1=even/odd, 2=odd/even)
//     blocksize       Number of pixels comprising a cluster block's side dimension
//                        (blocksize x blocksize) where one can use the ClusteringBlocksize
//                        recommendation value.
//     ntuplet         Number of neighbors + self to declare a clustered pixel
//     tiny_variance   Smallest non-zero variance to be encountered for given data type
//     saturation      Largest value to e encountered for a given data type
//     sfactor         Standard deviation factor to find exceedance count attribute
//                        threshold = <x> + sfactor * sigma
//     SsqNdB_threshold  Signal independent squared threshold for detection (typically 0.0 dB)
//
//
//  OUTPUT:
//     cluster         Clusterinfo structure populated with the input arguments, block 
//                         info plus the required memory allocation
//
//  DEPENDENCIES:
//     None
//
//=======================================================================================

void  ClusteringInitialization( long maxclusters, long nrows, long ncols, long coldim, long datatype, 
	                            long interleave, long blocksize, long ntuplet, 
                                double tiny_variance, double saturation, double sfactor, double SsqNdB_threshold, 
                                struct clusterinfo *cluster )
{
long  blkrow, kpixel, npixels, nbasis;


    //-------- Do some basic warning and error checks

    if( tiny_variance <= 0.0 )  {
		printf("WARNING:  tiny_variance <= 0 in ClusteringInitialization. Resetting to 0.01\n");
		tiny_variance = 0.01;
		Delay_msec(10000);
	}

    if( nrows <= 0  ||  ncols <= 0 )  {
		printf("ERROR:  nrows or ncols must be positive integers in ClusteringInitialization\n");
		Delay_msec(10000);
		exit(1);
	}

    if( coldim < ncols )  {
		printf("ERROR:  coldim < ncols in ClusteringInitialization\n");
		Delay_msec(10000);
		exit(1);
	}

    if( ntuplet <= 0 )  {
		printf("WARNING:  ntuplet <= 0 in ClusteringInitialization. Resetting to 2\n");
		ntuplet = 2;
		Delay_msec(10000);
	}

    if( blocksize <= 0 )  {
		printf("WARNING:  blocksize <= 0 in ClusteringInitialization. Resetting to 32\n");
		blocksize = 32;
		Delay_msec(10000);
	}


    //-------- Assign input parameters to the clusterinfo structure elements

	cluster->maxclusters = maxclusters;
	cluster->nrows       = nrows;
	cluster->ncols       = ncols;
	cluster->coldim      = coldim;
	cluster->datatype    = datatype;
	cluster->interleave  = interleave;
	cluster->ntuplet     = ntuplet;
	cluster->blocksize   = blocksize;
	cluster->nblkrows    = 1 + nrows / blocksize;
	cluster->nblkcols    = 1 + ncols / blocksize;

	cluster->tiny_variance    = tiny_variance;
	cluster->saturation       = saturation;
	cluster->sfactor          = sfactor;
	cluster->SsqNdB_threshold = SsqNdB_threshold;


	//--------- Allocate memory for attribute structures

	cluster->measurements_odds = ( struct  attributeinfo* ) malloc( maxclusters * sizeof( struct  attributeinfo ) );
	cluster->measurements_even = ( struct  attributeinfo* ) malloc( maxclusters * sizeof( struct  attributeinfo ) );
	cluster->measurements_full = ( struct  attributeinfo* ) malloc( maxclusters * sizeof( struct  attributeinfo ) );

	if( cluster->measurements_odds == NULL  ||  cluster->measurements_even == NULL  ||  cluster->measurements_full == NULL )  {
		printf( " Memory not allocated for attribute vectors in ClusteringInitialization \n" );
		Delay_msec(10000);
		exit(1);
	}


	//--------- Allocate memory for the cluster peak information working vectors 
    //               s.t. dimension = maxclusters + 1 for insertion sort indexing

	cluster->counts_peak = ( long* ) malloc( ( maxclusters + 1 ) * sizeof( long ) );
	cluster->blkrow_peak = ( long* ) malloc( ( maxclusters + 1 ) * sizeof( long ) );
	cluster->blkcol_peak = ( long* ) malloc( ( maxclusters + 1 ) * sizeof( long ) );

	if( cluster->counts_peak == NULL  ||  cluster->blkrow_peak == NULL  ||  cluster->blkcol_peak == NULL )  {
		printf( " Memory not allocated for peak info vectors in ClusteringInitialization \n" );
		Delay_msec(10000);
		exit(1);
	}


	//--------- Allocate memory for the pixel block working arrays
	//             Each array can be referenced by two dimensional indexing  -->  block_***[i][j]

	cluster->block_counts_odds = (long**) malloc( ( cluster->nblkrows * sizeof(long*) ) + ( cluster->nblkrows * cluster->nblkcols * sizeof(long) ) );
	cluster->block_rowsum_odds = (long**) malloc( ( cluster->nblkrows * sizeof(long*) ) + ( cluster->nblkrows * cluster->nblkcols * sizeof(long) ) );
	cluster->block_colsum_odds = (long**) malloc( ( cluster->nblkrows * sizeof(long*) ) + ( cluster->nblkrows * cluster->nblkcols * sizeof(long) ) );

	cluster->block_counts_even = (long**) malloc( ( cluster->nblkrows * sizeof(long*) ) + ( cluster->nblkrows * cluster->nblkcols * sizeof(long) ) );
	cluster->block_rowsum_even = (long**) malloc( ( cluster->nblkrows * sizeof(long*) ) + ( cluster->nblkrows * cluster->nblkcols * sizeof(long) ) );
	cluster->block_colsum_even = (long**) malloc( ( cluster->nblkrows * sizeof(long*) ) + ( cluster->nblkrows * cluster->nblkcols * sizeof(long) ) );

	if( cluster->block_counts_odds == NULL  ||  cluster->block_rowsum_odds == NULL  ||  cluster->block_colsum_odds == NULL  ||
	    cluster->block_counts_even == NULL  ||  cluster->block_rowsum_even == NULL  ||  cluster->block_colsum_even == NULL )  {
		printf( " Memory not allocated for arrays in ClusteringInitialization \n" );
		Delay_msec(10000);
		exit(1);
	}

    for( blkrow = 0; blkrow < cluster->nblkrows; ++blkrow) {
        cluster->block_counts_odds[blkrow] = (long* )( (cluster->block_counts_odds + cluster->nblkrows) + blkrow * cluster->nblkcols );
        cluster->block_rowsum_odds[blkrow] = (long* )( (cluster->block_rowsum_odds + cluster->nblkrows) + blkrow * cluster->nblkcols );
        cluster->block_colsum_odds[blkrow] = (long* )( (cluster->block_colsum_odds + cluster->nblkrows) + blkrow * cluster->nblkcols );

        cluster->block_counts_even[blkrow] = (long* )( (cluster->block_counts_even + cluster->nblkrows) + blkrow * cluster->nblkcols );
        cluster->block_rowsum_even[blkrow] = (long* )( (cluster->block_rowsum_even + cluster->nblkrows) + blkrow * cluster->nblkcols );
        cluster->block_colsum_even[blkrow] = (long* )( (cluster->block_colsum_even + cluster->nblkrows) + blkrow * cluster->nblkcols );
    }



	//--------- Allocate memory for the signal and noise sum vectors used during centroid refinement

	cluster->sigsum_across_row = (double*) malloc( ( cluster->nrows ) * sizeof(double) );
	cluster->sigsum_downcolumn = (double*) malloc( ( cluster->ncols ) * sizeof(double) );
	cluster->noisum_downcolumn = (double*) malloc( ( cluster->ncols ) * sizeof(double) );

	if( cluster->sigsum_across_row == NULL  ||  cluster->sigsum_downcolumn == NULL  ||  cluster->noisum_downcolumn == NULL )  {
		printf( " Memory not allocated for vectors in ClusteringInitialization \n" );
		Delay_msec(10000);
		exit(1);
	}


	//--------- Allocate memory for the Hueckel working array

	npixels = blocksize * blocksize;
	nbasis  = 7;

	cluster->hueckel_coef = (double**) malloc( ( npixels * sizeof(double*) ) + ( npixels * nbasis * sizeof(double) ) );

	if( cluster->hueckel_coef == NULL )  {
		printf( " Memory not allocated for hueckel_coef in ClusteringInitialization \n" );
		Delay_msec(10000);
		exit(1);
	}

    for( kpixel = 0; kpixel < npixels; ++kpixel) {
        cluster->hueckel_coef[kpixel] = (double* )( (cluster->hueckel_coef + npixels) + kpixel * nbasis );
    }


	//--------- Initialize the Hueckel working array


}


//#######################################################################################
//  The ClusteringExceedances function locates the largest clusters of pixels in 
//  overlapping 2x2 blocks of macro-cells and makes a binary valued centroid estimate. 
//  The function assumes a pre-thresholded set of pixel locations (a.k.a. exceedances)
//  are available for each frame to be processed. The exceedance pixel's row and  
//  column positions are specified as pointer offsets into the original frame.  The
//  exceedance information is passed in as the total number of threshold crossers 
//  and a vector of position offsets of each exceedance pixel relative to the imagery 
//  array's start pixel. Thus the clustering operation is done one frame (or frame 
//  pair if differencing) at a time. The offsets are assumed listed in column 
//  precedence ordering (C array ordering). This function will work with the data 
//  captured either in a progressive scan mode (full frame) or an interleaved mode
//  (even-followed-by-odd or odd-followed-by-even rows in time) and the centroid is 
//  based on a spatial region of interest (ROI) appropriate to the mode. The function 
//  finds binary value based centroids in those macro-cells that contain the highest
//  exceedance counts. The peak count list is then culled to eliminate spatially close 
//  neighborhood peaks (taking the higher count peak relative to nearby macro-cells).
//
//=======================================================================================
//
//  INPUTS:
//     nexceed_pixels  Number of exceedance pixels for this frame
//    *pixel_offsets   Vector of pixel offsets of all the exceedances for this frame
//                        Offsets re to the 1st image pixel in column precedence order
//                        equivalent to the pointer difference to &image[0][0]
//    *cluster         Pointer to the clusterinfo structure containing:
//        nrows           Number of rows in the image frame
//        ncols           Number of columns in the image frame
//        coldim          Column wrap dimensionality of the image frame; Often = ncols
//        ntuplet         Number of neighbors + self to declare a clustered exceedance pixel
//        blocksize       Number of pixels comprising a cluster block's side dimension
//        nblkrows        Number of block rows in the block arrays
//        nblkcols        Number of block columns in the block arrays
//
//  OUTPUT:
//    *cluster         Pointer to the clusterinfo structure containing:
//        **block_counts      Array of exceedance pixel counts per block via block_counts[i][j]
//        nclusters_odds      Number of cluster peaks (odd  rows)
//        nclusters_even      Number of cluster peaks (even rows)
//        nclusters_full      Number of cluster peaks (full rows)
//        measurements_odds   Attributes for odd  rows (1st guess for centroids)
//        measurements_even   Attributes for even rows (1st guess for centroids)
//        measurements_full   Attributes for all  rows (1st guess for centroids)
//                   Note that attributes assigned are the first guess for row & col centroids
//
//  DEPENDENCIES:
//     None
//
//=======================================================================================

void  ClusteringExceedances( long nexceed_pixels, long *pixel_offsets, struct clusterinfo *cluster )
{
long    rcindex, kcolpixel, test_offset, kexceed, jpeak, kpeak, npeaks;
long    counts, counts_odds, counts_even;
long    blkrow, rowsum_odds, rowsum_even, row;
long    blkcol, colsum_odds, colsum_even, col;
long   *topleft_ptr, *uppleft_ptr, *dwnleft_ptr, *botleft_ptr, *last_ptr, *rcindex_ptr;
double  dsq;


    //===================================================================================
    //   Find clusters defined by a neighborhood + self exceedance pixel count that
    //      is greater than or equal to the user defined "ntuplet" value. The 5x5 
    //      neighborhood pixels that are checked as being exceedance pixels, are
    //      spatially located as shown in the diagrams below and labeled as "o" 
    //      relative to the center test exceedance pixel "t". Those not checked as
    //      being exceedance pixels to form the cluster are labeled "x". 
    //      
    //                Interleaved Fields            Progressive Scan
    //                ------------------            ----------------
    //     "top"           x o o o x                    x o o o x
    //     "upp"           x x x x x                    o o o o o
    //     "cen"           o o t o o                    o o t o o
    //     "dwn"           x x x x x                    o o o o o
    //     "bot"           x o o o x                    x o o o x 
    //
    //===================================================================================

	//======== Early exit if insufficient number of exceedance pixels to cluster

    if( nexceed_pixels < 5 )  {
		cluster->nclusters_odds = 0;
		cluster->nclusters_even = 0;
		cluster->nclusters_full = 0;
		return;
	}


    //======== Clear the clustered block arrays

	for( blkrow=0; blkrow<cluster->nblkrows; blkrow++ )  {

	     for( blkcol=0; blkcol<cluster->nblkcols; blkcol++ )  {

			  cluster->block_counts_odds[blkrow][blkcol] = 0;
			  cluster->block_rowsum_odds[blkrow][blkcol] = 0;
			  cluster->block_colsum_odds[blkrow][blkcol] = 0;

			  cluster->block_counts_even[blkrow][blkcol] = 0;
			  cluster->block_rowsum_even[blkrow][blkcol] = 0;
			  cluster->block_colsum_even[blkrow][blkcol] = 0;
		 }
    }


    //======== Set starting pointers and end of exceedance vector pointer

    topleft_ptr  = pixel_offsets; //... pointer to exceedance pixel offsets 2 rows above test pixel
    uppleft_ptr  = pixel_offsets; //... pointer to exceedance pixel offsets 1 row  above test pixel
	dwnleft_ptr  = pixel_offsets; //... pointer to exceedance pixel offsets 1 row  below test pixel
	botleft_ptr  = pixel_offsets; //... pointer to exceedance pixel offsets 2 rows below test pixel
	last_ptr     = pixel_offsets + nexceed_pixels - 4L;  //... pointer offset gets tested up to +4


	//======== Loop over each exceedance pixel to test the neighborhood pixel exceedance count.         
	//            Skip the first and last 2 exceedance pixels to avoid extra IF tests around
	//            the center test pixel at the loop's beginning and end.

    for( kexceed=2; kexceed<nexceed_pixels-2; kexceed++ )  {

		 //------ Get the pointer to the next exceedance pixel and its offset re first pixel

		 rcindex_ptr = pixel_offsets + kexceed; 

		 rcindex = *rcindex_ptr;


		 //------ Get the row and column indices of the test pixel, ensure > 2 pixels from any edge

		 row = rcindex / cluster->coldim;

		 if( row < 2  ||  row > cluster->nrows-3 )  continue; 


		 col = rcindex % cluster->coldim;

		 if( col < 2  ||  col > cluster->ncols-3 )  continue;


		 //------ Initialize the cluster count to 1 (self)

		 counts = 1;


		 //------ Check the center row for exceedance pixels within two columns of either side

		 if( rcindex - *(rcindex_ptr-2) <= +2 )  counts++;

		 if( rcindex - *(rcindex_ptr-1) <= +2 )  counts++;


		 if( *(rcindex_ptr+1) - rcindex <= +2 )  counts++;

		 if( *(rcindex_ptr+2) - rcindex <= +2 )  counts++;

		 
		 //------ Check two rows up for exceedances within one column of the center pixel's column
		 //          Proceed for either progressive scan or interleaved, skip if cluster metric met

		 if( counts < cluster->ntuplet )  {

			 //...... Shift the top left exceedance pixel pointer to align at or to the right of the left edge

			 while( rcindex - *topleft_ptr > 2*cluster->coldim + 2  &&  topleft_ptr != last_ptr )  topleft_ptr++;

		     for( kcolpixel=0; kcolpixel<5; kcolpixel++ )  {

			      test_offset = *(topleft_ptr + kcolpixel) - (rcindex - 2*cluster->coldim);

			      if( test_offset > +1 ) break;  // exit early if past the test pixels in this row

		          if( labs( test_offset ) <= +1 )  counts++;

		     }

		 }


		 //------ Check one row up for exceedances within two columns of the center pixel's column
		 //          Proceed for progressive scan only, skip if cluster metric met

		 if( counts < cluster->ntuplet  &&  cluster->interleave == 0 )  {

			 //...... Shift the upper left exceedance pixel pointer to align at or to the right of the left edge

			 while( rcindex - *uppleft_ptr > cluster->coldim + 2  &&  uppleft_ptr != last_ptr )  uppleft_ptr++;

		     for( kcolpixel=0; kcolpixel<5; kcolpixel++ )  {

			      test_offset = *(uppleft_ptr + kcolpixel) - (rcindex - cluster->coldim);

			      if( test_offset > +2 ) break;  // exit early if past the test pixels in this row

		          if( labs( test_offset ) <= +2 )  counts++;

		     }

		 }


		 //------ Check one row down for exceedances within two columns of the center pixel's column
		 //          Proceed for progressive scan only, skip if cluster metric met

		 if( counts < cluster->ntuplet  &&  cluster->interleave == 0 )  {

			 //...... Shift the down left exceedance pixel pointer to align at or to the right of the left edge

			 while( *dwnleft_ptr - rcindex < cluster->coldim - 2  &&  dwnleft_ptr != last_ptr )  dwnleft_ptr++;
		     
			 for( kcolpixel=0; kcolpixel<5; kcolpixel++ )  {

			      test_offset = *(dwnleft_ptr + kcolpixel) - (rcindex + cluster->coldim);

			      if( test_offset > +2 ) break;  // exit early if past the test pixels in this row

		          if( labs( test_offset ) <= +2 )  counts++;

			 }

		 }


		 //------ Check two rows down for exceedances within one column of the center pixel's column
		 //          Proceed for either progressive scan or interleaved, skip if cluster metric met

		 if( counts < cluster->ntuplet )  {

			 //...... Shift the bottom left exceedance pixel pointer to align at or to the right of the left edge

			 while( *botleft_ptr - rcindex < 2*cluster->coldim - 2  &&  botleft_ptr != last_ptr )  botleft_ptr++;
		     
			 for( kcolpixel=0; kcolpixel<5; kcolpixel++ )  {

			      test_offset = *(botleft_ptr + kcolpixel) - (rcindex + 2*cluster->coldim);

			      if( test_offset > +1 ) break;  // exit early if past the test pixels in this row

		          if( labs( test_offset ) <= +1 )  counts++;

			 }

		 }


		 //------ Increment the appropriate block array element if the test pixel meets the clustered metric
	     //            Note that due to C indexing rows 0, 2, 4, ... are odd rows
		 //                               and that rows 1, 3, 5, ... are even rows

		 if( counts >= cluster->ntuplet )  {

			 blkrow = row / cluster->blocksize;
			 blkcol = col / cluster->blocksize;

			 if( row % 2 == 0 )  {  //... odd row contribution
			     cluster->block_counts_odds[blkrow][blkcol] += 1;
			     cluster->block_rowsum_odds[blkrow][blkcol] += row;
			     cluster->block_colsum_odds[blkrow][blkcol] += col;
			 }
			 else  {                //... even row contribution
			     cluster->block_counts_even[blkrow][blkcol] += 1;
			     cluster->block_rowsum_even[blkrow][blkcol] += row;
			     cluster->block_colsum_even[blkrow][blkcol] += col;
			 }

		 }

	} //... end of exceedance pixel loop


    //===================================================================================
	//   Aggregate the blocks into overlapping 2x2 macro cells, sorting them by their
	//      total aggregated counts for combined odd and even rows. This searches for 
	//      peak macro cell counts and computes the associated full frame centroids 
	//      based solely on clustered pixel exceedance locations. Note the centroid 
	//      calculations assume binary values for the pixels at this stage and thus 
	//      row and column sums suffice to compute the centroids. The highest count 
	//      peak centroids are culled to eliminate close neighbors. Note that the
	//      centroids are later refined in the function ClusteringCentroids which
	//      uses all pixels in a region (not just exceedances) and their actual
	//      value, variance, and mean values per pixel.
	//
    //===================================================================================

	//======== Overlapped 2x2 block aggregation into macro cells and peak count sorting

	npeaks = 0;

	for( blkrow=0; blkrow<cluster->nblkrows-1; blkrow++ )  {

		 for( blkcol=0; blkcol<cluster->nblkcols-1; blkcol++ )  {

			  //-------- 2x2 block aggregation for macro cell total count of odd + even rows

			  counts = cluster->block_counts_odds[blkrow  ][blkcol  ]
			         + cluster->block_counts_odds[blkrow  ][blkcol+1]
					 + cluster->block_counts_odds[blkrow+1][blkcol  ]
					 + cluster->block_counts_odds[blkrow+1][blkcol+1]

			         + cluster->block_counts_even[blkrow  ][blkcol  ]
			         + cluster->block_counts_even[blkrow  ][blkcol+1]
					 + cluster->block_counts_even[blkrow+1][blkcol  ]
					 + cluster->block_counts_even[blkrow+1][blkcol+1];

			  if( counts == 0 )  continue;

			  //-------- Perform a fast insertion sort on counts in a macro cell

			  jpeak = npeaks - 1;

			  while( jpeak >= 0  &&  counts > cluster->counts_peak[jpeak] )  {

				     cluster->counts_peak[jpeak+1] = cluster->counts_peak[jpeak];
				     cluster->blkrow_peak[jpeak+1] = cluster->blkrow_peak[jpeak];
				     cluster->blkcol_peak[jpeak+1] = cluster->blkcol_peak[jpeak];
					 jpeak--;
			  }

			  cluster->counts_peak[jpeak+1] = counts;
			  cluster->blkrow_peak[jpeak+1] = blkrow;
			  cluster->blkcol_peak[jpeak+1] = blkcol;

			  if( npeaks < cluster->maxclusters )  npeaks++;

		 } //... end of blkcol loop

	} //... end of blkrow loop


	//======== Compute the exceedance (binary pixel value) centroids for the highest peaks
	//            using row and col sums from each contributing block (the sums were 
	//            previously computed for odd rows and even rows separately and are now 
	//            combined to form the odd rows centroid, even rows centroid, and full rows 
	//            centroid).

	for( jpeak=0; jpeak<npeaks; jpeak++ )  {

		 blkrow   = cluster->blkrow_peak[jpeak];
		 blkcol   = cluster->blkcol_peak[jpeak];

		 counts_odds = cluster->block_counts_odds[blkrow  ][blkcol  ]
			         + cluster->block_counts_odds[blkrow  ][blkcol+1]
					 + cluster->block_counts_odds[blkrow+1][blkcol  ]
					 + cluster->block_counts_odds[blkrow+1][blkcol+1];

		 counts_even = cluster->block_counts_even[blkrow  ][blkcol  ]
			         + cluster->block_counts_even[blkrow  ][blkcol+1]
					 + cluster->block_counts_even[blkrow+1][blkcol  ]
					 + cluster->block_counts_even[blkrow+1][blkcol+1];

		 rowsum_odds = cluster->block_rowsum_odds[blkrow  ][blkcol  ]
			         + cluster->block_rowsum_odds[blkrow  ][blkcol+1]
			         + cluster->block_rowsum_odds[blkrow+1][blkcol  ]
			         + cluster->block_rowsum_odds[blkrow+1][blkcol+1];

	     colsum_odds = cluster->block_colsum_odds[blkrow  ][blkcol  ]
			         + cluster->block_colsum_odds[blkrow  ][blkcol+1]
			         + cluster->block_colsum_odds[blkrow+1][blkcol  ]
			         + cluster->block_colsum_odds[blkrow+1][blkcol+1];

	     rowsum_even = cluster->block_rowsum_even[blkrow  ][blkcol  ]
			         + cluster->block_rowsum_even[blkrow  ][blkcol+1]
			         + cluster->block_rowsum_even[blkrow+1][blkcol  ]
			         + cluster->block_rowsum_even[blkrow+1][blkcol+1];

	     colsum_even = cluster->block_colsum_even[blkrow  ][blkcol  ]
			         + cluster->block_colsum_even[blkrow  ][blkcol+1]
			         + cluster->block_colsum_even[blkrow+1][blkcol  ]
			         + cluster->block_colsum_even[blkrow+1][blkcol+1];


		 //-------- Compute the full patch binary centroid (there is never a total counts = 0)

		 cluster->measurements_full[jpeak].rowcentroid = (double)(rowsum_odds + rowsum_even) / (double)(counts_odds + counts_even);
		 cluster->measurements_full[jpeak].colcentroid = (double)(colsum_odds + colsum_even) / (double)(counts_odds + counts_even);


		 //-------- Compute the odd row patch binary centroid

		 if( counts_odds == 0 ) counts_odds = 1;

		 cluster->measurements_odds[jpeak].rowcentroid = (double)rowsum_odds / (double)counts_odds;
		 cluster->measurements_odds[jpeak].colcentroid = (double)colsum_odds / (double)counts_odds;


		 //-------- Compute the even row patch binary centroid

		 if( counts_even == 0 ) counts_even = 1;

		 cluster->measurements_even[jpeak].rowcentroid = (double)rowsum_even / (double)counts_even;
		 cluster->measurements_even[jpeak].colcentroid = (double)colsum_even / (double)counts_even;

	} //... end of loop on jpeak


	//======== Flag the spatially close pairs of peaks found in the list - compare every pair 
	//            by examining the full-all-rows centroid position distances

	for( jpeak=0; jpeak<npeaks-1; jpeak++ )  {

	       for( kpeak=jpeak+1; kpeak<npeaks; kpeak++ )  {

			    dsq = ( cluster->measurements_full[jpeak].rowcentroid - cluster->measurements_full[kpeak].rowcentroid )
					* ( cluster->measurements_full[jpeak].rowcentroid - cluster->measurements_full[kpeak].rowcentroid )
					+ ( cluster->measurements_full[jpeak].colcentroid - cluster->measurements_full[kpeak].colcentroid )
					* ( cluster->measurements_full[jpeak].colcentroid - cluster->measurements_full[kpeak].colcentroid );

				//... Flag a lower count peak spatially close to a higher count peak

		        if( dsq < 2.0 * (double)cluster->blocksize * (double)cluster->blocksize )  {
					
					if( cluster->counts_peak[jpeak] < cluster->counts_peak[kpeak] )  cluster->counts_peak[jpeak] = 0;
					else                                                             cluster->counts_peak[kpeak] = 0;
				}

		   } //... end of loop on kpeak testing lower count peaks

	} //... end of loop on jpeak

						 
	//======== Cull out any 0, 0 centroids for odd-only-rows peaks

	kpeak  = 0;

	for( jpeak=0; jpeak<npeaks; jpeak++ )  {

		 if( cluster->counts_peak[jpeak] > 0 )  {  //... not spatially close peak
			 
			 if( cluster->measurements_odds[jpeak].rowcentroid != 0.0  ||        
			     cluster->measurements_odds[jpeak].colcentroid != 0.0      )  {  //... non-zero-zero peak found

			     cluster->measurements_odds[kpeak].rowcentroid = cluster->measurements_odds[jpeak].rowcentroid;
			     cluster->measurements_odds[kpeak].colcentroid = cluster->measurements_odds[jpeak].colcentroid;

			     kpeak++;
		     } 

		 }

	} //... end of loop on jpeak

	cluster->nclusters_odds = kpeak;

						 
	//======== Cull out any 0, 0 centroids for even-only-rows peaks

	kpeak  = 0;

	for( jpeak=0; jpeak<npeaks; jpeak++ )  {

		 if( cluster->counts_peak[jpeak] > 0 )  {  //... not spatially close peak
		 
			 if( cluster->measurements_even[jpeak].rowcentroid != 0.0  ||
			     cluster->measurements_even[jpeak].colcentroid != 0.0      )  {  //... non-zero-zero peak found

			     cluster->measurements_even[kpeak].rowcentroid = cluster->measurements_even[jpeak].rowcentroid;
			     cluster->measurements_even[kpeak].colcentroid = cluster->measurements_even[jpeak].colcentroid;

			     kpeak++;
			 }

		 } 

	} //... end of loop on jpeak

	cluster->nclusters_even = kpeak;

						 
	//======== Cull out any 0, 0 centroids for full-all-rows peaks

	kpeak  = 0;

	for( jpeak=0; jpeak<npeaks; jpeak++ )  {

		 if( cluster->counts_peak[jpeak] > 0 )  {  //... not spatially close peak
		 
			 if( cluster->measurements_full[jpeak].rowcentroid != 0.0  ||
			     cluster->measurements_full[jpeak].colcentroid != 0.0      )  {  //... non-zero-zero peak found

			     cluster->measurements_full[kpeak].rowcentroid = cluster->measurements_full[jpeak].rowcentroid;
			     cluster->measurements_full[kpeak].colcentroid = cluster->measurements_full[jpeak].colcentroid;

			     kpeak++;
			 }

		 } 

	} //... end of loop on jpeak

	cluster->nclusters_full = kpeak;
						 

}


//#######################################################################################
//  The ClusteringCentroids function refines the estimates of either the odd and even
//  row centroids OR the full rows centroid for the specific interleave mode given. The
//  other inputs are the centroid position guess from the function ClusteringExceedances
//  and the full frame image, the meteor-free mean image, and the standard deviation
//  image, all for the same user specified data type. The centroiding algorithm uses a 
//  row or column weighted sum of the whitened signal in an image patch around the last 
//  centroid estimated (the whitened signal is a mean-removed, variance-normalized image 
//  patch). The refinement is iterated several times to converge to a best centroid.
//
//=======================================================================================
//
//  INPUTS:
//    *ximage          Pointer to the image array (single frame)
//    *xtmean          Pointer to the meteor-free mean array (could be a image frame in 
//                        in the past or a temporal mean per pixel. For an image represented
//                        by differenced frames, the mean array is usually taken as zeros)
//    *xsigma          Pointer to the sigma array (could be a temporal standard deviation
//                        per pixel or the square root of a meteor-free image if necessary)
//    *cluster         Pointer to the clusterinfo structure containing:
//        nrows           Number of rows in the image frame
//        ncols           Number of columns in the image frame
//        coldim          Column wrap dimensionality of the image frame (often = ncols)
//        datatype        Data type specification of x* arrays (all arrays must = same type)
//                           IS_UNSIGNED_CHAR, IS_UNSIGNED_SHORT, IS_SHORT, IS_LONG, 
//                           IS_FLOAT, IS_DOUBLE
//        interleave      Interleave flag (0=progressive, 1=even/odd, 2=odd/even)
//        blocksize       Number of pixels comprising a cluster block's side dimension
//        nclusters_odds      Number of cluster peaks (odd  rows)
//        nclusters_even      Number of cluster peaks (even rows)
//        nclusters_full      Number of cluster peaks (full rows)
//        measurements_odds   Attributes for odd  rows (1st guess for centroids)
//        measurements_even   Attributes for even rows (1st guess for centroids)
//        measurements_full   Attributes for all  rows (1st guess for centroids)
//                   Note that attributes available are the first guess for row & col centroids
//
//  OUTPUT:
//    *cluster         Pointer to the clusterinfo structure containing:
//        measurements_odds   Attributes for odd  rows (refined centroids)
//        measurements_even   Attributes for even rows (refined for centroids)
//        measurements_full   Attributes for all  rows (refined for centroids)
//                   Note that attributes assigned are the refinements for row & col centroids
//
//  DEPENDENCIES:
//     None
//
//=======================================================================================

void  ClusteringCentroids( void *ximage, void *xtmean, void *xsigma, struct clusterinfo *cluster )
{
long    jpeak, npeaks, kiter, niterations, krowmode, nrowmodes, startrow, startcol, row, col, rcindex, msize;
double  numer, denom, signal, rowcentroid, colcentroid, variance;

unsigned char  *uc_image, *uc_tmean, *uc_sigma;  //--- pointers for any of the data types allowed
unsigned short *us_image, *us_tmean, *us_sigma;
short          *ss_image, *ss_tmean, *ss_sigma;
long           *sl_image, *sl_tmean, *sl_sigma;
float          *fl_image, *fl_tmean, *fl_sigma;
double         *db_image, *db_tmean, *db_sigma;



    //======== Function specific parameters

    niterations = 5;
	nrowmodes   = 3;  //... Includes all possible row combinations (krowmode) as follows:
	                  //        ALL rows (0),  ODD rows only (1),  EVEN rows only (2)


	//======== Set the array pointers according to data type enumerated

	switch ( cluster->datatype )
	{
		case IS_UNSIGNED_CHAR:
			uc_image = (unsigned char*)ximage;
			uc_tmean = (unsigned char*)xtmean;
			uc_sigma = (unsigned char*)xsigma;
		    break;
		case IS_UNSIGNED_SHORT:
			us_image = (unsigned short*)ximage;
			us_tmean = (unsigned short*)xtmean;
			us_sigma = (unsigned short*)xsigma;
		    break;
		case IS_SHORT:
			ss_image = (short*)ximage;
			ss_tmean = (short*)xtmean;
			ss_sigma = (short*)xsigma;
		    break;
		case IS_LONG:
			sl_image = (long*)ximage;
			sl_tmean = (long*)xtmean;
			sl_sigma = (long*)xsigma;
		    break;
		case IS_FLOAT:
			fl_image = (float*)ximage;
			fl_tmean = (float*)xtmean;
			fl_sigma = (float*)xsigma;
		    break;
		case IS_DOUBLE:
			db_image = (double*)ximage;
			db_tmean = (double*)xtmean;
			db_sigma = (double*)xsigma;
		    break;
		default:
			uc_image = (unsigned char*)ximage;
			uc_tmean = (unsigned char*)xtmean;
			uc_sigma = (unsigned char*)xsigma;
			break;
	}


	//======== Loop over each row mode and only refine those centroids that are appropriate to the interleave flag

    for( krowmode=0; krowmode<nrowmodes; krowmode++ )  {

	     if( krowmode == 0  &&  cluster->interleave == 1 )  continue;  // skip all rows centroid if interleaved
	     if( krowmode == 0  &&  cluster->interleave == 2 )  continue;  // skip all rows centroid if interleaved
		 if( krowmode == 1  &&  cluster->interleave == 0 )  continue;  // skip odd & even row centroids if progressive scan
		 if( krowmode == 2  &&  cluster->interleave == 0 )  continue;  // skip odd & even row centroids if progressive scan

		 //=-=-=-=-= Select the number of clustered peaks to process
		 
		 switch ( krowmode )
		 {
			case 1:  //... Interleaved (odd rows only)
		        npeaks = cluster->nclusters_odds;
			    break;
			case 2:  //... Interleaved (even rows only)
		        npeaks = cluster->nclusters_even;
			    break;
			default: //... Progressive scan (all rows)
		        npeaks = cluster->nclusters_full;
			    break;
		 }


         //=-=-=-=-=  Loop over each peak in the clustered list
	     //               Loop to iteratively refine the centroids
		 
        for( jpeak=0; jpeak<npeaks; jpeak++ )  {

              for( kiter=0; kiter<niterations; kiter++ )  {

			       //-------- Get the next peak centroid estimate for this iteration for the
				   //           appropriate frame/fields mode.

				   switch ( krowmode )
				   {
				       case 1:  //... Interleaved (odd rows only)
		                   startrow = (long)cluster->measurements_odds[jpeak].rowcentroid - cluster->blocksize / 4;
		                   startcol = (long)cluster->measurements_odds[jpeak].colcentroid - cluster->blocksize / 4;
						   msize    = cluster->blocksize / 2;
						   break;
				       case 2:  //... Interleaved (even rows only)
		                   startrow = (long)cluster->measurements_even[jpeak].rowcentroid - cluster->blocksize / 4;
		                   startcol = (long)cluster->measurements_even[jpeak].colcentroid - cluster->blocksize / 4;
						   msize    = cluster->blocksize / 2;
						   break;
				       default: //... Progressive scan (all rows)
		                   startrow = (long)cluster->measurements_full[jpeak].rowcentroid - cluster->blocksize / 2;
		                   startcol = (long)cluster->measurements_full[jpeak].colcentroid - cluster->blocksize / 2;
						   msize    = cluster->blocksize;
						   break;
				   }



			       //-------- Check bounds of the starting row and column

			       if( startrow < 0 )  startrow = 0;
			       if( startcol < 0 )  startcol = 0;

			       if( startrow > cluster->nrows - msize )  startrow = cluster->nrows - msize;
			       if( startcol > cluster->ncols - msize )  startcol = cluster->ncols - msize;


			       //-------- Clear the row and column sums for the signal

			       for( row=startrow; row<startrow+msize; row++ )  cluster->sigsum_across_row[row] = 0.0;
			       for( col=startcol; col<startcol+msize; col++ )  cluster->sigsum_downcolumn[col] = 0.0;


			       //-------- Loop over each pixel in the blocksize x blocksize patch to sum the signal
			       //            across rows or down columns. The signal is whitened such that:
			       //                          signal = [ x - <x> ] / variance
			       //            Note that due to C indexing rows 0, 2, 4, ... are odd rows
			       //                               and that rows 1, 3, 5, ... are even rows

			       for( row=startrow; row<startrow+msize; row++ )  {

					    if( krowmode == 1  &&  row % 2 == 1 )  continue;  //...skip even rows for odd rows centroid
					    if( krowmode == 2  &&  row % 2 == 0 )  continue;  //...skip odd rows for even rows centroid

			            rcindex = row * cluster->coldim + startcol;

						//........ Compute signal partial sums. Branch according to data type.

						switch ( cluster->datatype )
						{
						    case IS_UNSIGNED_CHAR:
			                    for( col=startcol; col<startcol+msize; col++ )  {

							         variance = (double)uc_sigma[rcindex] * (double)uc_sigma[rcindex] + cluster->tiny_variance;
					                 signal = ( (double)uc_image[rcindex] - (double)uc_tmean[rcindex++] ) / variance;

						             cluster->sigsum_across_row[row] += signal;
						             cluster->sigsum_downcolumn[col] += signal;

			                    } 
								break;

						    case IS_UNSIGNED_SHORT:
			                    for( col=startcol; col<startcol+msize; col++ )  {

							         variance = (double)us_sigma[rcindex] * (double)us_sigma[rcindex] + cluster->tiny_variance;
					                 signal = ( (double)us_image[rcindex] - (double)us_tmean[rcindex++] ) / variance;

						             cluster->sigsum_across_row[row] += signal;
						             cluster->sigsum_downcolumn[col] += signal;

			                    } 
								break;

						    case IS_SHORT:
			                    for( col=startcol; col<startcol+msize; col++ )  {

							         variance = (double)ss_sigma[rcindex] * (double)ss_sigma[rcindex] + cluster->tiny_variance;
					                 signal = ( (double)ss_image[rcindex] - (double)ss_tmean[rcindex++] ) / variance;

						             cluster->sigsum_across_row[row] += signal;
						             cluster->sigsum_downcolumn[col] += signal;

			                    } 
								break;

						    case IS_LONG:
			                    for( col=startcol; col<startcol+msize; col++ )  {

							         variance = (double)sl_sigma[rcindex] * (double)sl_sigma[rcindex] + cluster->tiny_variance;
					                 signal = ( (double)sl_image[rcindex] - (double)sl_tmean[rcindex++] ) / variance;

						             cluster->sigsum_across_row[row] += signal;
						             cluster->sigsum_downcolumn[col] += signal;

			                    } 
								break;

						    case IS_FLOAT:
			                    for( col=startcol; col<startcol+msize; col++ )  {

							         variance = (double)fl_sigma[rcindex] * (double)fl_sigma[rcindex] + cluster->tiny_variance;
					                 signal = ( (double)fl_image[rcindex] - (double)fl_tmean[rcindex++] ) / variance;

						             cluster->sigsum_across_row[row] += signal;
						             cluster->sigsum_downcolumn[col] += signal;

			                    } 
								break;

						    case IS_DOUBLE:
			                    for( col=startcol; col<startcol+msize; col++ )  {

							         variance = (double)db_sigma[rcindex] * (double)db_sigma[rcindex] + cluster->tiny_variance;
					                 signal = ( (double)db_image[rcindex] - (double)db_tmean[rcindex++] ) / variance;

						             cluster->sigsum_across_row[row] += signal;
						             cluster->sigsum_downcolumn[col] += signal;

			                    } 
								break;

						    default:  //---- unsigned char
			                    for( col=startcol; col<startcol+msize; col++ )  {

							         variance = (double)uc_sigma[rcindex] * (double)uc_sigma[rcindex] + cluster->tiny_variance;
					                 signal = ( (double)uc_image[rcindex] - (double)uc_tmean[rcindex++] ) / variance;

						             cluster->sigsum_across_row[row] += signal;
						             cluster->sigsum_downcolumn[col] += signal;

			                    } 
								break;

						} //... end of switch/case on data type


			       } //... end of row loop


			       //-------- Row weighted sums, pre-screened by interleave mode, 
				   //            and the resultant row centroid computation
			  
			       numer = 0.0;
			       denom = 0.0;

			       for( row=startrow; row<startrow+msize; row++ )  {

					    if( krowmode == 1  &&  row % 2 == 1 )  continue;  //...skip even rows for odd rows centroid
					    if( krowmode == 2  &&  row % 2 == 0 )  continue;  //...skip odd rows for even rows centroid

						numer += cluster->sigsum_across_row[row] * (double)row;
				        denom += cluster->sigsum_across_row[row];

			       }

			       if( denom != 0.0 )  rowcentroid = numer / denom;
			       else                rowcentroid = (double)( startrow + msize / 2 );

				   
				   //-------- Column weighted sums, which are already screened into odd-only, 
				   //             even-only, or all-rows column sums, and resultant column centroid
			  			  
			       numer = 0.0;  
			       denom = 0.0;

			       for( col=startcol; col<startcol+msize; col++ )  {

				        numer += cluster->sigsum_downcolumn[col] * (double)col;
				        denom += cluster->sigsum_downcolumn[col];

			       }

			       if( denom != 0.0 )  colcentroid = numer / denom;
			       else                colcentroid = (double)( startcol + msize / 2 );


				   //-------- Save centroid estimate from this iteration to the 
				   //            appropriate frame/fields mode.

				   switch ( krowmode )
				   {
				       case 1:  //... Interleaved (odd rows only)
		                   cluster->measurements_odds[jpeak].rowcentroid = rowcentroid;
		                   cluster->measurements_odds[jpeak].colcentroid = colcentroid;
						   break;
				       case 2:  //... Interleaved (even rows only)
		                   cluster->measurements_even[jpeak].rowcentroid = rowcentroid;
		                   cluster->measurements_even[jpeak].colcentroid = colcentroid;
						   break;
				       default: //... Progressive scan (all rows)
		                   cluster->measurements_full[jpeak].rowcentroid = rowcentroid;
		                   cluster->measurements_full[jpeak].colcentroid = colcentroid;
						   break;
				   }


		      } //... end of refinement iteration loop

	     } //... end of clustered peak loop for given mode

    } //... end of interleave mode loop

}


//#######################################################################################
//  The ClusteringDetector function estimates various attributes of each cluster such as
//  the signal-independent matched-filter squared-numerator metric for detection, the
//  entropy, orientation measures, exceedance pixel count, and saturated count. 
//  The inputs are the refined centroids from the function ClusteringCentroids, the full
//  frame image, the meteor-free mean image, and the standard deviation image, all for 
//  the same user specified data type. The detection algorithm uses a signal independent
//  matched filter with squared numerator integrated over an image sub-patch and then 
//  converted to dB (done on a per frame or per field basis).
//
//         SsqdB = 10 * log10( signal^2 / noise )
//            signal = [ x - <x> ] / variance
//            noise  = 1 / variance
//
//  The function also performs a detection threshold and saves the detection metric
//  dependent on the sensor mode (either full frame for progressive scan OR separate 
//  measurement lists of odd and even fields for interleaved). The centroid/detection
//  measurement list will be culled independently for each mode so they may not have
//  the same number of measurements for each field (if interleaved).
//
//=======================================================================================
//
//  INPUTS:
//    *ximage          Pointer to the image array (single frame)
//    *xtmean          Pointer to the meteor-free mean array (could be a image frame in 
//                        in the past or a temporal mean per pixel. For an image represented
//                        by differenced frames, the mean array is usually taken as zeros)
//    *xsigma          Pointer to the sigma array (could be a temporal standard deviation
//                        per pixel or the square root of a meteor-free image if necessary)
//    *cluster         Pointer to the clusterinfo structure containing:
//        nrows                 Number of rows in the image frame
//        ncols                 Number of columns in the image frame
//        coldim                Column wrap dimensionality of the image frame (often = ncols)
//        datatype        Data type specification of x* arrays (all arrays must = same type)
//                           IS_UNSIGNED_CHAR, IS_UNSIGNED_SHORT, IS_SHORT, IS_LONG, 
//                           IS_FLOAT, IS_DOUBLE
//        interleave            Interleave flag (0=progressive, 1=odd/even, 2=even/odd)
//        blocksize             Number of pixels comprising a cluster block's side dimension
//        tiny_variance   Smallest non-zero variance to be encountered for given data type
//        saturation      Largest value to e encountered for a given data type
//        sfactor         Standard deviation factor to find exceedance count attribute
//                        threshold = <x> + sfactor * sigma
//        SsqdB_threshold  MF squared threshold for detection in dB
//        nclusters_odds      Number of cluster peaks (odd  rows)
//        nclusters_even      Number of cluster peaks (even rows)
//        nclusters_full      Number of cluster peaks (full rows)
//        measurements_odds   Attributes for odd  rows (refined centroids)
//        measurements_even   Attributes for even rows (refined centroids)
//        measurements_full   Attributes for all  rows (refined centroids)
//                   Note that attributes available are the refinements for row & col centroids
//
//  OUTPUT:
//    *cluster         Pointer to the clusterinfo structure containing:
//        nclusters_odds      Number of cluster peaks (culled if interleaved)
//        nclusters_even      Number of cluster peaks (culled if interleaved)
//        nclusters_full      Number of cluster peaks (culled if progressive scan)
//        measurements_odds   Attributes for odd  rows (culled if interleaved)
//        measurements_even   Attributes for even rows (culled if interleaved)
//        measurements_full   Attributes for all  rows (culled if progressive scan)
//                     Note that attributes assigned are the culled nclusters, SsqdB, entropy,
//                        orientation, exceedance count, saturation count, ...
//
//  DEPENDENCIES:
//     None
//
//=======================================================================================

void  ClusteringDetector( void *ximage, void *xtmean, void *xsigma, struct clusterinfo *cluster )
{
long    jpeak, npeaks, kpeak, krowmode, nrowmodes;
long    startrow, startcol, row, col, rcindex, msize;
long    ecount, scount, pcount, hcount, khisto, histogram[256], nhisto;
double  centrrow, centrcol, drow, dcol, drhosq, rsq, resum, imsum;
double  PCD_angle, PCD_width, PCD_mag, PCD_magmax, PCD_wmax, PCD_r;
double  Hueckel_angle;
double  pixval, xmean, signal, noise, stddev, variance, SsqN, SsqNdB, p, entropy;

unsigned char  *uc_image, *uc_tmean, *uc_sigma;
unsigned short *us_image, *us_tmean, *us_sigma;
short          *ss_image, *ss_tmean, *ss_sigma;
long           *sl_image, *sl_tmean, *sl_sigma;
float          *fl_image, *fl_tmean, *fl_sigma;
double         *db_image, *db_tmean, *db_sigma;


    //======== Function specific parameters

	nrowmodes = 3;  //... Includes all possible row combinations (krowmode) as follows:
	                //        ALL rows (0),  ODD rows only (1),  EVEN rows only (2)

	nhisto = 256;   //... Number of histogram bins for entropy calculation

	rsq = (double)cluster->blocksize * (double)cluster->blocksize / 4.0;


	//======== Set the array pointers according to the data type enumerated

	switch ( cluster->datatype )
	{
		case IS_UNSIGNED_CHAR:
			uc_image = (unsigned char*)ximage;
			uc_tmean = (unsigned char*)xtmean;
			uc_sigma = (unsigned char*)xsigma;
		    break;
		case IS_UNSIGNED_SHORT:
			us_image = (unsigned short*)ximage;
			us_tmean = (unsigned short*)xtmean;
			us_sigma = (unsigned short*)xsigma;
		    break;
		case IS_SHORT:
			ss_image = (short*)ximage;
			ss_tmean = (short*)xtmean;
			ss_sigma = (short*)xsigma;
		    break;
		case IS_LONG:
			sl_image = (long*)ximage;
			sl_tmean = (long*)xtmean;
			sl_sigma = (long*)xsigma;
		    break;
		case IS_FLOAT:
			fl_image = (float*)ximage;
			fl_tmean = (float*)xtmean;
			fl_sigma = (float*)xsigma;
		    break;
		case IS_DOUBLE:
			db_image = (double*)ximage;
			db_tmean = (double*)xtmean;
			db_sigma = (double*)xsigma;
		    break;
		default: //... unsigned char
			uc_image = (unsigned char*)ximage;
			uc_tmean = (unsigned char*)xtmean;
			uc_sigma = (unsigned char*)xsigma;
			break;
	}


	//======== Loop over each row mode and only perform detection appropriate to the interleave mode

	for( krowmode=0; krowmode<nrowmodes; krowmode++ )  {

		 //=-=-=-=-= Do EITHER progressive scan OR interleave odd & even detection processing

	     if( krowmode == 0  &&  cluster->interleave == 1 )  continue;  // skip all rows detection if interleaved
	     if( krowmode == 0  &&  cluster->interleave == 2 )  continue;  // skip all rows detection if interleaved
		 if( krowmode == 1  &&  cluster->interleave == 0 )  continue;  // skip odd & even row detection if progressive scan
		 if( krowmode == 2  &&  cluster->interleave == 0 )  continue;  // skip odd & even row detection if progressive scan


		      //======== Select the number of clustered peaks to process and
		      //             reset running centroid/detection count to zero.
		 
		      switch ( krowmode )
		      {
			     case 1:  //... Interleaved (odd rows only)
		             npeaks = cluster->nclusters_odds;
				     cluster->nclusters_odds = 0;
			         break;
			     case 2:  //... Interleaved (even rows only)
		             npeaks = cluster->nclusters_even;
				     cluster->nclusters_even = 0;
			         break;
			     default: //... Progressive scan (all rows)
		             npeaks = cluster->nclusters_full;
				     cluster->nclusters_full = 0;
			         break;
		      }


	          //======== Loop over each peak in the clustered list for the specified mode

              for( jpeak=0; jpeak<npeaks; jpeak++ )  {

			       //-------- Get the latest centroid estimate for the
				   //           appropriate frame/fields mode.

				   switch ( krowmode )
				   {
				       case 1:  //... Interleaved (odd rows only)
		                   centrrow =       cluster->measurements_odds[jpeak].rowcentroid;
		                   centrcol =       cluster->measurements_odds[jpeak].colcentroid;
		                   startrow = (long)cluster->measurements_odds[jpeak].rowcentroid - cluster->blocksize / 4;
		                   startcol = (long)cluster->measurements_odds[jpeak].colcentroid - cluster->blocksize / 4;
						   msize    = cluster->blocksize / 2;
						   break;
				       case 2:  //... Interleaved (even rows only)
		                   centrrow =       cluster->measurements_even[jpeak].rowcentroid;
		                   centrcol =       cluster->measurements_even[jpeak].colcentroid;
		                   startrow = (long)cluster->measurements_even[jpeak].rowcentroid - cluster->blocksize / 4;
		                   startcol = (long)cluster->measurements_even[jpeak].colcentroid - cluster->blocksize / 4;
						   msize    = cluster->blocksize / 2;
						   break;
				       default: //... Progressive scan (all rows)
		                   centrrow =       cluster->measurements_full[jpeak].rowcentroid;
		                   centrcol =       cluster->measurements_full[jpeak].colcentroid;
		                   startrow = (long)cluster->measurements_full[jpeak].rowcentroid - cluster->blocksize / 2;
		                   startcol = (long)cluster->measurements_full[jpeak].colcentroid - cluster->blocksize / 2;
						   msize    = cluster->blocksize;
						   break;
				   }


			       //-------- Check bounds of the starting row and column

			       if( startrow < 0 )  startrow = 0;
			       if( startcol < 0 )  startcol = 0;

			       if( startrow > cluster->nrows - msize )  startrow = cluster->nrows - msize;
			       if( startcol > cluster->ncols - msize )  startcol = cluster->ncols - msize;


			       //-------- Clear the row and column sums for the noise and signal

			       for( col=startcol; col<startcol+msize; col++ )  {
			            cluster->noisum_downcolumn[col] = 0.0;
					    cluster->sigsum_downcolumn[col] = 0.0;
				   }


				   //-------- Clear the histogram counter for the entropy calculation, zero the
				   //            number of pixels contributing to the histogram, zero the
				   //            exceedance and saturation pixel counters, zero the PCD real
				   //            and imaginary sums.

				   for( khisto=0; khisto<nhisto; khisto++ )  histogram[khisto] = 0;

			       hcount = 0;
			       ecount = 0;
			       scount = 0;
				   pcount = 0;

				   resum = 0.0;
				   imsum = 0.0;


			       //-------- Loop over each pixel in the blocksize x blocksize patch to sum the signal and noise
			       //            across rows or down columns. The signal is whitened such that:
			       //                          signal = [ x - <x> ] / variance
				   //            The noise is the inverse of a diaginal covariance such that:
				   //                          noise = 1 / variance 
			       //            Note that due to C indexing rows 0, 2, 4, ... are odd rows
			       //                               and that rows 1, 3, 5, ... are even rows

			       for( row=startrow; row<startrow+msize; row++ )  {

					    if( krowmode == 1  &&  row % 2 == 1 )  continue;  //...skip even rows for odd rows detection
					    if( krowmode == 2  &&  row % 2 == 0 )  continue;  //...skip odd rows for even rows detection

			            rcindex = row * cluster->coldim + startcol;


						//........ Loop over each column extracting standard deviation and demeaned signal components. 
						//            Branch according to data type.

			            for( col=startcol; col<startcol+msize; col++ )  {

						     switch ( cluster->datatype )
						     {
						         case IS_UNSIGNED_CHAR:
							         stddev = (double)uc_sigma[rcindex];
					                 pixval = (double)uc_image[rcindex];
					                 xmean  = (double)uc_tmean[rcindex++];
								     break;

						         case IS_UNSIGNED_SHORT:
							         stddev = (double)us_sigma[rcindex];
					                 pixval = (double)us_image[rcindex];
					                 xmean  = (double)us_tmean[rcindex++];
								     break;

						         case IS_SHORT:
							         stddev = (double)ss_sigma[rcindex];
					                 pixval = (double)ss_image[rcindex];
					                 xmean  = (double)ss_tmean[rcindex++];
								     break;

						         case IS_LONG:
							         stddev = (double)sl_sigma[rcindex];
					                 pixval = (double)sl_image[rcindex];
					                 xmean  = (double)sl_tmean[rcindex++];
								     break;

						         case IS_FLOAT:
							         stddev = (double)fl_sigma[rcindex];
					                 pixval = (double)fl_image[rcindex];
					                 xmean  = (double)fl_tmean[rcindex++];
								     break;

						         case IS_DOUBLE:
							         stddev = (double)db_sigma[rcindex];
					                 pixval = (double)db_image[rcindex];
					                 xmean  = (double)db_tmean[rcindex++];
								     break;

						         default:  //---- unsigned char
							         stddev = (double)uc_sigma[rcindex];
					                 pixval = (double)uc_image[rcindex];
					                 xmean  = (double)uc_tmean[rcindex++];
								     break;

						     } //... end of switch/case on data type


						     //........ Compute noise and signal sums 

						     variance = stddev * stddev  +  cluster->tiny_variance; 
						     noise    = 1.0 / variance;
						     signal   = ( pixval - xmean ) * noise;

						     cluster->noisum_downcolumn[col] += noise;
						     cluster->sigsum_downcolumn[col] += signal;


							 //........ Get saturation count, exceedance count, and for every exceedance compute
							 //           the running sums of the PCD binary convolution (real and imaginary)

							 pcount++;

							 if( pixval >= cluster->saturation )  scount++;

							 if( pixval >= xmean + cluster->sfactor * stddev )  {
								 
								 ecount++;

								 drow = (double)row - centrrow;
								 dcol = (double)col - centrcol;
								 drhosq = drow * drow + dcol * dcol;

								 if( drhosq > 0.0  &&  drhosq <= rsq )  {
								     resum += ( drow * drow - dcol * dcol ) / drhosq;
									 imsum += 2.0 * drow * dcol / drhosq;
								 }

							 }


							 //........ Entropy histogram counter of pixel levels (only nhisto bins)
							 //            Avoids zero signal pixel values
							 //            Avoids saturated pixel values

							 khisto = (long)( (double)nhisto * pixval / cluster->saturation );

							 if( khisto > 0  &&  khisto < nhisto-1 )  {
								 histogram[khisto] += 1;
								 hcount++;
							 }

							 //......................................................................

						} //... end of column loop

			       } //... end of row loop


			       //-------- Signal independent squared estimate over noise
			  
			       signal = 0.0;
			       noise  = 0.0;

			       for( col=startcol; col<startcol+msize; col++ )  {

				        signal += cluster->sigsum_downcolumn[col];
				        noise  += cluster->noisum_downcolumn[col];

			       }

			       if( noise != 0.0 )  SsqN = signal * signal / noise;
			       else                SsqN = 0.0;

			       SsqNdB = 10.0 * log10( SsqN + 1.0e-30 );


				   //-------- Compute the entropy with output normalized to range = [0,1]

				   if( hcount == 0 )  entropy = 1.0;
				   else  {

					   entropy = 0.0;

				       for( khisto=0; khisto<nhisto; khisto++ )  {

					        p = (double)histogram[khisto] / (double)hcount;

					        entropy += -p * log10( p + 1.0e-30 );

				       }

				       entropy /= log10( (double)nhisto );

				   }


				   //-------- Compute the phase coded disk (PCD) line orientation and width

				   PCD_angle = 0.5 * 57.29577951 * atan2( imsum, resum );

				   PCD_r = cluster->blocksize / 2.0;

				   PCD_magmax = 0.72461 * PCD_r * PCD_r;

				   PCD_wmax = 0.78847 * PCD_r;

				   PCD_mag = sqrt( resum * resum + imsum * imsum );

				   if( PCD_mag > PCD_magmax )  PCD_mag = PCD_magmax;

				   PCD_width = ( 0.5558 * PCD_mag * PCD_r * PCD_r - 0.5865 * PCD_mag * PCD_mag )
						     / ( PCD_r * PCD_r * PCD_r - 1.202 * PCD_mag * PCD_r );


				   //-------- Compute the Hueckle line orientation and width

				   Hueckel_angle = 0.0;


			       //-------- If a detection, then save the matched filter squared output 
				   //           to the appropriate frame/fields mode and increment the
			       //           number of "detected" cluster peaks.

			       if( SsqNdB >= cluster->SsqNdB_threshold )  { 

			           switch ( krowmode )
			           {
			                case 1:  //... Interleaved (odd rows only)
								kpeak = cluster->nclusters_odds;
		                        cluster->measurements_odds[kpeak].sumsignal     = signal;
		                        cluster->measurements_odds[kpeak].sumnoise      = noise;
		                        cluster->measurements_odds[kpeak].SsqNdB        = SsqNdB;
		                        cluster->measurements_odds[kpeak].entropy       = entropy;
		                        cluster->measurements_odds[kpeak].nexceed       = ecount;
		                        cluster->measurements_odds[kpeak].nsaturated    = scount;
		                        cluster->measurements_odds[kpeak].npixels       = pcount;
		                        cluster->measurements_odds[kpeak].PCD_angle     = PCD_angle;
		                        cluster->measurements_odds[kpeak].PCD_width     = PCD_width;
		                        cluster->measurements_odds[kpeak].Hueckel_angle = Hueckel_angle;
					            cluster->nclusters_odds++;
					            break;
			                case 2:  //... Interleaved (even rows only)
								kpeak = cluster->nclusters_even;
		                        cluster->measurements_even[kpeak].sumsignal     = signal;
		                        cluster->measurements_even[kpeak].sumnoise      = noise;
		                        cluster->measurements_even[kpeak].SsqNdB        = SsqNdB;
		                        cluster->measurements_even[kpeak].entropy       = entropy;
		                        cluster->measurements_even[kpeak].nexceed       = ecount;
		                        cluster->measurements_even[kpeak].nsaturated    = scount;
		                        cluster->measurements_even[kpeak].npixels       = pcount;
		                        cluster->measurements_even[kpeak].PCD_angle     = PCD_angle;
		                        cluster->measurements_even[kpeak].PCD_width     = PCD_width;
		                        cluster->measurements_even[kpeak].Hueckel_angle = Hueckel_angle;
					            cluster->nclusters_even++;
					            break;
			                default: //... Progressive scan (all rows)
								kpeak = cluster->nclusters_full;
		                        cluster->measurements_full[kpeak].sumsignal     = signal;
		                        cluster->measurements_full[kpeak].sumnoise      = noise;
		                        cluster->measurements_full[kpeak].SsqNdB        = SsqNdB;
		                        cluster->measurements_full[kpeak].entropy       = entropy;
		                        cluster->measurements_full[kpeak].nexceed       = ecount;
		                        cluster->measurements_full[kpeak].nsaturated    = scount;
		                        cluster->measurements_full[kpeak].npixels       = pcount;
		                        cluster->measurements_full[kpeak].PCD_angle     = PCD_angle;
		                        cluster->measurements_full[kpeak].PCD_width     = PCD_width;
		                        cluster->measurements_full[kpeak].Hueckel_angle = Hueckel_angle;
					            cluster->nclusters_full++;
					            break;
			           }

				   } //... end if a detection


	          } //... end of "jpeak" clustered peak loop for a given interleave/progressive mode

    } //... end of interleave mode loop


}


//#######################################################################################
//  The ClusteringFreeMemory function frees any memory allocated by the function
//  ClusteringInitialization that is embedded in the clusterinfo structure. The free 
//  memory function may be called once at the end, after all the image processing has 
//  been completed for all the frames.
//
//=======================================================================================
//
//  INPUTS:
//     cluster      Clusterinfo structure pointer containing allocated 
//                     array and vector memory that is to be released.
//
//=======================================================================================

void  ClusteringFreeMemory( struct clusterinfo *cluster )
{
	free( cluster->block_counts_odds );
	free( cluster->block_rowsum_odds );
	free( cluster->block_colsum_odds );

	free( cluster->block_counts_even );
	free( cluster->block_rowsum_even );
	free( cluster->block_colsum_even );

	free( cluster->sigsum_across_row );
	free( cluster->sigsum_downcolumn );
	free( cluster->noisum_downcolumn );

	free( cluster->measurements_odds );
	free( cluster->measurements_even );
	free( cluster->measurements_full );

	free( cluster->counts_peak );
	free( cluster->blkrow_peak );
	free( cluster->blkcol_peak );

	free( cluster->hueckel_coef );



}


//#######################################################################################
