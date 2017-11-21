//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Version supporting 204  -  June 16, 2017
//
// Series of image processing functions for SHORT data (unsigned and signed):
//
//    Dark frame ingest/generation and estimating/applying a flat field from
//    multi-frame and star-filled image files.
//            ReadDarkField
//            ZeroDarkField
//            ImageOnlyBelowPcut
//            ZapStars
//            FlatApplied2Image
//            WriteFlatFieldMultiplier
//            ReadFlatFieldMultiplier
//            UnityFlatFieldMultiplier
//
//    Estimate the global background level and equalize the image
//            GlobalBackgndFromHistogram
//            EqualizeImageWithGlobalBackgnd
//
//    Decimating an image by NxN pixels by computing the mean of
//    the N highest pixel values per tiled NxN block
//            DecimateInitialization
//            DecimateFreeMemory
//            DecimateNxNviaMeanN
//
//    Obtain exceedance pixels based on thresholding per frame
//            ThresholdPreCompute
//            ThresholdExceedances
//
//    Matched filter functions
//            GaussianPSF
//            FreeMemoryPSF
//            GaussianPSFtable
//            FreeMemoryPSFtable
//            ConvolvePSF
//            Demean
//            DemeanWhiten
//            DemeanWhitenConvolve
//            CovarianceInverse
//            GenerateMFtemplate
//            GenerateMFtemplatePSF
//            GenerateMFtemplatePSFsubpixel
//            GenerateMFsubtemplateLine
//            GenerateMFtemplateAOI
//            BilinearInterpWeights
//            FreeMFtemplate
//            FreeMFsubtemplate
//            IntegratedSignalFromDemeaned
//            MaximumSignalFromDemeaned
//            BackgroundVarianceFromDemeaned
//            AverageSignalFromWhitened
//            PSFdimension
//            FrameMatchedFilter
//            MultiFrameMatchedFilter
//            MultiFrameMatchedFilterRefinement
//            FrameMatchedFilterInterpolated
//            MultiFrameMatchedFilterInterpolated
//            MultiFrameMatchedFilterRefinementInterpolated
//            MultiFrameMeasurements
//
//    Miscelaneous functions
//            Allocate2Darray
//            Free2Darray
//            FindClosestRingBufferTime
//            SubtractImage_BiasResult
//            DuplicationTest
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#define  UCHAR         0    // unsigned char
#define  USHORT        1    // unsigned short
#define  SSHORT        2    // short (signed)
#define  UINT          3    // unsigned int
#define  IINT          4    // int (signed)
#define  ULONG         5    // unsigned long
#define  LLONG         6    // long (signed)
#define  FFLOAT        7    // float
#define  DDOUBLE       8    // double

#define  OUTSIDEFOV    0
#define  INSIDEFOV     1

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

struct  decimator
{
	int                ndecim;       // Decimation factor applied to both rows and cols
	int                nrows_image;  // Number of rows in the original full image
	int                ncols_image;  // Number of cols in the original full image
	int                nrows_decim;  // Number of rows in the decimated image
	int                ncols_decim;  // Number of cols in the decimated image
	unsigned short    *image_ptr;    // Pointer to the decimated image array
};


struct  PointSpreadFunc
{
	int       pdimen;      // Size of PSF in pixels (square kernel, odd#)
	double    phalfwidth;  // Half-width at half-peak for a Gaussian PSF in pixels
	double   *pvalue;      // Array of PSF values normalized sum to unity (vector)

};


struct  PointSpreadLookupTable
{
	int       minpdimen;   // Minimum size of PSF in pixels (square kernel, odd#)
	int       maxkrsq;     // Maximum 100*r^2 index in table
	double    phalfwidth;  // Half-width at half-peak for a Gaussian PSF in pixels
	double   *ptable;      // Vector of PSF values exp(-0.693*r^2/h^2) hashed by 100*r^2
	                       //     So for r=2, hash index is pvalue[400]
};


struct  MFtemplate
{
	long      npixels;     // Number of pixels in the template
	long     *pixeloffset; // Pointer to offsets re 1st array pixel of the template
	double   *pixelsignal; // Pointer to the hypothesized signal values of the template
};


struct  MFsubtemplate
{
	long      nsubpixels;  // Number of subpixel positions in the template
	double    subpixelval; // Normalized subpixel value (integrated over a pixel ==> 1)
	long     *pixeloffset; // Pointer to nearest (upper left 11) integer pixel position
	long      offset12;    // Offset from pixel 11 to pixel 12
	long      offset21;    // Offset from pixel 11 to pixel 21
	long      offset22;    // Offset from pixel 11 to pixel 22
	double   *w11;         // Pointer to 1st weight in bilinear interpolation
	double   *w12;         // Pointer to 2nd weight in bilinear interpolation
	double   *w21;         // Pointer to 3rd weight in bilinear interpolation
	double   *w22;         // Pointer to 4th weight in bilinear interpolation
};


struct  MFhypothesis
{
	double    Xrow;        // Row position in pixels
	double    Xcol;        // Col position in pixels
	double    Vrow;        // Row velocity in pixels/second
	double    Vcol;        // Col velocity in pixels/second
	double    TRT;         // Signal dependent Max Likelihood Estimate
	double    MLEdb;       // Signal independent MLE in dB
};


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int     ReadDarkField( unsigned short *darkfield_ptr, int *mrows, int *mcols, char *filename );

void    ZeroDarkField( unsigned short *darkfield_ptr, int nrows, int ncols );

void    ImageOnlyBelowPcut( unsigned short *image_ptr, int nrows, int ncols, int Pcut );

void    ZapStarRegions( unsigned short *image_ptr, int nrows, int ncols, int halfsize, int nstars );

void    FlatApplied2Image( unsigned short *image_ptr, double *flatfield_ptr, unsigned short *flattened_ptr, int nrows, int ncols );

int     WriteFlatFieldMultiplier( double *flatfield_ptr, int nrows, int ncols, char *filename );

int     ReadFlatFieldMultiplier( double *flatfield_ptr, int *mrows, int *mcols, char *filename );

void    UnityFlatFieldMultiplier( double *flatfield_ptr, int nrows, int ncols );



double  GlobalBackgndFromHistogram( unsigned short *image_ptr, int nrows, int ncols, int Plow, int Phigh );

void    EqualizeImageWithGlobalBackgnd( unsigned short *image_ptr, unsigned short *equal_ptr,
	                                    double image_backgnd, double ref_backgnd, int nrows, int ncols );



int     DecimateInitialization( int ndecim, int nrows_image, int ncols_image, struct decimator *decim );

int     DecimateFreeMemory( struct decimator *decim );

int     DecimateNxNviaMeanN( unsigned short *image_ptr, struct decimator *decim );



void    ThresholdPreCompute( int nrows_image, int ncols_image,
	                         unsigned short *mean_ptr, unsigned short *sigma_ptr, double sfactor,
							 unsigned short *threshold_ptr );

void    ThresholdExceedances( unsigned short *image_ptr, int nrows_image, int ncols_image,
	                          unsigned short *mean_ptr, unsigned short *sigma_ptr, double sfactor,
	                          long *pixelcount_ptr, long *mpoffset_ptr );



void    GaussianPSF( int pdimen, double phalfwidth, struct PointSpreadFunc *psf );

void    FreeMemoryPSF( struct PointSpreadFunc *psf );

void    GaussianPSFtable( int minpdimen, double phalfwidth, struct PointSpreadLookupTable *psf_lookup );

void    FreeMemoryPSFtable( struct PointSpreadLookupTable *psf_lookup );

void    ConvolvePSF( double *dimage_ptr, struct PointSpreadFunc *psf, double *convolved_ptr, int nrows, int ncols );

void    Demean( unsigned short *image_ptr, unsigned short *mean_ptr, double *demeaned_ptr, int nrows, int ncols );

void    DemeanWhiten( unsigned short *image_ptr, unsigned short *mean_ptr, unsigned short *sigma_ptr, double *whitened_ptr, int nrows, int ncols );

void    DemeanWhitenConvolve( unsigned short *image_ptr, unsigned short *mean_ptr, unsigned short *sigma_ptr, double *convolved_ptr, struct PointSpreadFunc *psf, double *workarray_ptr, int nrows, int ncols );

void    CovarianceInverse( unsigned short *sigma_ptr, double *covarinv_ptr, int nrows, int ncols );

void    GenerateMFtemplate( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, struct MFtemplate *template_ptr, int nrows, int ncols );

void    GenerateMFtemplatePSF( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, struct PointSpreadFunc *psf, struct MFtemplate *template_ptr, int nrows, int ncols );

void    GenerateMFtemplateAOI( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, int aoi_dimension, struct MFtemplate *template_ptr, int nrows, int ncols );

void    GenerateMFtemplatePSFsubpixel( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, int psf_dimen, struct PointSpreadLookupTable *psf_lookup, struct MFtemplate *template_ptr, int nrows, int ncols );

void    GenerateMFsubtemplateLine( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, struct MFsubtemplate *subtemplate_ptr, int nrows, int ncols );

int     BilinearInterpWeights( int nrows, int ncols, double subpixelrow, double subpixelcol,
							   long *pixeloffset, double *w11, double *w12, double *w21, double *w22 );

void    FreeMFtemplate( struct MFtemplate *template_ptr );

void    FreeMFsubtemplate( struct MFsubtemplate *subtemplate_ptr );

void    IntegratedSignalFromDemeaned( struct MFtemplate *template_ptr,
	                                  double *demeaned_sig_ptr,
							          double *integrated_signal  );

void    MaximumSignalFromDemeaned( struct MFtemplate *template_ptr,
	                               double *demeaned_sig_ptr,
							       double *maximum_signal  );

void    BackgroundVarianceFromDemeaned( struct MFtemplate *template_ptr,
	                                    double *demeaned_sig_ptr,
							            double *background_variance  );

void    AverageSignalFromWhitened( struct MFtemplate *template_line_ptr,
	                          double *whitened_sig_ptr, double *covarinv_ptr,
					          double sigma_factor, double *average_signal, double *average_sigma );

void    PSFdimensionSignalSigma( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                             double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
					             int psf_minpdimen, double psf_phalfwidth,
	                             double Xrow,  double Xcol,
	                             double Vrow,  double Vcol,
					             int *psf_dimen, double *average_signal, double *average_sigma  );


void    FrameMatchedFilter( int kframe, double hypo_signal_amp, struct MFtemplate *template_PSF_ptr,
	                        double *whitened_sig_ptr, double *covarinv_ptr,
	                        double *sum_sRINVat, double *sum_atRINVat, double *sum_sRINVt, double *sum_tRINVt,
							double *TRT, double *MLEdb );

void    MultiFrameMatchedFilter( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                             double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
								 struct PointSpreadLookupTable *psf_lookup,
	                             double Xrow,  double Xcol,  double Vrow,  double Vcol,
							     double *TRT,  double *MLEdb );

void    MultiFrameMatchedFilterRefinement( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                                       double **whitened_ptr_ptr,  double *assoctime_ptr, double *covarinv_ptr,
	                                       int psfminsize, double psfhalfwidth,
                                           struct PSOparameters *PSOparams, struct MFhypothesis *MFmotion );


void    FrameMatchedFilterInterpolated( int kframe, double hypo_signal_amp, struct MFsubtemplate *subtemplate_ptr,
	                                    double *whitened_sig_ptr, double *covarinv_ptr,
	                                    double *sum_sRINVat, double *sum_atRINVat, double *sum_sRINVt, double *sum_tRINVt,
							            double *TRT, double *MLEdb );

void    MultiFrameMatchedFilterInterpolated( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                                         double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
	                                         double Xrow,  double Xcol,
	                                         double Vrow,  double Vcol,
							                 double *TRT,  double *MLEdb );

void    MultiFrameMatchedFilterRefinementInterpolated( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                                                   double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
                                                       struct PSOparameters *PSOparams, struct  MFhypothesis  *MFmotion );


void    MultiFrameMeasurements( struct trackerinfo *track,  struct EMCCDparameters *params,
	                            double **demeaned_ptr_ptr,  double *assoctime_ptr,  int *psf_dimen,
								struct plate_s *plate_solution,
								struct EMCCDtrackdata *trackmeasurements );

int     FindClosestRingBufferTime( double time_desired, double *time_ringbuffer, int maxringbuffer );

void    SubtractImage_BiasResult( unsigned short *image1_ptr, unsigned short *image2_ptr, unsigned short *image1minus2_ptr,
	                              unsigned short diffbias, int nrows, int ncols );

int     DuplicationTest( double Xrow1, double Xcol1, double Vrow1, double Vcol1, double Metric1,
	                     double Xrow2, double Xcol2, double Vrow2, double Vcol2, double Metric2,
						 double AngleLimitDegrees, double RelVelocityPercentage, double DistanceLimitPixels );


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%      Dark and Flat Fielding            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//  Two functions for dark fields are provided. One for a default all-zeros setup
//  using ZeroDarkField. The second to read a dark field previously collected using
//  ReadDarkField. The dark can be removed from an image using the function
//  SubtractImage_BiasResult with a bias of zero. The dark removal should be the
//  first processing step after a new image read.
//
//  The flat fielding functions are used to obtain a flat field multiplier estimate
//  using a sequence of random pointing or stare pointing imagery files.
//
//  To estimate a flat field, an image can be screened for bright pixels above a
//  cutoff probability [0,1] that are then all zeroed, and the image with the lower
//  level pixel values is returned using the function ImageOnlyBelowPcut.
//  Alternatively one can zap stars by sequentially finding the brightest remaining
//  pixel and zeroing regions around each one until a user specified bright object
//  removal count is reached. Both of these methods removes stars and hot pixels
//  and operates most effectively if the input image can be approximately flattened
//  before calling either of these functions.
//
//  The star free images are fed an accumulating sum on a per pixel basis to form
//  a mean flat field after all the images are processed. This is stored as a
//  scaled flat field MULTIPLIER using the WriteFlatFieldMultiplier. Read back in
//  using ReadFlatFieldMultiplier and apply to an image with FlatApplied2Image. An
//  alternative is to use the default unity multiplier of UnityFlatFieldMultiplier
//  to fill the entire flat field multiplier array with ones.
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int   ReadDarkField( unsigned short *darkfield_ptr, int *mrows, int *mcols, char *filename )
{
int    nrc;
FILE  *readfile;


    if( ( readfile = fopen( filename, "rb" ) ) == NULL )  {
		printf(" ERROR ===>  Cannot open output file %s in ReadDarkField\n", filename );
		return(15);
	}

	fread( mrows, sizeof(int), 1, readfile );
	fread( mcols, sizeof(int), 1, readfile );

    nrc = *mrows * *mcols;

	fread( darkfield_ptr, sizeof(unsigned short), nrc, readfile );

	fclose( readfile );

	return(0);

}


//**********************************************************************************

void  ZeroDarkField( unsigned short *darkfield_ptr, int nrows, int ncols )
{
int     krc, nrc;

    nrc = nrows * ncols;

	for( krc=0; krc<nrc; krc++ )  *darkfield_ptr++ = 0;

}


//**********************************************************************************

void    ImageOnlyBelowPcut( unsigned short *image_ptr, int nrows, int ncols, int Pcut )
{
int     krc, nrc, kgray, kcut, *histogram, *cumprob, totalcounts;


    //======== Allocate memory for the histogram of unsigned shorts --> 65535

    histogram = (int*) malloc( 65536 * sizeof(int) );
    cumprob   = (int*) malloc( 65536 * sizeof(int) );

	if( histogram == NULL  ||  cumprob == NULL )  {
		printf(" ERROR ===>  Memory not allocated in ImageBelowPcut\n" );
		Delay_msec(5000);
		exit(1);
	}


	//======== Fill the histogram

	nrc = nrows * ncols;

	for( kgray=0; kgray<65536; kgray++ )  histogram[kgray] = 0;

	for( krc=0; krc<nrc; krc++ )  {

		kgray = (int)image_ptr[krc];

		histogram[kgray] += 1;

	}


	//========= Compute the cumulative counts and cumulative percentage from gray level 1 to 65534
	//            to avoid zero valued and saturated pixels

	totalcounts = nrc - histogram[0] - histogram[65535];


	cumprob[0] = 0;

	for( kgray=1; kgray<65535; kgray++ )  cumprob[kgray] = cumprob[kgray-1] + histogram[kgray];

	cumprob[65535] = totalcounts;


	for( kgray=0; kgray<65536; kgray++ )  cumprob[kgray] = ( 100 * cumprob[kgray] ) / totalcounts;


	//========= Get the cut limit index

	kcut = 1;

	for( kgray=1; kgray<65535; kgray++ )  {

		if( Pcut > cumprob[kgray] )  kcut = kgray;
		else                         break;

	}


	//======== Overwrite the image with zeros for gray levels above the cut line

	for( krc=0; krc<nrc; krc++ )  if( image_ptr[krc] > kcut )  image_ptr[krc] = 0;


    //======== Free memory for this function

	free( histogram );
	free( cumprob );

}


//**********************************************************************************

void    ZapStarRegions( unsigned short *image_ptr, int nrows, int ncols, int halfsize, int nstars )
{
int     krc, jrc, nrc, kgray, kstar;
int     row, col, found_next_pixel;
int    *hbase;
int    *hlink;
//int    *hcount, kount;


    //======== Allocate memory for the linked list histogram

    //hcount = (int*) malloc(        65536L * sizeof(int) );
    hbase  = (int*) malloc(        65536L * sizeof(int) );
    hlink  = (int*) malloc( nrows * ncols * sizeof(int) );

	if( /*hcount == NULL  ||*/  hbase == NULL  ||  hlink == NULL )  {
		printf(" ERROR ===>  Memory not allocated in ZapStarRegions\n" );
		Delay_msec(5000);
		exit(1);
	}


	//======== Fill the linked list histogram (linked list of pixel position offsets)

	nrc = nrows * ncols;

	for( kgray=0; kgray<65536L; kgray++ )  hbase[kgray] = -1;  // end of linked list for this gray level


	for( krc=0; krc<nrc; krc++ )  {

		kgray = (int)image_ptr[krc];

		hlink[krc] = hbase[kgray];

		hbase[kgray] = krc;

	}


	//======== Get the histogram counts
	/*
	for( kgray=0; kgray<65536L; kgray++ )  {

		kount = 0;

		krc = hbase[kgray];

		while( krc >= 0 )  {

			kount++;

			krc = hlink[krc];

		}

		hcount[kgray] = kount;

	}
	*/


	//======== Loop over the brightest "nstars"

	kgray = 65535L;  // Start at the highest gray level

	krc = hbase[kgray];

	for( kstar=0; kstar<nstars; kstar++ )  {

		//------- Find the next pixel (star region) to zap

		found_next_pixel = 0;

		while( found_next_pixel == 0 )  {

			if( krc == -1 )  {  // Shift one lower gray level if no more pixels in this gray level
			    kgray--;
				if( kgray < 0 ) break;
			    krc = hbase[kgray];
			}

			else if( image_ptr[krc] == 0 )  {  // If pixel has been zapped, move to next pixel
				krc = hlink[krc];
			}

			else  {  // Pixel and its surrounding region needs to be zapped
				found_next_pixel = 1;
			}

		}

		if( kgray < 0 )  break;

		//------- Convert offset index krc to row and column. Don't walk off edge of image

		row = krc / ncols;
		col = krc % ncols;

		if( row < halfsize )  row = halfsize;
		if( col < halfsize )  col = halfsize;

		if( row > nrows - halfsize - 1 )  row = nrows - halfsize - 1;
		if( col > ncols - halfsize - 1 )  col = ncols - halfsize - 1;


		//------- Zap the region around this pixel (start in upper left corner)

		jrc = row * ncols + col - halfsize * ncols - halfsize;

		for( row=-halfsize; row<=+halfsize; row++ )  {

			for( col=-halfsize; col<=+halfsize; col++ )  {

				image_ptr[jrc] = 0;

				jrc++;

			}

			jrc += ncols - 2 * halfsize - 1;   // Next row, left edge of region

		}


	    //------- Go to next pixel in the linked list

		krc = hlink[krc];

	}


    //======== Free memory for this function

	//free( hcount );
	free(  hbase );
	free(  hlink );

}


//**********************************************************************************

void  FlatApplied2Image( unsigned short *image_ptr, double *flatfield_ptr, unsigned short *flattened_ptr, int nrows, int ncols )
{
int     krc, nrc;


    nrc = nrows * ncols;

	for( krc=0; krc<nrc; krc++ )  *flattened_ptr++ = (unsigned short)( (double)*image_ptr++ * *flatfield_ptr++ );

}


//**********************************************************************************

int   WriteFlatFieldMultiplier( double *flatfield_ptr, int nrows, int ncols, char *filename )
{
int             krc;
unsigned short *flatscaledby4k;
double          flat4k;
FILE           *writefile;


    //======== Allocate memory for scaled unsigned integer storage

	flatscaledby4k = (unsigned short*) malloc( nrows * ncols * sizeof(unsigned short) );

	if( flatscaledby4k == NULL )  {
		printf(" ERROR ===>  Memory not allocated in WriteFlatFieldMultiplier\n" );
		Delay_msec(5000);
		exit(1);
	}


	//======== Scale the flat field multiplier (around unity) by 4096
	//            which when truncated will cover = [ 1/16, 16 ]

	for( krc=0; krc<nrows*ncols; krc++ )  {

		flat4k = flatfield_ptr[krc] * 4096.0;

		if( flat4k > 65535.0 )  flat4k = 65535.0;
		if( flat4k <     0.0 )  flat4k =     0.0;

		flatscaledby4k[krc] = (unsigned short)flat4k;

	}


	//======== Open file, write scaled flat field mulitplier, close the file

    if( ( writefile = fopen( filename, "wb" ) ) == NULL )  {
		printf(" ERROR ===>  Cannot open output file %s in WriteFlatFieldMultiplier\n", filename );
		return(15);
	}

	fwrite( &nrows,         sizeof(int),                      1, writefile );
	fwrite( &ncols,         sizeof(int),                      1, writefile );
	fwrite( flatscaledby4k, sizeof(unsigned short), nrows*ncols, writefile );

	fclose( writefile );


	//======== free memeory

	free( flatscaledby4k );

	return(0);

}

//**********************************************************************************

int   ReadFlatFieldMultiplier( double *flatfield_ptr, int *mrows, int *mcols, char *filename )
{
int             krc, nrc;
unsigned short *flatscaledby4k;
FILE           *readfile;



	//======== Open the scaled flat field multiplier file

    if( ( readfile = fopen( filename, "rb" ) ) == NULL )  {
		printf(" ERROR ===>  Cannot open output file %s in ReadFlatFieldMultiplier\n", filename );
		return(15);
	}

	fread( mrows,         sizeof(int),    1,   readfile );
	fread( mcols,         sizeof(int),    1,   readfile );

    nrc = *mrows * *mcols;


	//======== Allocate memory for the scaled unsigned integer ingest

	flatscaledby4k = (unsigned short*) malloc( nrc * sizeof(unsigned short) );

	if( flatscaledby4k == NULL )  {
		printf(" ERROR ===>  Memory not allocated in ReadFlatFieldMultiplier\n" );
		Delay_msec(5000);
		exit(1);
	}


	//======== Read the flat and close the file

	fread( flatscaledby4k, sizeof(unsigned short), nrc, readfile );

	fclose( readfile );


	//======== Scale the flat field multiplier back down by 4096 and save as a double

	for( krc=0; krc<nrc; krc++ )  flatfield_ptr[krc] = ( (double)flatscaledby4k[krc] ) / 4096.0;


	return(0);

}


//**********************************************************************************

void  UnityFlatFieldMultiplier( double *flatfield_ptr, int nrows, int ncols )
{
int     krc, nrc;

    nrc = nrows * ncols;

	for( krc=0; krc<nrc; krc++ )  *flatfield_ptr++ = 1.0;

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%             Equalization               %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//  Call GlobalBackgndFromHistogram to obtain a global mean background level in the
//  range of user defined gray level probability percentages. Plow to Phigh are the
//  cumulative probability levels obtained using a integer histogram. The image data
//  is assumed to be 2 byte unsigned integers (unsigned short). The mean level in
//  the band of gray level count percentages is returned as a double. Plow and Phigh
//  are input as integer percentages. For example use 5% and 50% to avoid dead
//  pixels below the bottom 1/20 of pixel gray values and avoid stars in the top 1/2
//  of the gray levels respectively. The function does not consider zero valued
//  or saturated pixels in determining the mean.
//
//  Call EqualizeImageWithGlobalBackgnd to scale an image by a reference background
//  level divided by the image's background level (obtained on a previous call to
//  GlobalBackgndFromHistogram). The resultant scaled image is truncated to an
//  unsigned integer array, but uses doubles for the background levels input. The
//  output array pointer equal_ptr is assumed its memory is previously allocated.
//  The operation can occur "in-place" such that the output array can be the same
//  as the input array image_ptr.
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

double    GlobalBackgndFromHistogram( unsigned short *image_ptr, int nrows, int ncols, int Plow, int Phigh )
{
int     krc, nrc, kgray, klow, khigh, *histogram, *cumprob, totalcounts;
double  graysum, countsum, xmean;


    //======== Allocate memory for the histogram of unsigned shorts --> 65535

    histogram = (int*) malloc( 65536L * sizeof(int) );
    cumprob   = (int*) malloc( 65536L * sizeof(int) );

	if( histogram == NULL  ||  cumprob == NULL )  {
		printf(" ERROR ===>  Memory not allocated in BackgndFromHistogram\n" );
		Delay_msec(5000);
		exit(1);
	}


	//======== Fill the histogram

	nrc = nrows * ncols;

	for( kgray=0; kgray<65536L; kgray++ )  histogram[kgray] = 0;

	for( krc=0; krc<nrc; krc++ )  {

		kgray = (int)image_ptr[krc];

		histogram[kgray] += 1;

	}


	//========= Compute the cumulative counts and cumulative percentage from gray level 1 to 65534
	//            to avoid zero valued and saturated pixels

	totalcounts = nrc - histogram[0] - histogram[65535L];

	if( totalcounts == 0 )  totalcounts = nrc;


	cumprob[0] = 0;

	for( kgray=1; kgray<65535L; kgray++ )  cumprob[kgray] = cumprob[kgray-1] + histogram[kgray];

	cumprob[65535L] = totalcounts;


	for( kgray=0; kgray<65536L; kgray++ )  cumprob[kgray] = ( 100 * cumprob[kgray] ) / totalcounts;


	//========= Get the limit indices

	klow  =     1L;
	khigh = 65534L;

	for( kgray=1; kgray<65535L; kgray++ )  {

		if( Plow  >= cumprob[kgray] )  klow  = kgray;

		if( Phigh >  cumprob[kgray] )  khigh = kgray;
		else                           break;

	}


	//======== Compute the mean from the histogram

	graysum  = 0.0;
	countsum = 0.0;

    for( kgray=klow; kgray<=khigh; kgray++ )  {
		graysum  += (double)histogram[kgray] * (double)kgray;
		countsum += (double)histogram[kgray];
	}

	if( countsum > 0.0 )  xmean = graysum / countsum;
	else                  xmean = 0.0;


    //======== Free memory for this function

	free( histogram );
	free( cumprob );

	return( xmean );

}

//**********************************************************************************

void  EqualizeImageWithGlobalBackgnd( unsigned short *image_ptr, unsigned short *equal_ptr, double image_backgnd, double ref_backgnd, int nrows, int ncols )
{
int     krc, nrc, kgray;
double  scale;
unsigned short  *scaled_integer;


    //======== Allocate memory for the scaled gray levels

    scaled_integer = (unsigned short*) malloc( 65536L * sizeof(short) );

	if( scaled_integer == NULL )  {
		printf(" ERROR ===>  Memory not allocated in EqualizeImageWithGlobalBackgnd\n" );
		Delay_msec(10000);
		exit(1);
	}


	//======== Determine the resultant gray levels for all possible input gray levels and the global scaling

    scale = ref_backgnd / image_backgnd;

	for( kgray=0; kgray<65536L; kgray++ )  scaled_integer[kgray] = (unsigned short)( (double)kgray * scale );


	//======== Apply the global scaling to the entire image

    nrc = nrows * ncols;

	for( krc=0; krc<nrc; krc++ )  *equal_ptr++ = scaled_integer[ *image_ptr++ ];


	//======== Free memory

	free( scaled_integer );

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%               Decimation               %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//  Call DecimateInitialization at program start once you know the image size to
//  allocate memory for the decimated image array and store dimensions for the
//  decimated image. The full resolution input image does NOT need to have size
//  dimensions that are integer multiples of the decimation factor "ndecim". But
//  the right and bottom edges of pixels that fall outside a complete ndecim by
//  ndecim block will not be used.
//
//  Call DecimateNxNviaMeanN given a new image to decimate, which will return the
//  decimated image. Between the first initialization call and the actual
//  decimation calls, the dimensional sizes must not change for the input image
//  and the allocated decimated array (info stored in the decimator structure).
//  The decimation finds the ndecim highest pixel values in each ndecin by ndecim
//  block tile and averages those to replace that block of pixels with a
//  single value.
//
//  Remember to free the memory in the decimator structure using function
//  DecimateFreeMemory at program completion.
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


int    DecimateInitialization( int ndecim, int nrows_image, int ncols_image, struct decimator *decim )
{

    //======== Test for proper dimensions

    if( nrows_image < 1  ||  ncols_image < 1 )  {
		printf(" ERROR ===>  image %d x %d rows or columns size must be positive integers in DecimateInitialization\n", nrows_image, ncols_image );
		return(11);
	}

    if( ndecim < 1  ||  ndecim > nrows_image  ||  ndecim > ncols_image )  {
		printf(" ERROR ===>  ndecim = %d must be >0 and less than the image dimensions DecimateInitialization\n", ndecim );
		return(11);
	}

	decim->ndecim      = ndecim;

	decim->nrows_image = nrows_image;
	decim->ncols_image = ncols_image;

	decim->nrows_decim = (int)( nrows_image / ndecim );  //... truncated to nearest lower integer
	decim->ncols_decim = (int)( ncols_image / ndecim );


	//======== Allocate memory for the decimated array

	decim->image_ptr = NULL;

	decim->image_ptr = (unsigned short*) malloc( decim->nrows_decim * decim->ncols_decim * sizeof( unsigned short ) );

	if( decim->image_ptr == NULL )  {
		printf(" ERROR ===>  Memory not allocated for decim->array_ptr in DecimateInitialization\n" );
		return(12);
	}

	return(0);

}

//***********************************************************************************

int    DecimateFreeMemory( struct decimator *decim )
{
	free( decim->image_ptr );

	decim->image_ptr = NULL;

	return(0);
}

//***********************************************************************************

int    DecimateNxNviaMeanN( unsigned short *image_ptr, struct decimator *decim )
{
int              blockrow, blockcol, row, col, dmean, ksort, nsort;
unsigned short  *inp_ptr, *out_ptr, *highval, datum;


 	//======== Allocate memory for the high pixel value sorted list
    //            NOTE: Dimension is 1 higher to handle insertion sort index bounds

	highval = (unsigned short*) malloc( ( decim->ndecim + 1 ) * sizeof( unsigned short ) );

	if( highval == NULL )  {
		printf(" ERROR ===>  Memory not allocated for highval in DecimateNxNviaMeanN\n" );
		return(14);
	}


	//======== Loop over starting row and column of each block's upper left corner

	out_ptr = (unsigned short*)decim->image_ptr;

	for( blockrow=0; blockrow<decim->nrows_decim; blockrow++ )  {

		for( blockcol=0; blockcol<decim->ncols_decim; blockcol++ )  {

	        inp_ptr = image_ptr + ( blockrow * decim->ncols_image + blockcol ) * decim->ndecim;

			//-------- Loop over rows and columns within each ndecim-by-ndecim block of pixels

			nsort = 0;

			for( row=0; row<decim->ndecim; row++ )  {

				for( col=0; col<decim->ndecim; col++ )  {

					datum = *inp_ptr++;

					//........ Insertion sort of ndecim highest pixel values

				    ksort = nsort - 1;

					while( ksort >= 0  &&  datum > highval[ksort] )  {

					    highval[ksort+1] = highval[ksort];

						ksort--;

					}

					highval[ksort+1] = datum;

					if( nsort < decim->ndecim )  nsort++;

				} //... end of intra-block column loop

				inp_ptr += decim->ncols_image - decim->ndecim;

			} //... end of intra-block row loop


			//-------- Save the mean of the ndecim highest pixel values

			dmean = 0;

			for( ksort=0; ksort<decim->ndecim; ksort++ )  dmean += (int)highval[ksort];

			*out_ptr++ = (unsigned short)( dmean / decim->ndecim );

		} //... end of inter-block column loop

	} //... end of inter-block row loop


	//========== Free sorted list memory

	free( highval );

	return(0);

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%                Thresholding             %%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// These functions perform a thresholding operation given an input image, the mean
// and standard deviation images, and a user defined scaling factor "sfactor".
// An exceedance pixel is defined when:  image >= mean + sfactor * sigma
//
// The first function ThresholdPreCompute can be used to pre-compute the threshold
// array values of  mean + sfactor * sigma  if the three components (threshold)
// does not change and can be applied to multiple images.
//
// The second function ThresholdExceedances performs the actual thresholding
// operation and returns a vector of pointer offsets "mpoffset_ptr" to each
// exceedance pixel and the number of exceedance pixels "pixelcount_ptr".
// If this function uses the result from ThresholdPreCompute, pass the
// "threshold_ptr" output array of the previous function as the "mean_ptr"
// to ThresholdExceedances and also use a zero value for sfactor.
//
// E.g.  If there was NO ThresholdPreCompute call:
//
//   ThresholdExceedances( image, nrows, ncols, mean, sigma, sfactor, pixelcount, mpoffsets );
//
//
// E.g.  If there was a ThresholdPreCompute call:
//
//   ThresholdPreCompute( nrows, ncols, mean, sigma, sfactor, threshold );
//
//   ThresholdExceedances( image, nrows, ncols, threshold, NULL, 0.0, pixelcount, mpoffsets );
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void    ThresholdPreCompute( int nrows_image, int ncols_image,
	                         unsigned short *mean_ptr, unsigned short *sigma_ptr, double sfactor,
							 unsigned short *threshold_ptr )
{
int  krc, nrc;


	//======== Compute the threshold for each pixel based on mean, sigma, and standard deviation factor

	nrc = nrows_image * ncols_image;

	for( krc=0; krc<nrc; krc++ )  *threshold_ptr++ = *mean_ptr++  +  (unsigned short)( sfactor * (double)*sigma_ptr++ );


}

//***********************************************************************************

void    ThresholdExceedances( unsigned short *image_ptr, int nrows_image, int ncols_image,
	                          unsigned short *mean_ptr, unsigned short *sigma_ptr, double sfactor,
	                          long *pixelcount_ptr, long *mpoffset_ptr )
{
int             krc, nrc, kount;
unsigned short  ksigma;


    kount = 0;

	nrc = nrows_image * ncols_image;


	//======== Compute the threshold for each pixel based on mean, sigma and the standard deviation factor
	//            while accumulating the exceedance pixel positions as pointer offsets into the image array.

	if( sfactor > 0.0 )  {

	    for( krc=0; krc<nrc; krc++ )  {

		    ksigma = (unsigned short)( sfactor * (double)*sigma_ptr++ );

		    if( *image_ptr++ >= *mean_ptr++ + ksigma )  {

			    mpoffset_ptr[kount] = krc;

			    kount++;

		    }

	    }

	}


	//======== Else use the mean image ONLY for the threshold when sfactor is zero.
	//             E.g. a pre-computed threshold may be passed in via the mean_ptr.

	else  {   //... sfactor == 0.0

	    for( krc=0; krc<nrc; krc++ )  {

		    if( *image_ptr++ >= *mean_ptr++ )  {

			    mpoffset_ptr[kount] = krc;

			    kount++;

		    }

	    }

	}


	//======== Return the number of threshold exceedances

	*pixelcount_ptr = kount;


}

//***********************************************************************************


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%            Matched Filtering           %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Series of functions to perform matched filter processing given a starting pixel
// position and velocity vector. Resultant products is a refined state information
// and signal dependent and independent metrics.
//
// This includes a function GaussianPSF for generating a two-dimensional Gaussian
// point spread function (PSF) given its pixel size and half-width/half-max value.
// The PSF is normalized to a cefficient sum of unity.
//
// The function ConvolvePSF forms the two-dimensional spatial convolution of an
// image and a user defined PSF (e.g. GaussianPSF).
//
// The function DemeanWhiten removes a mean from an image and divides by the
// variance estimate on a per pixel basis. The function DemeanWhitenConvolve does
// the same first steps but also convolves the resultant image with a user
// defined PSF. The function Demean just removes the mean.
//
// The function CovarianceInverse assumes the covariance matrix posseses the
// property that noise is pixel-to-pixel independent and thus diagonal elements
// only make up each pixel's variance estimate. The covariance inverse array
// is just a diagonal array (vector) of 1/variance per pixel.
//
// The function GenerateMFtemplate creates a thin line of pixel positions (pointer
// offsets into the working array dimensions) that represent the streak generated
// in a single frame given the relative time offset for the frame since the start,
// the reference starting pixel position, the velocity vector, and time per frame.
// The streak length is given by the velocity vector and time per frame. The output
// is sampled at the integer pixel discretization level.
//
// The function GenerateMFtemplatePSF convolves a point spread function around a
// thin line streak to produce a blurred/extended streak using a pixel level
// discretized (integer) stepping. The function GenerateMFtemplatePSFsubpixel
// does the same, but samples the thin line streak at floating point positions
// along the streak at 1/10 pixel resolution to build up the blurred template.
// In both cases tha template is normalized to a max value of unity and
// generated at an output sampling equivalent to integer pixel resolution.
//
// The GenerateMFsubtemplateLine generates a normalized thin line streak point
// by point template (without PSF blurring) at a 1/10 pixel resolution spacing
// using the subpixel start to end positions as endpoints. Normalization is
// such that the integrated template over one pixel length is unity.
//
// The function AverageSignal computes the average of the demeaned signal along
// a thin line hypothsis template for use as the hypothesized signal amplitude
// in the signal dependent matched filter.
//
// The function FrameMatchedFilter computes an accumulating sum of matched filter
// metrics on a per frame basis that include both the signal dependent TRT and the
// signal independent MLEdB maximum likelihood estimates. Calling this function
// with kframe equal to zero, resets the sums and metrics to zero.
//
//    Signal ind = 5 log10 { Frame_Sum & Pixel_Sum [ (SignalFrame-<S>) R^-1 ]^2
//             / Frame_Sum & Pixel_Sum of Variance [ ( NoiseFrame-<S>) R^-1 ]
//
//    Signal dep = Frame_Sum [ (S-<S>) R^-1 a T ] / Frame_Sum [ a T R^-1 a T ]
//
//         R = diagonal( sigma^2 )
//         a = average of signal in the k-th frame along a thin line template
//         T = blurred motion template (thin line of 1's convolved with a PSF)
//
// The function MultiframeMatchedFilter loops through the sequence of frames to
// generate templates and compute the contribution to the matched filter sums
// and accumulating metrics.
//
// The function MultiFrameMatchedFilterRefinement calls MutliframeMatchedFilter
// many times to find the metric minimization for the best fitting starting
// position and velocity vector. This is a simple parameter tweaking algorithm
// that should get replaced by a simplex or particle stream minimization routine.
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void    GaussianPSF( int pdimen, double phalfwidth, struct PointSpreadFunc *psf )
{
int     prow, pcol, halfdim, kp;
double  rsq, psum;


    if( pdimen % 2 == 0 )  {
		printf("ERROR ===> PSF must have odd # for dimension in call to GaussianPSF\n");
		Delay_msec(10000);
		exit(1);
	}

	psf->pvalue = (double*) malloc( pdimen * pdimen * sizeof( double ) );

	if( psf->pvalue == NULL )  {
		printf("ERROR ===> PSF memory not allocated in GaussianPSF\n");
		Delay_msec(10000);
		exit(1);
	}


    psf->pdimen     = pdimen;
	psf->phalfwidth = phalfwidth;

	halfdim = ( pdimen - 1 ) / 2;

	kp   = 0;
	psum = 0.0;

	for( prow=-halfdim; prow<=+halfdim; prow++ )  {

		for( pcol=-halfdim; pcol<=+halfdim; pcol++ )  {

			 rsq = (double)( prow * prow + pcol * pcol );

			 psf->pvalue[kp] = exp( -0.69314718 * rsq / (phalfwidth * phalfwidth) );

			 psum += psf->pvalue[kp];

			 kp++;

		}

	}

	//-------- Normalize to a unity integral

	for( kp=0; kp<pdimen*pdimen; kp++ )  psf->pvalue[kp] /= psum;

	//for( kp=0; kp<pdimen*pdimen; kp++ )  printf("psf %lf\n", psf->pvalue[kp] );

}


//***********************************************************************************

void    FreeMemoryPSF( struct PointSpreadFunc *psf )
{

	free( psf->pvalue );

}

//***********************************************************************************

void    GaussianPSFtable( int minpdimen, double phalfwidth, struct PointSpreadLookupTable *psf_lookup )
{
int     krsq, maxrsize;
double  rsq, term;


    //======== Set the minimum PSF size

    if( minpdimen % 2 == 0 )  {
		printf("ERROR ===> PSF must have odd # for minpdimen in call to GaussianPSFtable\n");
		Delay_msec(10000);
		exit(1);
	}

	psf_lookup->minpdimen = minpdimen;


	//======== Malloc and fill psf lookup table

	psf_lookup->phalfwidth = phalfwidth;

	maxrsize = 30;  //... hardcoded 30 pixel max radius of PSF

	psf_lookup->maxkrsq = 100 * maxrsize * maxrsize;

	psf_lookup->ptable = (double*) malloc( psf_lookup->maxkrsq * sizeof(double) );

	term = -0.69314718 / ( phalfwidth * phalfwidth );

	for( krsq=0; krsq<psf_lookup->maxkrsq; krsq++ )  {

		rsq = (double)krsq / 100.0;    //... hash index = 100 * radius squared

		psf_lookup->ptable[krsq] = exp( term * rsq );

	}


}


//***********************************************************************************

void    FreeMemoryPSFtable( struct PointSpreadLookupTable *psf_lookup )
{

	free( psf_lookup->ptable );

}


//***********************************************************************************

void    ConvolvePSF( double *dimage_ptr, struct PointSpreadFunc *psf, double *convolved_ptr, int nrows, int ncols )
{
int     krc_in, krc_out, krow, kcol, prc, prow, pcol, halfdim;
double  csum;



	//======== Convolve the input double image (likely whitened) with the PSF (does not do the edges < halfdim)

    memcpy( convolved_ptr, dimage_ptr, nrows * ncols * sizeof(double) );


	halfdim = ( psf->pdimen - 1 ) / 2;

	for( krow=halfdim; krow<nrows-halfdim; krow++ )  {

		krc_out = krow * ncols + halfdim;

		for( kcol=halfdim; kcol<ncols-halfdim; kcol++ )  {

	        krc_in = ( krow - halfdim ) * ncols  +  ( kcol - halfdim );

			csum = 0.0;

			prc = 0;

			for( prow=0; prow<psf->pdimen; prow++ )  {

				for( pcol=0; pcol<psf->pdimen; pcol++ )  csum += psf->pvalue[prc++] * dimage_ptr[krc_in++];

				krc_in += ncols - psf->pdimen;

			}

			convolved_ptr[krc_out++] = csum;


		} //... end of output col loop

	} //.. end of output row loop


}


//***********************************************************************************

void    Demean( unsigned short *image_ptr, unsigned short *mean_ptr, double *demeaned_ptr, int nrows, int ncols )
{
int      krc, nrc;


    nrc = nrows * ncols;


	//======== Remove the provided mean from the image
	//            Note if the image_ptr is NULL, then the calling function has passed the image_ptr
	//                 via the demeaned_ptr and thus the latter gets overwritten.

	if( image_ptr == NULL )  {
	    for( krc=0; krc<nrc; krc++ )  demeaned_ptr[krc] =      demeaned_ptr[krc] - (double)mean_ptr[krc];
    }
	else  {
	    for( krc=0; krc<nrc; krc++ )  demeaned_ptr[krc] = (double)image_ptr[krc] - (double)mean_ptr[krc];
	}


}


//***********************************************************************************

void    DemeanWhiten( unsigned short *image_ptr, unsigned short *mean_ptr, unsigned short *sigma_ptr, double *whitened_ptr, int nrows, int ncols )
{
int      krc, nrc;


    nrc = nrows * ncols;


	//======== Remove the mean and whiten (addition of 0.0001 to variance to avoid divide by zero)
	//            Note if the image_ptr is NULL, then the calling function has passed the image_ptr
	//                 via the whitened_ptr and thus the latter gets overwritten.

	if( image_ptr == NULL )  {
	    for( krc=0; krc<nrc; krc++ )  whitened_ptr[krc] = (      whitened_ptr[krc] - (double)mean_ptr[krc] ) / ( 0.0001 + (double)sigma_ptr[krc] * (double)sigma_ptr[krc] );
    }
	else  {
	    for( krc=0; krc<nrc; krc++ )  whitened_ptr[krc] = ( (double)image_ptr[krc] - (double)mean_ptr[krc] ) / ( 0.0001 + (double)sigma_ptr[krc] * (double)sigma_ptr[krc] );
	}


}


//***********************************************************************************

void    DemeanWhitenConvolve( unsigned short *image_ptr, unsigned short *mean_ptr, unsigned short *sigma_ptr, double *convolved_ptr, struct PointSpreadFunc *psf, double *workarray_ptr, int nrows, int ncols )
{
int      krc, nrc;


    nrc = nrows * ncols;


	//======== Remove the mean and whiten (addition of 0.0001 to variance to avoid divide by zero)
	//            Note if the image_ptr is NULL, then the calling function has passed the image_ptr
	//                 via the convolved_ptr and thus the latter gets overwritten.

	if( image_ptr == NULL )  {
	    for( krc=0; krc<nrc; krc++ )  convolved_ptr[krc] = (     convolved_ptr[krc] - (double)mean_ptr[krc] ) / ( 0.0001 + (double)sigma_ptr[krc] * (double)sigma_ptr[krc] );
    }
	else  {
	    for( krc=0; krc<nrc; krc++ )  convolved_ptr[krc] = ( (double)image_ptr[krc] - (double)mean_ptr[krc] ) / ( 0.0001 + (double)sigma_ptr[krc] * (double)sigma_ptr[krc] );
	}


	//======== PSF convolution such that the outer edges are filled with whitened data that will NOT get PSF convolved

	memcpy( workarray_ptr, convolved_ptr, nrc * sizeof(double) );

	ConvolvePSF( workarray_ptr, psf, convolved_ptr, nrows, ncols );


}


//***********************************************************************************

void    CovarianceInverse( unsigned short *sigma_ptr, double *covarinv_ptr, int nrows, int ncols )
{
int     krc, nrc;


    nrc = nrows * ncols;


	//======== Fill the covariance inverse array with 1 / sigma^2  (assumes spatially independent noise)

	for( krc=0; krc<nrc; krc++ )  covarinv_ptr[krc] = 1.0 / ( 0.0001 + (double)sigma_ptr[krc] * (double)sigma_ptr[krc] );

}


//***********************************************************************************

void    GenerateMFtemplate( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, struct MFtemplate *template_ptr, int nrows, int ncols )
{
int     krow, kcol, nhalflength, row, col, kt, nt, ntmax;
double  rowcenter, colcenter;


    //======== Determine the template orientation and half size dimension

    if( fabs(Vrow) > fabs(Vcol) )  nhalflength = (int)( 0.5 + fabs(Vrow * TimePerFrame) / 2.0 );
	else                           nhalflength = (int)( 0.5 + fabs(Vcol * TimePerFrame) / 2.0 );


    //======== Allocate space for a new template of imagery pixels referenced via pointer offsets

	ntmax = 2 * nhalflength + 1;

	template_ptr->pixeloffset =   (long*) malloc( ntmax * sizeof(   long ) );
    template_ptr->pixelsignal = (double*) malloc( ntmax * sizeof( double ) );

	if( template_ptr->pixeloffset == NULL ||  template_ptr->pixelsignal == NULL )  {
		printf("ERROR ===> Template memory not allocated in GenerateMFtemplate\n");
		Delay_msec(10000);
		exit(1);
	}

	for( kt=0; kt<ntmax; kt++ )  template_ptr->pixelsignal[kt] = 0.0;


	//======== Loop through integer spaced row or column based on starting position and speed

	rowcenter = Xrow + Time * Vrow;
    colcenter = Xcol + Time * Vcol;

	//-------- No smearing motion so single pixel

	if( Vrow == 0.0  &&  Vcol == 0.0 )  {

		nt = 0;

		row = (int)( 0.5 + rowcenter );
		col = (int)( 0.5 + colcenter );

	    if( row >= 0  &&  row <= nrows-1  &&  col >= 0  &&  col <= ncols-1 )  {

            template_ptr->pixeloffset[nt] = (long)( row * ncols + col );
			template_ptr->pixelsignal[nt]  = 1.0;

		    nt = 1;

		}

	}


	//-------- Step evenly in rows for generally vertical motion

	else if( fabs(Vrow) > fabs(Vcol) )  {

		nt = 0;

		for( krow=-nhalflength; krow<=+nhalflength; krow++ )  {

			row = (int)( 0.5 + rowcenter + (double)krow );
			col = (int)( 0.5 + colcenter + (double)krow * Vcol / fabs(Vrow) );

			if( row < 0  ||  row > nrows-1 )  continue;
			if( col < 0  ||  col > ncols-1 )  continue;

            template_ptr->pixeloffset[nt] = (long)( row * ncols + col );
			template_ptr->pixelsignal[nt]  = 1.0;

			nt++;

			if( nt >= ntmax )  break;

		}

	}


	//-------- Step evenly in columns for generally horizontal or diagonal motion

	else  {       //... step evenly in columns

		nt = 0;

		for( kcol=-nhalflength; kcol<=+nhalflength; kcol++ )  {

			row = (int)( 0.5 + rowcenter + (double)kcol * Vrow / fabs(Vcol) );
			col = (int)( 0.5 + colcenter + (double)kcol );

			if( row < 0  ||  row > nrows-1 )  continue;
			if( col < 0  ||  col > ncols-1 )  continue;

            template_ptr->pixeloffset[nt] = (long)( row * ncols + col );
			template_ptr->pixelsignal[nt]  = 1.0;

			nt++;

			if( nt >= ntmax )  break;

		}

	}


	//======== Save the actual number of template pixels within the image boundaries

    template_ptr->npixels = nt;

}


//***********************************************************************************

void    GenerateMFtemplatePSF( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, struct PointSpreadFunc *psf, struct MFtemplate *template_ptr, int nrows, int ncols )
{
int     krow, kcol, nhalflength, nhalfwidth, row, col, kt, ktfill, nt, ntmax, kpsf, prow, pcol;
long    pixeloffset;
double  rowcenter, colcenter, maxsignal;


    //======== Given the template orientation, determine this line segment's half length dimension

    if( fabs(Vrow) > fabs(Vcol) )  nhalflength = (int)( 0.5 + fabs(Vrow * TimePerFrame) / 2.0 );
	else                           nhalflength = (int)( 0.5 + fabs(Vcol * TimePerFrame) / 2.0 );


    //======== Use the PSF size as the template half width dimension

    nhalfwidth = (int)( (psf->pdimen - 1) / 2 );


    //======== Allocate space for a new template of imagery pixels referenced via pointer offsets
	//         Its size is the thin line segment length + PSF halfwidth extended away from the
	//         line segment, but has been made oversized just in case of double to int rounding.

	ntmax = ( 2 * nhalflength + 1 ) * ( 2 * nhalfwidth + 1 ) * ( 2 * nhalfwidth + 1 );

	template_ptr->pixeloffset =   (long*) malloc( ntmax * sizeof(   long ) );
    template_ptr->pixelsignal = (double*) malloc( ntmax * sizeof( double ) );

	if( template_ptr->pixeloffset == NULL  ||  template_ptr->pixelsignal == NULL )  {
		printf("ERROR ===> Template memory not allocated in GenerateMFtemplatePSF\n");
		Delay_msec(10000);
		exit(1);
	}

	for( kt=0; kt<ntmax; kt++ )  template_ptr->pixelsignal[kt] = 0.0;


	//======== Loop through integer spaced row or column based on starting position and speed

	rowcenter = Xrow + Time * Vrow;
    colcenter = Xcol + Time * Vcol;


	//-------- No smearing motion so single pixel convolved with the PSF

	if( Vrow == 0.0  &&  Vcol == 0.0 )  {

		nt   = 0;
		kpsf = 0;

		for( prow=-nhalfwidth; prow<=+nhalfwidth; prow++ )  {
		for( pcol=-nhalfwidth; pcol<=+nhalfwidth; pcol++ )  {

			row = (int)( 0.5 + rowcenter + (double)prow );
		    col = (int)( 0.5 + colcenter + (double)pcol );

	        if( row >= 0  &&  row <= nrows-1  &&  col >= 0  &&  col <= ncols-1 )  {

				pixeloffset = (long)( row * ncols + col );

                template_ptr->pixeloffset[nt] = pixeloffset;
                template_ptr->pixelsignal[nt] = psf->pvalue[kpsf];

		        nt++;

		    }

			kpsf++;

		}  //... end psf col loop
		}  //... end PSF row loop

	}


	//-------- Step evenly in rows for generally vertical motion convolving with PSF

	else if( fabs(Vrow) > fabs(Vcol) )  {

		nt   = 0;
		kpsf = 0;

		for( prow=-nhalfwidth; prow<=+nhalfwidth; prow++ )  {  //... PSF double loop
		for( pcol=-nhalfwidth; pcol<=+nhalfwidth; pcol++ )  {

		    for( krow=-nhalflength; krow<=+nhalflength; krow++ )  {  //... line segment loop

				row = (int)( 0.5 + rowcenter + (double)prow + (double)krow );
			    col = (int)( 0.5 + colcenter + (double)pcol + (double)krow * Vcol / fabs(Vrow) );

			    if( row < 0  ||  row > nrows-1 )  continue;
			    if( col < 0  ||  col > ncols-1 )  continue;

				pixeloffset = (long)( row * ncols + col );

				ktfill = nt;

				for( kt=0; kt<nt; kt++ )  {  //... Check if we have already assigned this pixel

					if( pixeloffset == template_ptr->pixeloffset[kt] )  {
						ktfill = kt;
						break;
					}

				}

				if( ktfill == nt )  {  //... template pixel not previously assigned, add it to the template
                    template_ptr->pixeloffset[nt]  = pixeloffset;
                    nt++;
				}

				template_ptr->pixelsignal[ktfill] += psf->pvalue[kpsf];  // add PSF value to the template pixel

			}  //... end of loop along line segment

			kpsf++;

		}  //... end psf col loop
		}  //... end PSF row loop

	}


	//-------- Step evenly in columns for generally horizontal or diagonal motion

	else  {       //... step evenly in columns

		nt   = 0;
		kpsf = 0;

		for( prow=-nhalfwidth; prow<=+nhalfwidth; prow++ )  {  //... PSF double loop
		for( pcol=-nhalfwidth; pcol<=+nhalfwidth; pcol++ )  {

		    for( kcol=-nhalflength; kcol<=+nhalflength; kcol++ )  {  //... line segment loop

			    row = (int)( 0.5 + rowcenter + (double)prow + (double)kcol * Vrow / fabs(Vcol) );
			    col = (int)( 0.5 + colcenter + (double)pcol + (double)kcol );

			    if( row < 0  ||  row > nrows-1 )  continue;
			    if( col < 0  ||  col > ncols-1 )  continue;

				pixeloffset = (long)( row * ncols + col );

				ktfill = nt;

				for( kt=0; kt<nt; kt++ )  {  //... Check if we have already assigned this pixel

					if( pixeloffset == template_ptr->pixeloffset[kt] )  {
						ktfill = kt;
						break;
					}

				}

				if( ktfill == nt )  {  //... template pixel not previously assigned, add it to the template
                    template_ptr->pixeloffset[nt]  = pixeloffset;
                    nt++;
				}

				template_ptr->pixelsignal[ktfill] += psf->pvalue[kpsf];  // add PSF value to the template pixel

			}  //... end of loop along line segment

			kpsf++;

		}  //... end psf col loop
		}  //... end PSF row loop

	}


	//======== Save the actual number of template pixels within the image boundaries

    template_ptr->npixels = nt;


	//======== Normalize the signal template to unity at the max

	maxsignal = template_ptr->pixelsignal[0];

	for( kt=0; kt<nt; kt++ )  {

		if( template_ptr->pixelsignal[kt] > maxsignal )  maxsignal = template_ptr->pixelsignal[kt];

	}

	for( kt=0; kt<nt; kt++ )  template_ptr->pixelsignal[kt] /= maxsignal;


}


//***********************************************************************************

void    GenerateMFtemplatePSFsubpixel( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, int psf_dimen, struct PointSpreadLookupTable *psf_lookup, struct MFtemplate *template_ptr, int nrows, int ncols )
{
int     nhalflength, nhalfwidth, row, col, kt, ktfill, nt, ntmax, prow, pcol, krsq;
int     nsubpixels, nlength, kstep;
long    pixeloffset;
double  rowtrailingedge, coltrailingedge, frow, fcol, flength, deltarow, deltacol, rsq, maxsignal;


    //======== Given the template orientation, determine this line segment's maximum half length dimension

    if( fabs(Vrow) > fabs(Vcol) )  nhalflength = (int)( 0.5 + fabs(Vrow * TimePerFrame) / 2.0 );
	else                           nhalflength = (int)( 0.5 + fabs(Vcol * TimePerFrame) / 2.0 );


	//======== Use the PSF size as the template half width dimension

    nhalfwidth = (int)( (psf_dimen - 1) / 2 );


    //======== Allocate space for a new template of imagery pixels referenced via pointer offsets
	//         Its size is the thin line segment length + PSF halfwidth extended away from the
	//         line segment, but has been made oversized just in case of double to int rounding.

	ntmax = ( 2 * nhalflength + 1 ) * ( 2 * nhalfwidth + 1 ) * ( 2 * nhalfwidth + 1 );

	template_ptr->pixeloffset =   (long*) malloc( ntmax * sizeof(   long ) );
    template_ptr->pixelsignal = (double*) malloc( ntmax * sizeof( double ) );

	if( template_ptr->pixeloffset == NULL  ||  template_ptr->pixelsignal == NULL )  {
		printf("ERROR ===> Template memory not allocated in GenerateMFtemplatePSFsubpixel\n");
		Delay_msec(10000);
		exit(1);
	}

	for( kt=0; kt<ntmax; kt++ )  template_ptr->pixelsignal[kt] = 0.0;



	//======== Compute the subpixel resolution to build the template over

	nsubpixels = 10;  //... 1/10 pixel stepping

	flength = sqrt( Vrow * Vrow + Vcol * Vcol ) * TimePerFrame;  //... pixels per frame

	nlength = (int)( flength * (double)nsubpixels ) + 2;

	deltarow = Vrow * TimePerFrame / (double)( nlength - 1 );  //... pixels per step
	deltacol = Vcol * TimePerFrame / (double)( nlength - 1 );


	//======== Loop through sub-pixel spaced rows and columns based on trailing edge starting position
	//            with assumption that Xrow and Xcol are line segment centroids for Time = 0.
	//            Trailing edge = Centroid at time zero
	//                            Plus the movement forward due to speed at time "Time"
	//                            Minus the backstep by the speed for half the frame time

	rowtrailingedge = Xrow + Time * Vrow - 0.5 * TimePerFrame * Vrow;
    coltrailingedge = Xcol + Time * Vcol - 0.5 * TimePerFrame * Vcol;

	nt = 0;

	for( kstep=0; kstep<nlength; kstep++ )  {  //... line segment loop

		frow = rowtrailingedge + (double)kstep * deltarow;
		fcol = coltrailingedge + (double)kstep * deltacol;

		for( prow=-nhalfwidth; prow<=+nhalfwidth; prow++ )  {  //... PSF double loop
		for( pcol=-nhalfwidth; pcol<=+nhalfwidth; pcol++ )  {

			row = (int)( 0.5 + frow + (double)prow );  //... nearest integer pixel position
			col = (int)( 0.5 + fcol + (double)pcol );

			if( row < 0  ||  row > nrows-1 )  continue;
			if( col < 0  ||  col > ncols-1 )  continue;

			pixeloffset = (long)( row * ncols + col );

			ktfill = nt;

			for( kt=0; kt<nt; kt++ )  {  //... Check if we have already assigned this pixel

				if( pixeloffset == template_ptr->pixeloffset[kt] )  {
					ktfill = kt;
					break;
				}

			}

			if( ktfill == nt )  {  //... template pixel not previously assigned, add it to the template
                template_ptr->pixeloffset[nt]  = pixeloffset;
                nt++;
			}


			//-------- Add the Guassian PSF value to the template pixel at the distance of the pixel center
			//              from the current subpixel segment position.

			rsq = ( frow - (double)row ) * ( frow - (double)row )
				+ ( fcol - (double)col ) * ( fcol - (double)col );

			krsq = (int)( 100.0 * rsq );

			if( krsq < psf_lookup->maxkrsq )  template_ptr->pixelsignal[ktfill] += psf_lookup->ptable[krsq];

			if( nt >= ntmax )  break;

		}  //... end psf col loop

		if( nt >= ntmax )  break;

		}  //... end PSF row loop


		if( nt >= ntmax )  break;

	}  //... end kstep loop marching along the line segment


	//======== Save the actual number of template pixels within the image boundaries

    template_ptr->npixels = nt;


	//======== Normalize the signal template to unity at the max

	maxsignal = template_ptr->pixelsignal[0];

	for( kt=0; kt<nt; kt++ )  {

		if( template_ptr->pixelsignal[kt] > maxsignal )  maxsignal = template_ptr->pixelsignal[kt];

	}

	for( kt=0; kt<nt; kt++ )  template_ptr->pixelsignal[kt] /= maxsignal;


}

//***********************************************************************************
void    GenerateMFsubtemplateLine( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, struct MFsubtemplate *subtemplate_ptr, int nrows, int ncols )
{
int     nsubpixels, nlength, kstep, nt, withinFOV;
long    pixeloffset;
double  w11, w12, w21, w22;
double  rowtrailingedge, coltrailingedge, flength, deltarow, deltacol, subpixelrow, subpixelcol;


	//======== Compute the subpixel resolution to build the template over

	nsubpixels = 10;   //... 1/10 pixel stepping

	flength = sqrt( Vrow * Vrow + Vcol * Vcol ) * TimePerFrame;  //... pixels per frame

	nlength = (int)( flength * (double)nsubpixels ) + 1;


    //======== Allocate space for a new template of subpixel positions

	subtemplate_ptr->pixeloffset = (  long*) malloc( nlength * sizeof(   long ) );
    subtemplate_ptr->w11         = (double*) malloc( nlength * sizeof( double ) );
    subtemplate_ptr->w12         = (double*) malloc( nlength * sizeof( double ) );
    subtemplate_ptr->w21         = (double*) malloc( nlength * sizeof( double ) );
    subtemplate_ptr->w22         = (double*) malloc( nlength * sizeof( double ) );

	if( subtemplate_ptr->pixeloffset == NULL  ||  subtemplate_ptr->w11 == NULL  ||  subtemplate_ptr->w12 == NULL
		                                      ||  subtemplate_ptr->w21 == NULL  ||  subtemplate_ptr->w22 == NULL  )  {
		printf("ERROR ===> Subtemplate memory not allocated in GenerateMFsubtemplateLine\n");
		Delay_msec(10000);
		exit(1);
	}


	deltarow = Vrow * TimePerFrame / (double)( nlength - 1 );  //... pixels per step
	deltacol = Vcol * TimePerFrame / (double)( nlength - 1 );


	//======== Loop through sub-pixel spaced rows and columns based on trailing edge starting position
	//            with assumption that Xrow and Xcol are line segment centroids for Time = 0.
	//            Trailing edge = Centroid at time zero
	//                            Plus the movement forward due to speed at time "Time"
	//                            Minus the backstep by the speed for half the frame time

	rowtrailingedge = Xrow + Time * Vrow - 0.5 * TimePerFrame * Vrow;
    coltrailingedge = Xcol + Time * Vcol - 0.5 * TimePerFrame * Vcol;

	nt = 0;

	for( kstep=0; kstep<nlength; kstep++ )  {  //... line segment loop

		subpixelrow = rowtrailingedge + (double)kstep * deltarow;
		subpixelcol = coltrailingedge + (double)kstep * deltacol;

		withinFOV = BilinearInterpWeights( nrows,
			                               ncols,
			                               subpixelrow,
							               subpixelcol,
							               &pixeloffset,
							               &w11, &w12, &w21, &w22 );

		if (withinFOV == INSIDEFOV) {  //... version 203 - defines template only within the FOV
			                           //                  do not extrapolate beyond edges

			subtemplate_ptr->pixeloffset[nt] = pixeloffset,
			subtemplate_ptr->w11[nt]         = w11;
			subtemplate_ptr->w12[nt]         = w12;
			subtemplate_ptr->w21[nt]         = w21;
			subtemplate_ptr->w22[nt]         = w22;
			nt++;

		}



	}  //... end kstep loop marching along the line segment


	//======== Normalize the signal template to unity when integrated over 1 pixel distance

    subtemplate_ptr->nsubpixels = nt;

	subtemplate_ptr->offset12 = 1;
	subtemplate_ptr->offset21 = ncols;
	subtemplate_ptr->offset22 = ncols + 1;

	subtemplate_ptr->subpixelval = 1.0;


}

//***********************************************************************************

int    BilinearInterpWeights( int      nrows,
			                  int      ncols,
			                  double   subpixelrow,
							  double   subpixelcol,
							  long    *pixeloffset,
							  double  *w11,
							  double  *w12,
							  double  *w21,
							  double  *w22    )
{
long    krc, introw, intcol;
double  drow, dcol;


     //... version 203 - defines template only within the FOV
     //                  do not extrapolate beyond edges

     //======== Get the row pixel offset to the upper left corner of the 2x2 data block

     introw = (long)subpixelrow;

     if( subpixelrow < 0.0 )  {                     //... linear extrapolation from below row 0
		 return(OUTSIDEFOV);
		 //krc  = 0;
		 //drow = subpixelrow;
	 }
	 else if( subpixelrow > (double)(nrows-2) )  {  //... linear extrapolation from above nrows-2
		 return(OUTSIDEFOV);
		 //krc  = (long)(nrows-2) * (long)ncols;
		 //drow = subpixelrow - (double)(nrows-2);
	 }
	 else  {                                        //... bilinear interpolation
		 krc  = introw * (long)ncols;
		 drow = subpixelrow - (double)introw;
	 }


     //======== Add the col pixel offset to the upper left corner of the 2x2 data block

     intcol = (long)subpixelcol;

	 if( subpixelcol < 0.0 )  {                     //... linear extrapolation from left of col 0
		 return(OUTSIDEFOV);
		 //krc += 0;
		 //dcol = subpixelcol;
	 }
	 else if( subpixelcol > (double)(ncols-2) )  {  //... linear extrapolation from right of ncols-1
		 return(OUTSIDEFOV);
		 //krc += (long)( ncols-2 );
		 //dcol = subpixelcol - (double)(ncols-2);
	 }
	 else  {                                        //... bilinear interpolation
		 krc += intcol;
		 dcol = subpixelcol - (double)intcol;
	 }


	 //======== Now get the bilinear interpolation weights

	 *pixeloffset = krc;

	 *w11 = ( 1.0 - drow ) * ( 1.0 - dcol );
	 *w12 = ( 1.0 - drow ) * (       dcol );
	 *w21 = (       drow ) * ( 1.0 - dcol );
	 *w22 = (       drow ) * (       dcol );

	 return(INSIDEFOV);


}


//***********************************************************************************

void    GenerateMFtemplateAOI( double Time, double TimePerFrame, double Xrow, double Xcol, double Vrow, double Vcol, int aoi_dimension, struct MFtemplate *template_ptr, int nrows, int ncols )
{
int     nhalflength, nhalfwidth, row, col, kt, ktfill, nt, ntmax, prow, pcol;
int     nsubpixels, nlength, kstep;
long    pixeloffset;
double  rowtrailingedge, coltrailingedge, frow, fcol, flength, deltarow, deltacol;


    //======== Given the template orientation, determine this line segment's maximum half length dimension

    if( fabs(Vrow) > fabs(Vcol) )  nhalflength = (int)( 0.5 + fabs(Vrow * TimePerFrame) / 2.0 );
	else                           nhalflength = (int)( 0.5 + fabs(Vcol * TimePerFrame) / 2.0 );


	//======== Use the PSF size as the template half width dimension

    nhalfwidth = (int)( (aoi_dimension - 1) / 2 );


    //======== Allocate space for a new template of imagery pixels referenced via pointer offsets
	//         Its size is the thin line segment length + PSF halfwidth extended away from the
	//         line segment, but has been made oversized just in case of double to int rounding.

	ntmax = ( 2 * nhalflength + 1 ) * ( 2 * nhalfwidth + 1 ) * ( 2 * nhalfwidth + 1 );

	template_ptr->pixeloffset =   (long*) malloc( ntmax * sizeof(   long ) );
    template_ptr->pixelsignal = (double*) malloc( ntmax * sizeof( double ) );

	if( template_ptr->pixeloffset == NULL  ||  template_ptr->pixelsignal == NULL )  {
		printf("ERROR ===> Template memory not allocated in GenerateMFtemplateAOI\n");
		Delay_msec(10000);
		exit(1);
	}

	for( kt=0; kt<ntmax; kt++ )  template_ptr->pixelsignal[kt] = 1.0;



	//======== Compute the subpixel resolution to build the template over

	nsubpixels = 10;  //... 1/10 pixel stepping

	flength = sqrt( Vrow * Vrow + Vcol * Vcol ) * TimePerFrame;  //... pixels per frame

	nlength = (int)( flength * (double)nsubpixels ) + 2;

	deltarow = Vrow * TimePerFrame / (double)( nlength - 1 );  //... pixels per step
	deltacol = Vcol * TimePerFrame / (double)( nlength - 1 );


	//======== Loop through sub-pixel spaced rows and columns based on trailing edge starting position
	//            with assumption that Xrow and Xcol are line segment centroids for Time = 0.
	//            Trailing edge = Centroid at time zero
	//                            Plus the movement forward due to speed at time "Time"
	//                            Minus the backstep by the speed for half the frame time

	rowtrailingedge = Xrow + Time * Vrow - 0.5 * TimePerFrame * Vrow;
    coltrailingedge = Xcol + Time * Vcol - 0.5 * TimePerFrame * Vcol;

	nt = 0;

	for( kstep=0; kstep<nlength; kstep++ )  {  //... line segment loop

		frow = rowtrailingedge + (double)kstep * deltarow;
		fcol = coltrailingedge + (double)kstep * deltacol;

		for( prow=-nhalfwidth; prow<=+nhalfwidth; prow++ )  {  //... AOI double loop
		for( pcol=-nhalfwidth; pcol<=+nhalfwidth; pcol++ )  {

			row = (int)( 0.5 + frow + (double)prow );  //... nearest integer pixel position
			col = (int)( 0.5 + fcol + (double)pcol );

			if( row < 0  ||  row > nrows-1 )  continue;
			if( col < 0  ||  col > ncols-1 )  continue;

			pixeloffset = (long)( row * ncols + col );

			ktfill = nt;

			for( kt=0; kt<nt; kt++ )  {  //... Check if we have already assigned this pixel

				if( pixeloffset == template_ptr->pixeloffset[kt] )  {
					ktfill = kt;
					break;
				}

			}

			if( ktfill == nt )  {  //... template pixel not previously assigned, add it to the template
                template_ptr->pixeloffset[nt]  = pixeloffset;
                nt++;
			}


			if( nt >= ntmax )  break;

		}  //... end aoi col loop


		if( nt >= ntmax )  break;

		}  //... end aoi row loop


		if( nt >= ntmax )  break;

	}  //... end kstep loop marching along the line segment


	//======== Save the actual number of template pixels within the image boundaries

    template_ptr->npixels = nt;

}


//***********************************************************************************

void    FreeMFtemplate( struct MFtemplate *template_ptr )
{

	free( template_ptr->pixeloffset );

	free( template_ptr->pixelsignal );

	template_ptr->pixeloffset = NULL;

	template_ptr->pixelsignal = NULL;

}

//***********************************************************************************

void    FreeMFsubtemplate( struct MFsubtemplate *subtemplate_ptr )
{

	free( subtemplate_ptr->pixeloffset );

	free( subtemplate_ptr->w11 );
	free( subtemplate_ptr->w12 );
	free( subtemplate_ptr->w21 );
	free( subtemplate_ptr->w22 );

	subtemplate_ptr->pixeloffset = NULL;

	subtemplate_ptr->w11 = NULL;
	subtemplate_ptr->w12 = NULL;
	subtemplate_ptr->w21 = NULL;
	subtemplate_ptr->w22 = NULL;

}


//***********************************************************************************

void    IntegratedSignalFromDemeaned( struct MFtemplate *template_ptr,
	                                  double *demeaned_sig_ptr,
							          double *integrated_signal  )

{
int     kt, krc;
double  sum_signal;


	//======== Sum within the "template" to get an integrated demeaned signal

	sum_signal = 0.0;

	for( kt=0; kt<template_ptr->npixels; kt++ )  {

		krc = template_ptr->pixeloffset[kt];  //... integrate over the template's pixels

		sum_signal += demeaned_sig_ptr[krc];  //... (s-smean)  demeaned pixel sum

	}

	*integrated_signal = sum_signal;


}

//***********************************************************************************

void    MaximumSignalFromDemeaned( struct MFtemplate *template_ptr,
	                               double *demeaned_sig_ptr,
							       double *maximum_signal  )

{
int     kt, krc;
double  max_signal;


	//======== Search within the "template" to get a maximum demeaned signal

	max_signal = -1.0e+30;

	for( kt=0; kt<template_ptr->npixels; kt++ )  {

		krc = template_ptr->pixeloffset[kt];  //... search over the template's pixels

		if( demeaned_sig_ptr[krc] > max_signal )  max_signal = demeaned_sig_ptr[krc];

	}

	*maximum_signal = max_signal;


}

//***********************************************************************************

void    BackgroundVarianceFromDemeaned( struct MFtemplate *template_ptr,
	                                    double *demeaned_sig_ptr,
							            double *background_variance  )

{
int     kt, krc;
double  sum, sumsq, en;


	//======== Sum within the "template" to get an integrated demeaned signal

	sum   = 0.0;
	sumsq = 0.0;

	for( kt=0; kt<template_ptr->npixels; kt++ )  {

		krc = template_ptr->pixeloffset[kt];  //... integrate over the template's pixels

		sum   += demeaned_sig_ptr[krc];  //... (s-smean)  demeaned pixel sum

		sumsq += demeaned_sig_ptr[krc] * demeaned_sig_ptr[krc];

	}

	en = (double)template_ptr->npixels + 1.0e-30;


	*background_variance = ( sumsq - sum * sum / en ) / ( en - 1.0 );


}

//***********************************************************************************

void    AverageSignalFromWhitened( struct MFtemplate *template_line_ptr,
	                               double *whitened_sig_ptr, double *covarinv_ptr,
					               double sigma_factor, double *average_signal, double *average_sigma )

{
int     kt, krc;
double  sum_signal, sum_sigma;


	//======== Sum along the "thin line signal template" to get an average demeaned signal
	//              for the amplitude hypothesis (for given frame) and an average standard
    //              deviation signal for the minimum signal limit.

	sum_signal = 0.0;

	sum_sigma  = 0.0;

	for( kt=0; kt<template_line_ptr->npixels; kt++ )  {

		krc = template_line_ptr->pixeloffset[kt];  //... integrate over the thin line template pixels

		sum_signal += whitened_sig_ptr[krc] / covarinv_ptr[krc];  //... (s-smean)  demeaned pixel sum

		sum_sigma  += sqrt( 1.0 / covarinv_ptr[krc] );

	}

	*average_signal = sum_signal / ( (double)template_line_ptr->npixels + 1.0e-30 );


	//======== Do not use an average signal less than a user factor times the average standard deviation

	*average_sigma  = sum_sigma  / ( (double)template_line_ptr->npixels + 1.0e-30 );

	if( *average_signal < sigma_factor * *average_sigma )  *average_signal = sigma_factor * *average_sigma;

}


//======================================================================================================
//     Compute the PSF coverage region, average signal and average sigma for all frames based on signal
//     and noise estimations from the whitened imagery. The return vectors must be pre-allocated of size
//     at least nframes.
//======================================================================================================

void    PSFdimensionSignalSigma( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                             double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
					             int psf_minpdimen, double psf_phalfwidth,
	                             double Xrow,  double Xcol,
	                             double Vrow,  double Vcol,
					             int *psf_dimen, double *average_signal, double *average_sigma  )
{
int     sigframe, psf_dimension;
double  Time;
struct  MFtemplate       MFstreakTHIN;


    for( sigframe=0; sigframe<nframes; sigframe++ )  {

		//-------- Time relative to first point in track (we assume 1st point starts at Xrow, Xcol)

		Time = assoctime_ptr[sigframe] - assoctime_ptr[0];


		//-------- Generate a thin line template along track to get average signal (for this frame) that
		//            is no smaller than 1.5 times the average standard deviation

        GenerateMFtemplate( Time, time_per_frame, Xrow, Xcol, Vrow, Vcol, &MFstreakTHIN, nrows, ncols );

		AverageSignalFromWhitened( &MFstreakTHIN, &whitened_ptr_ptr[sigframe][0], covarinv_ptr, 1.5, &average_signal[sigframe], &average_sigma[sigframe] );

	    FreeMFtemplate( &MFstreakTHIN );


		//-------- Compute a PSF size based on the average signal and noise strength and PSF halfwidth

		if( average_signal[sigframe] > 0.0 )  psf_dimension = 1 + 2 * (long)( 1.0 + sqrt( psf_phalfwidth * psf_phalfwidth * log( average_sigma[sigframe] / average_signal[sigframe] ) / -0.69314718 ) );
		else                                  psf_dimension = 1;

		if( psf_dimension < psf_minpdimen )  psf_dimension = psf_minpdimen;

		psf_dimen[sigframe] = psf_dimension;  //... return the PSF size used for later signal integration


	}  //... end of frame loop

}

//***********************************************************************************

void    FrameMatchedFilter( int kframe, double hypo_signal_amp, struct MFtemplate *template_PSF_ptr,
	                        double *whitened_sig_ptr, double *covarinv_ptr,
	                        double *sum_sRINVat, double *sum_atRINVat, double *sum_sRINVt, double *sum_tRINVt,
							double *TRT, double *MLEdb )

{
int     kt, krc;
double  T, aT;


    //======== First call for kframe = 0, clears the running sums over frames

    if( kframe == 0 )  {
	    *sum_sRINVat  = 0.0;
		*sum_atRINVat = 0.0;
	    *sum_sRINVt   = 0.0;
		*sum_tRINVt   = 0.0;
	}


	//======== Sum the components making up the MLE and TRT using the "PSF convolved signal template"

	for( kt=0; kt<template_PSF_ptr->npixels; kt++ )  {

		krc = template_PSF_ptr->pixeloffset[kt];  //... Integrate over the PSF distributed template pixels


		T = template_PSF_ptr->pixelsignal[kt];    //... PSF template value

		aT = hypo_signal_amp * T;                 //... PSF template scaled by the hypothesized signal
		                                          //    amplitude "amp" for this frame


		*sum_sRINVat += whitened_sig_ptr[krc] * aT;    //...    (s-smean)  Rinv  amp template

		*sum_atRINVat += aT * covarinv_ptr[krc] * aT;  //... amp template  Rinv  amp template


		*sum_sRINVt += whitened_sig_ptr[krc] * T;      //... (s-smean)  Rinv  template

		*sum_tRINVt += T * covarinv_ptr[krc] * T;      //...  template  Rinv  template

	}


	//======== Compute the running (intermediate frame) TRT and MLE

	*TRT = *sum_sRINVat / ( *sum_atRINVat + 1.0e-30 );

	if (*sum_tRINVt < 0.0) {
		*MLEdb = -99.0;
	} else {
		*MLEdb = 5.0 * log10( 1.584893e-20 + 0.5 * ( *sum_sRINVt * *sum_sRINVt ) / ( *sum_tRINVt + 1.0e-30 ) );   //  -99 dB minimum
	}

/*
	if (isnan(*MLEdb)) {
		*MLEdb = -99.0;
	}
*/

	////printf("TRT sRat atRat sRt tRt  %lf  %lf  %lf % lf  %lf\n", *TRT, *sum_sRINVat, *sum_atRINVat, *sum_sRINVt, *sum_tRINVt,  );

}


//***********************************************************************************

void    FrameMatchedFilterInterpolated( int kframe, double hypo_signal_amp, struct MFsubtemplate *subtemplate_ptr,
	                                    double *whitened_sig_ptr, double *covarinv_ptr,
	                                    double *sum_sRINVat, double *sum_atRINVat, double *sum_sRINVt, double *sum_tRINVt,
							            double *TRT, double *MLEdb )

{
int     kt, krc11, krc12, krc21, krc22;
double  T, aT, whitened_signal, covariance_inverse;


    //======== First call for kframe = 0, clears the running sums over frames

    if( kframe == 0 )  {
	    *sum_sRINVat  = 0.0;
		*sum_atRINVat = 0.0;
	    *sum_sRINVt   = 0.0;
		*sum_tRINVt   = 0.0;
	}


	//======== Sum the components making up the MLE and TRT using "bilinear interpolated" values

    T = subtemplate_ptr->subpixelval;         //... line segment template value

	aT = hypo_signal_amp * T;                 //... template scaled by the hypothesized signal
		                                      //    amplitude "amp" for this frame

	for( kt=0; kt<subtemplate_ptr->nsubpixels; kt++ )  {

		//------ Get the corner pixel positions for bilinear interpolation (offsets from 0,0)

		krc11 = subtemplate_ptr->pixeloffset[kt];    //... upper left
		krc12 = subtemplate_ptr->offset12 + krc11;   //... upper right
		krc21 = subtemplate_ptr->offset21 + krc11;   //... lower left
		krc22 = subtemplate_ptr->offset22 + krc11;   //... lower right


		//.... Bilinear interpolation of the whitened*convolved and inverse covariance
		//        at the subpixel position of the subtemplate

		whitened_signal = whitened_sig_ptr[krc11] * subtemplate_ptr->w11[kt]
		                + whitened_sig_ptr[krc12] * subtemplate_ptr->w12[kt]
						+ whitened_sig_ptr[krc21] * subtemplate_ptr->w21[kt]
						+ whitened_sig_ptr[krc22] * subtemplate_ptr->w22[kt];

		covariance_inverse  = covarinv_ptr[krc11] * subtemplate_ptr->w11[kt]
		                    + covarinv_ptr[krc12] * subtemplate_ptr->w12[kt]
						    + covarinv_ptr[krc21] * subtemplate_ptr->w21[kt]
						    + covarinv_ptr[krc22] * subtemplate_ptr->w22[kt];



		*sum_sRINVat  +=         whitened_signal * aT;  //...    (s-smean)  Rinv  amp template

		*sum_atRINVat += aT * covariance_inverse * aT;  //... amp template  Rinv  amp template


		*sum_sRINVt   +=        whitened_signal * T;    //... (s-smean)  Rinv  template

		*sum_tRINVt   += T * covariance_inverse * T;    //...  template  Rinv  template

	}


	//======== Compute the running (intermediate frame) TRT and MLE

	*TRT = *sum_sRINVat / ( *sum_atRINVat + 1.0e-30 );

	if (*sum_tRINVt < 0.0) {
		*MLEdb = -99.0;
	} else {
		*MLEdb = 5.0 * log10( 1.584893e-20 + 0.5 * ( *sum_sRINVt * *sum_sRINVt ) / ( *sum_tRINVt + 1.0e-30 ) );   //  -99 dB minimum
	}

/*
	if (isnan(*MLEdb)) {
		*MLEdb = -99.0;
	}
*/

	////printf("TRT sRat atRat sRt tRt  %lf  %lf  %lf % lf  %lf\n", *TRT, *sum_sRINVat, *sum_atRINVat, *sum_sRINVt, *sum_tRINVt,  );

}


//======================================================================================================
//     Loop over frames to compute the matched filter contribution of each frame. This is a running
//        cumulative sum on the frame loop. Note that kframe = 0 resets the matched filter sums to
//        zero in the function FrameMatchedFilter. The noise component for the MLE estimate uses 2
//        frames in advance (modulo wrap) to compute the pure noise component.
//======================================================================================================

void    MultiFrameMatchedFilter( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                             double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
								 struct PointSpreadLookupTable *psf_lookup,
	                             double Xrow,  double Xcol,
	                             double Vrow,  double Vcol,
							     double *TRT,  double *MLEdb )
{
int     sigframe, noiframe, *psf_dimension;
double  Time, *average_signal, *average_sigma;
double  sum_sRINVat, sum_atRINVat, sum_sRINVt, sum_tRINVt;
struct  MFtemplate       MFstreakPSF;


	//-------- Compute a PSF size per frame based on the average signal and noise strength and PSF halfwidth

    psf_dimension  =    (int*) malloc( nframes * sizeof(    int ) );
    average_signal = (double*) malloc( nframes * sizeof( double ) );
    average_sigma  = (double*) malloc( nframes * sizeof( double ) );


    PSFdimensionSignalSigma( nframes, nrows, ncols, time_per_frame,
	                         whitened_ptr_ptr, assoctime_ptr, covarinv_ptr,
				             psf_lookup->minpdimen, psf_lookup->phalfwidth,
	                         Xrow, Xcol, Vrow, Vcol,
				             psf_dimension, average_signal, average_sigma );


	//-------- Loop to compute the contribution per frame to add to the matched filter metrics

    for( sigframe=0; sigframe<nframes; sigframe++ )  {

		//-------- Want 2 frames prior for "signal-free" pixels (wraps) to estimate noise variance

		noiframe = ( sigframe + nframes - 2 ) % nframes;   // NOT USED ANYMORE


		//-------- Time relative to first point in track (we assume 1st point starts at Xrow, Xcol)

		Time = assoctime_ptr[sigframe] - assoctime_ptr[0];


		//-------- Compute the contribution to the multi-frame TRT and MLE sums (for this frame)

		GenerateMFtemplatePSFsubpixel( Time, time_per_frame, Xrow, Xcol, Vrow, Vcol, psf_dimension[sigframe], psf_lookup, &MFstreakPSF, nrows, ncols );

		FrameMatchedFilter( sigframe, average_signal[sigframe], &MFstreakPSF,
			                &whitened_ptr_ptr[sigframe][0], covarinv_ptr,
	                        &sum_sRINVat, &sum_atRINVat, &sum_sRINVt, &sum_tRINVt,
							TRT, MLEdb );

	    FreeMFtemplate( &MFstreakPSF );

	}


	free( psf_dimension  );
	free( average_signal );
	free( average_sigma  );

}

//======================================================================================================
//     Loop over frames to compute the matched filter contribution of each frame. This is a running
//        cumulative sum on the frame loop. Note that kframe = 0 resets the matched filter sums to
//        zero in the function FrameMatchedFilter. The noise component for the MLE estimate uses 2
//        frames in advance (modulo wrap) to compute the pure noise component. NOTE: The whitened
//        data array is assumed already convolved with the PSF.
//======================================================================================================

void    MultiFrameMatchedFilterInterpolated( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                                         double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
	                                         double Xrow,  double Xcol,
	                                         double Vrow,  double Vcol,
							                 double *TRT,  double *MLEdb )
{
int     sigframe, noiframe;
double  Time, average_signal, average_sigma;
double  sum_sRINVat, sum_atRINVat, sum_sRINVt, sum_tRINVt;
struct  MFtemplate       MFstreakTHIN;
struct  MFsubtemplate    MFsubpixelLINE;


    for( sigframe=0; sigframe<nframes; sigframe++ )  {

		//-------- Want 2 frames prior for "signal-free" pixels (wraps) to estimate noise variance

		noiframe = ( sigframe + nframes - 2 ) % nframes;   // NOT USED ANYMORE


		//-------- Time relative to first point in track (we assume 1st point starts at Xrow, Xcol)

		Time = assoctime_ptr[sigframe] - assoctime_ptr[0];


		//-------- Generate a thin line template along track to get average signal (for this frame) that
		//            is no smaller than 1.5 times the average standard deviation

        GenerateMFtemplate( Time, time_per_frame, Xrow, Xcol, Vrow, Vcol, &MFstreakTHIN, nrows, ncols );

		AverageSignalFromWhitened( &MFstreakTHIN, &whitened_ptr_ptr[sigframe][0], covarinv_ptr, 1.5, &average_signal, &average_sigma );

	    FreeMFtemplate( &MFstreakTHIN );


		//-------- Compute the contribution to the multi-frame TRT and MLE sums (for this frame)

		GenerateMFsubtemplateLine( Time, time_per_frame, Xrow, Xcol, Vrow, Vcol, &MFsubpixelLINE, nrows, ncols );

		FrameMatchedFilterInterpolated( sigframe, average_signal, &MFsubpixelLINE,
			                            &whitened_ptr_ptr[sigframe][0], covarinv_ptr,
	                                    &sum_sRINVat, &sum_atRINVat, &sum_sRINVt, &sum_tRINVt,
							            TRT, MLEdb );

	    FreeMFsubtemplate( &MFsubpixelLINE );

	}

}


//***********************************************************************************

//======================================================================================================
//     PSO based iterative refinement the MF motion model parameters Xrow, Xcol, Vrow, Vcol
//======================================================================================================

void    MultiFrameMatchedFilterRefinement( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                                       double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
										   int psfminsize, double psfhalfwidth,
                                           struct PSOparameters *PSOparams, struct  MFhypothesis  *MFmotion )
{
double  TRTtest, MLEtest, XVguess[4], XVshift[4], Vspeed;

struct  particleswarming  pso;

struct  PointSpreadLookupTable   psf_lookup;


   //======== Generate a quick lookup table for the Gaussian PSF versus radius

 	GaussianPSFtable( psfminsize, psfhalfwidth, &psf_lookup );


   //======== Pre-populate the PSO

	printf("PSO2:  ");

    //ParticleSwarm_PrePopulate( 25, 4, 1000, BOUNDARY_REFLECTIVE, LIMITS_ARE_LOOSE, PARTICLEDISTRO_GAUSS, 1.0e-10, 0.8, 1.0, 2.0, &pso );

    ParticleSwarm_PrePopulate( PSOparams->nparticles, 4, PSOparams->maxiter,
		                       PSOparams->boundary_flag, PSOparams->limits_flag, PSOparams->distro_flag,
							   PSOparams->eps_convergence,
							   PSOparams->winertia_init, PSOparams->wstubborness, PSOparams->wgrouppressure,
							   &pso );



    //======== Initialize the four parameters and set up the particle swarm

	XVguess[0] = MFmotion->Xrow;
	XVguess[1] = MFmotion->Xcol;
	XVguess[2] = MFmotion->Vrow;
	XVguess[3] = MFmotion->Vcol;

	Vspeed = sqrt( MFmotion->Vrow * MFmotion->Vrow  +  MFmotion->Vcol * MFmotion->Vcol );

	XVshift[0] = 16.0;
	XVshift[1] = 16.0;
	XVshift[2] = 0.15 * Vspeed;
	XVshift[3] = 0.15 * Vspeed;

	ParticleSwarm_Initialize( XVguess, XVshift, &pso );

//FILE *hout;

//hout= fopen( "particles.txt", "wt" );


	//======== Loop through particles and iterations until PSO convergence met

	while( pso.processing == CONTINUE_PROCESSING )  {

        MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						         whitened_ptr_ptr, assoctime_ptr, covarinv_ptr,
								 &psf_lookup,
			                     pso.xtest[0], pso.xtest[1], pso.xtest[2], pso.xtest[3],
							     &TRTtest, &MLEtest );

//for( int kpart=0; kpart<pso.nparticles; kpart++ ) fprintf( hout, "%lf  %lf  %lf  %lf\n", pso.xcurr[kpart][0], pso.xcurr[kpart][1], pso.xcurr[kpart][2], pso.xcurr[kpart][3] );



		ParticleSwarm_Update( -MLEtest, &pso );  // use negative MLEdB for minimization
		                                         // MLE does better than TRT

	} //... end of while loop for particle swarm until convergence


    //======== Compute the TRT and MLE for the "best" solution

    MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						     whitened_ptr_ptr, assoctime_ptr, covarinv_ptr,
							 &psf_lookup,
			                 pso.gbest[0], pso.gbest[1], pso.gbest[2], pso.gbest[3],
							 &MFmotion->TRT, &MFmotion->MLEdb );

	MFmotion->Xrow = pso.gbest[0];
	MFmotion->Xcol = pso.gbest[1];
	MFmotion->Vrow = pso.gbest[2];
	MFmotion->Vcol = pso.gbest[3];


	ParticleSwarm_PostCleanup( &pso );


 	FreeMemoryPSFtable( &psf_lookup );

//fclose(hout);

//exit(1);

    printf("TRT=%lf  MLE=%lf  #iter=%d\n", MFmotion->TRT, MFmotion->MLEdb, pso.niteration );

}

//======================================================================================================
//     PSO based iterative refinement the MF motion model parameters Xrow, Xcol, Vrow, Vcol
//======================================================================================================

void    MultiFrameMatchedFilterRefinementInterpolated( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                                                   double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
                                                       struct PSOparameters *PSOparams, struct  MFhypothesis  *MFmotion )
{
double  TRTtest, MLEtest, XVguess[4], XVshift[4], Vspeed;

struct  particleswarming  pso;



	printf("PSO1:  ");


    //======== Pass 1 of the PSO =======================================================================================================

    //ParticleSwarm_PrePopulate( 500, 4, 1000, BOUNDARY_REFLECTIVE, LIMITS_ARE_STRICT, PARTICLEDISTRO_RANDOM, 1.0e-10, 0.8, 1.0, 2.0, &pso );

    ParticleSwarm_PrePopulate( PSOparams->nparticles, 4, PSOparams->maxiter,
		                       PSOparams->boundary_flag, PSOparams->limits_flag, PSOparams->distro_flag,
							   PSOparams->eps_convergence,
							   PSOparams->winertia_init, PSOparams->wstubborness, PSOparams->wgrouppressure,
							   &pso );

    //======== Initialize the four parameters and set up the particle swarm

	XVguess[0] = MFmotion->Xrow;
	XVguess[1] = MFmotion->Xcol;
	XVguess[2] = MFmotion->Vrow;
	XVguess[3] = MFmotion->Vcol;

	Vspeed = sqrt( MFmotion->Vrow * MFmotion->Vrow  +  MFmotion->Vcol * MFmotion->Vcol );

	XVshift[0] = 32.0;
	XVshift[1] = 32.0;
	XVshift[2] = 0.3 * Vspeed;
	XVshift[3] = 0.3 * Vspeed;

	ParticleSwarm_Initialize( XVguess, XVshift, &pso );


	//======== Loop through particles and iterations until PSO convergence met

	while( pso.processing == CONTINUE_PROCESSING )  {

		//printf("xtest  %d  %lf  %lf  %lf  %lf\n", pso.niteration, pso.xtest[0], pso.xtest[1], pso.xtest[2], pso.xtest[3] );

        MultiFrameMatchedFilterInterpolated( nframes, nrows, ncols, time_per_frame,
						                     whitened_ptr_ptr, assoctime_ptr, covarinv_ptr,
			                                 pso.xtest[0], pso.xtest[1], pso.xtest[2], pso.xtest[3],
							                 &TRTtest, &MLEtest );



//for( int kpart=0; kpart<pso.nparticles; kpart++ ) fprintf( hout, "%lf  %lf  %lf  %lf\n", pso.xcurr[kpart][0], pso.xcurr[kpart][1], pso.xcurr[kpart][2], pso.xcurr[kpart][3] );



		ParticleSwarm_Update( -MLEtest, &pso );  // use negative MLEdB for minimization
		                                         // MLE does better than TRT

	} //... end of while loop for particle swarm until convergence


    //======== Compute the TRT and MLE for the "best" solution

    MultiFrameMatchedFilterInterpolated( nframes, nrows, ncols, time_per_frame,
						                 whitened_ptr_ptr, assoctime_ptr, covarinv_ptr,
			                             pso.gbest[0], pso.gbest[1], pso.gbest[2], pso.gbest[3],
							             &MFmotion->TRT, &MFmotion->MLEdb );

	MFmotion->Xrow = pso.gbest[0];
	MFmotion->Xcol = pso.gbest[1];
	MFmotion->Vrow = pso.gbest[2];
	MFmotion->Vcol = pso.gbest[3];


	ParticleSwarm_PostCleanup( &pso );


    printf("TRT=%lf  MLE=%lf  #iter=%d\n", MFmotion->TRT, MFmotion->MLEdb, pso.niteration );

}


//***********************************************************************************
/*
//======================================================================================================
//     Crude iterative refinement the MF motion model parameters Xrow, Xcol, Vrow, Vcol
//======================================================================================================

void    MultiFrameMatchedFilterRefinement( int nframes,  int nrows,  int ncols,  double time_per_frame,
	                                       double **whitened_ptr_ptr,  double *assoctime_ptr,  double *covarinv_ptr,
										   struct PointSpreadFunc *psf, struct  MFhypothesis  *MFmotion )
{
int     kstep, kiter, xvloop;
double  TRTtest[3], MLEtest[3], XV[4], Xstep, Vstep;

#define  POSROW  0  //... Four parameters to vary (xvloop indices for vector XV)
#define  POSCOL  1
#define  VELROW  2
#define  VELCOL  3


    //======== Initialize the 4 parameters and starting step size in pixels

	XV[POSROW] = MFmotion->Xrow;
	XV[POSCOL] = MFmotion->Xcol;
	XV[VELROW] = MFmotion->Vrow;
	XV[VELCOL] = MFmotion->Vcol;

	Xstep =  32.0;  //... pixels
	Vstep = 512.0;  //... pixels/second


	//======== Run through decreasing step size adjustments

	for( kstep=0; kstep<8; kstep++ )  {

         Xstep /= 2.0;
         Vstep /= 2.0;

		 //-------- Perform multiple iterations at each step size

	     for( kiter=0; kiter<7; kiter++ )  {


			 //........ Modify each of the 2 positional parameters, one at a time

		     for( xvloop=0; xvloop<2; xvloop++ )  {

				                               //... Nominal starting values

                  MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						                   whitened_ptr_ptr, assoctime_ptr, covarinv_ptr, psf,
			                               XV[POSROW], XV[POSCOL], XV[VELROW], XV[VELCOL],
							               &TRTtest[0], &MLEtest[0] );

			      XV[xvloop] += Xstep;        //... Nominal + step

                  MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						                   whitened_ptr_ptr, assoctime_ptr, covarinv_ptr, psf,
			                               XV[POSROW], XV[POSCOL], XV[VELROW], XV[VELCOL],
							               &TRTtest[1], &MLEtest[1] );

			      XV[xvloop] -= 2.0 * Xstep;  //... Nominal - step

                  MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						                   whitened_ptr_ptr, assoctime_ptr, covarinv_ptr, psf,
			                               XV[POSROW], XV[POSCOL], XV[VELROW], XV[VELCOL],
							               &TRTtest[2], &MLEtest[2] );

			      XV[xvloop] += Xstep;        //... Return to nominal values before adjust


				  //........ Test for highest TRT and adjust parameter

			      ////if( TRTtest[1] > TRTtest[0] )  {

				  ////    if( TRTtest[1] > TRTtest[2] )    XV[xvloop] += Xstep;
				  ////    else                             XV[xvloop] -= Xstep;

			      ////}
			      ////else  if( TRTtest[2] > TRTtest[0] )  XV[xvloop] -= Xstep;


				  //........ Test for highest TRT and adjust parameter

			      if( MLEtest[1] > MLEtest[0] )  {

				      if( MLEtest[1] > MLEtest[2] )    XV[xvloop] += Xstep;
				      else                             XV[xvloop] -= Xstep;

			      }
			      else  if( MLEtest[2] > MLEtest[0] )  XV[xvloop] -= Xstep;


		     } //... end of 2 positional parameters loop


			 //........ Modify each of the 2 velocity parameters, one at a time

		     for( xvloop=2; xvloop<4; xvloop++ )  {

				                               //... Nominal starting values

                  MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						                   whitened_ptr_ptr, assoctime_ptr, covarinv_ptr, psf,
			                               XV[POSROW], XV[POSCOL], XV[VELROW], XV[VELCOL],
							               &TRTtest[0], &MLEtest[0] );

			      XV[xvloop] += Vstep;        //... Nominal + step

                  MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						                   whitened_ptr_ptr, assoctime_ptr, covarinv_ptr, psf,
			                               XV[POSROW], XV[POSCOL], XV[VELROW], XV[VELCOL],
							               &TRTtest[1], &MLEtest[1] );

			      XV[xvloop] -= 2.0 * Vstep;  //... Nominal - step

                  MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						                   whitened_ptr_ptr, assoctime_ptr, covarinv_ptr, psf,
			                               XV[POSROW], XV[POSCOL], XV[VELROW], XV[VELCOL],
							               &TRTtest[2], &MLEtest[2] );

			      XV[xvloop] += Vstep;        //... Return to nominal values before adjust


				  //........ Test for highest TRT and adjust parameter

			      ////if( TRTtest[1] > TRTtest[0] )  {

				  ////    if( TRTtest[1] > TRTtest[2] )    XV[xvloop] += Vstep;
				  ////    else                             XV[xvloop] -= Vstep;

			      ////}
			      ////else  if( TRTtest[2] > TRTtest[0] )  XV[xvloop] -= Vstep;


				  //........ Test for highest TRT and adjust parameter

			      if( MLEtest[1] > MLEtest[0] )  {

				      if( MLEtest[1] > MLEtest[2] )    XV[xvloop] += Vstep;
				      else                             XV[xvloop] -= Vstep;

			      }
			      else  if( MLEtest[2] > MLEtest[0] )  XV[xvloop] -= Vstep;


		     } //... end of 2 velocity parameters loop


		 } //... end of iteration loop

	} //... end of step change loop


	//======== Final MF result for TRT and MLE using the converged 4 parameters

    MultiFrameMatchedFilter( nframes, nrows, ncols, time_per_frame,
						     whitened_ptr_ptr, assoctime_ptr, covarinv_ptr, psf,
			                 XV[POSROW], XV[POSCOL], XV[VELROW], XV[VELCOL],
							 &MFmotion->TRT, &MFmotion->MLEdb );


	MFmotion->Xrow = XV[POSROW];
	MFmotion->Xcol = XV[POSCOL];
	MFmotion->Vrow = XV[VELROW];
	MFmotion->Vcol = XV[VELCOL];

}
*/


//======================================================================================================
//     Loop over frames to compute the measurements required for each frame.
//======================================================================================================

void    MultiFrameMeasurements( struct trackerinfo *track,  struct EMCCDparameters *params,
	                            double **demeaned_ptr_ptr,  double *assoctime_ptr,  int *psf_dimen,
								struct plate_s *plate_solution,
								struct EMCCDtrackdata *trackmeasurements )
{
int     sigframe, noiframe, aoi_dimension;
double  Time, integrated_signal, maximum_signal, background_variance;
double  azimuth_nofe, zenith_angle, pi, jdt, jdt_unixstart;
double  LST, RA, Dec;
double  rowdelta_leadingedge, coldelta_leadingedge;
struct  MFtemplate       MFstreakAOI;


    //======== Compute offsets to get to leading edge

    pi = 4.0 * atan(1.0);

	rowdelta_leadingedge = track->multi_rowspeed * params->sample_time_sec / 2.0;
	coldelta_leadingedge = track->multi_colspeed * params->sample_time_sec / 2.0;


	//======== Loop over frames

    integrated_signal = 0.0;

    for( sigframe=0; sigframe<track->nummeas; sigframe++ )  {

		//-------- Store the relative time of the measurement to the reference time

		trackmeasurements[sigframe].time = track->detected[sigframe].time;


		//-------- Want 2 frames prior for "signal-free" pixels (wraps) to estimate noise variance

		noiframe = ( sigframe + track->nummeas - 2 ) % track->nummeas;


		//-------- Time relative to first measurement point in the track
		//            We assume 1st point starts at (track->multi_rowstart, track->multi_colstart)

		Time = assoctime_ptr[sigframe] - assoctime_ptr[0];


		//-------- Generate a template over the area of interest (AOI) that is 2 x PSF size

		aoi_dimension = 2 * psf_dimen[sigframe] + 1;  //... must be an odd number

 		GenerateMFtemplateAOI( Time, params->sample_time_sec,
			                   track->multi_rowstart, track->multi_colstart,
			                   track->multi_rowspeed, track->multi_colspeed,
							   aoi_dimension, &MFstreakAOI, params->nrows, params->ncols );


		//-------- Compute the integrated signal AOI

		IntegratedSignalFromDemeaned( &MFstreakAOI, &demeaned_ptr_ptr[sigframe][0], &integrated_signal );

		trackmeasurements[sigframe].logsumsignal = -2.5 * log10( integrated_signal + 1.0e-38 );


		//-------- Compute the maximum demeaned signal in the AOI

		MaximumSignalFromDemeaned( &MFstreakAOI, &demeaned_ptr_ptr[sigframe][0], &maximum_signal );

		trackmeasurements[sigframe].maxpixel_aoi = maximum_signal;


		//-------- Compute the variance in a signal-free AOI. Report the standard deviation

		BackgroundVarianceFromDemeaned( &MFstreakAOI, &demeaned_ptr_ptr[noiframe][0], &background_variance );

		trackmeasurements[sigframe].background = sqrt( background_variance + 1.0e-99 );


		//-------- Free the AOI template for this frame

	    FreeMFtemplate( &MFstreakAOI );


		//-------- Convet the row and column centroid to azimuth and zenith angles

		trackmeasurements[sigframe].row = track->detected[sigframe].rowcentroid + rowdelta_leadingedge;
		trackmeasurements[sigframe].col = track->detected[sigframe].colcentroid + coldelta_leadingedge;

		Convert_ColRow2ThetaPhi( plate_solution, trackmeasurements[sigframe].col, trackmeasurements[sigframe].row, &zenith_angle, &azimuth_nofe );

		trackmeasurements[sigframe].phi_deg   = ( 180.0 / pi ) * azimuth_nofe ;
		trackmeasurements[sigframe].theta_deg = ( 180.0 / pi ) * zenith_angle;


		//-------- Convert Azim/Elev to RA/Dec

        jdt_unixstart = 2440587.5;  //... Julian date for January 1, 1970  0:00:00 UTC

	    jdt = jdt_unixstart  +  trackmeasurements[sigframe].time / 86400.0;  //  .time is seconds since unix start

		LST = LocalSiderealTime( jdt, params->longitude_deg * pi / 180.0 );

        ThetaPhi2RADec( zenith_angle, azimuth_nofe, params->latitude_deg * pi / 180.0, LST, &RA, &Dec );

		trackmeasurements[sigframe].RA_deg  = RA  * pi / 180.0;
		trackmeasurements[sigframe].Dec_deg = Dec * pi / 180.0;


		//-------- Fill in the magnitude estimate (NOT yet corrected for airmass)

		trackmeasurements[sigframe].magnitude = 17.14 + trackmeasurements[sigframe].logsumsignal;


	}  //... end of signal frame loop

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Functions to allocate memory for 2D arrays and free them given the data type
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void**   Allocate2Darray( int nrows, int ncols, int datatype )
{
int      k;
void**   arr;


    arr = NULL;

    if( datatype == UCHAR )  {
	    arr = (void**) malloc ( nrows * sizeof(unsigned char*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate UCHAR** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (unsigned char*) malloc ( ncols * sizeof(unsigned char) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate UCHAR* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}


    if( datatype == USHORT )  {
	    arr = (void**) malloc ( nrows * sizeof(unsigned short*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate USHORT** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (unsigned short*) malloc ( ncols * sizeof(unsigned short) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate USHORT* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}


	if( datatype == SSHORT )  {
	    arr = (void**) malloc ( nrows * sizeof(short*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate SSHORT** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (short*) malloc ( ncols * sizeof(short) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate SSHORT* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}

    if( datatype == UINT )  {
	    arr = (void**) malloc ( nrows * sizeof(unsigned int*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate UINT** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (unsigned int*) malloc ( ncols * sizeof(unsigned int) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate UINT* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}

    if( datatype == IINT )  {
	    arr = (void**) malloc ( nrows * sizeof(int*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate IINT** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (int*) malloc ( ncols * sizeof(int) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate LLIINTONG* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}

    if( datatype == ULONG )  {
	    arr = (void**) malloc ( nrows * sizeof(unsigned long*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate ULONG** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (unsigned long*) malloc ( ncols * sizeof(unsigned long) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate ULONG* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}

    if( datatype == LLONG )  {
	    arr = (void**) malloc ( nrows * sizeof(long*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate LLONG** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (long*) malloc ( ncols * sizeof(long) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate LLONG* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}

    if( datatype == FFLOAT )  {
	    arr = (void**) malloc ( nrows * sizeof(float*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate FFLOAT** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (float*) malloc ( ncols * sizeof(float) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate FFLOAT* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}

    if( datatype == DDOUBLE )  {
	    arr = (void**) malloc ( nrows * sizeof(double*) );
		if( arr == NULL )  {
			printf("ERROR ===> Could not allocate DDOUBLE** memory in Allocate2Darray\n" );
			Delay_msec(10000);
		    exit(1);
		}
        for( k=0; k<nrows; k++ )  {
			arr[k] = (double*) malloc ( ncols * sizeof(double) );
		    if( arr[k] == NULL )  {
			    printf("ERROR ===> Could not allocate DDOUBLE* memory in Allocate2Darray\n" );
			    Delay_msec(10000);
		        exit(1);
		    }
		}
	}

	return( arr );

}

//********************************************************************************************************

void   Free2Darray( void** arr, int nrows )
{
int      k;

    for( k=0; k<nrows; k++ )  free( arr[k] );

	free( arr );

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Function to find the index in a ringbuffer vector of times "time_ringbuffer" that
// represents the closest time element to the "time_desired". The dimension of the
// ring buffer vector is maxringbuffer. This is needed because partial image
// products are stored in a image array ring buffer, and extraction of the correct
// image for detection processing relies on time stamps associated with the ring
// buffer.
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int   FindClosestRingBufferTime( double time_desired, double *time_ringbuffer, int maxringbuffer )
{
int     kring, kring_closest;
double  timedelta_minimum, timedelta;


    timedelta_minimum = +1.1e+30;

	kring_closest = 0;

	for( kring=0; kring<maxringbuffer; kring++ )  {

		timedelta = fabs( time_desired - time_ringbuffer[kring] );

		if( timedelta_minimum > timedelta )  {
			timedelta_minimum = timedelta;
			kring_closest     = kring;
		}
	}

	return( kring_closest );

}

//######################################################################
//  Function to subtract image #2 from image #1, where both are unsigned
//  short data. Resulting values below zero or above 65535 are truncated
//  to 0 and 65535 respectively. A bias can be added to the difference
//  image to retain small negative values in a biased difference result.
//  The difference and bias adding is done using signed INTs so the sign
//  is retained. UNSIGNED SHORT value limits testing is done just prior
//  to final assignment to the output UNSIGNED SHORT array.
//
//          difference_image = image1 - image2 + bias
//
//######################################################################

void    SubtractImage_BiasResult( unsigned short *image1_ptr, unsigned short *image2_ptr, unsigned short *image1minus2_ptr, unsigned short diffbias, int nrows, int ncols )
{
int     krc, nrc, delta;

    nrc = nrows * ncols;

	for( krc=0; krc<nrc; krc++ )  {

		delta = (int)*image1_ptr++ - (int)*image2_ptr++ + (int)diffbias;

		if( delta <      0 )  delta = 0;
		if( delta > +65535 )  delta = +65535;

		*image1minus2_ptr++ = (unsigned short)delta;

	}

}


//######################################################################
//  Function module to determine if two hypothesized motion tracks
//  are duplicates. The constraints are that the two motion velocity
//  vectors are aligned to within AngleLimitDegrees degrees, the
//  relative speed difference is within RelVelocityPercentage percent,
//  and the starting point of one track is within DistanceLimitPixels
//  pixels of the line defined by the second track.
//
//  If the tracks are found to be duplicates, then the track with the
//  largest "metric" passed to this function determines the returned
//  function value (track #1 or track #2). Tracks that are spatially
//  unique and thus not duplicates, return a zero for the "dupeflag".
//
//  Inputs:    Xrow1          Row starting position of track 1
//             Xcol1          Col starting position of track 1
//             Vrow1          Row velocity of track 1 in pixels/second
//             Vcol1          Col velocity of track 1 in pixels/second
//             Metric1        Detection metric (e.g. MLE) of track 1
//
//             Xrow2          Row starting position of track 2
//             Xcol2          Col starting position of track 2
//             Vrow2          Row velocity of track 2 in pixels/second
//             Vcol2          Col velocity of track 2 in pixels/second
//             Metric2        Detection metric (e.g. MLE) of track 2
//
//
//  Outputs:  dupeflag   duplication flag
//                          0 = Not duplicates
//                          1 = Duplicate and Metric higher for 1st track
//                          2 = Duplicate and Metric higher for 2nd track
//
//######################################################################

int     DuplicationTest( double Xrow1, double Xcol1, double Vrow1, double Vcol1, double Metric1,
	                     double Xrow2, double Xcol2, double Vrow2, double Vcol2, double Metric2,
						 double AngleLimitDegrees, double RelVelocityPercentage, double DistanceLimitPixels )
{
double  numer, norm1, norm2;


    //======== Perform cross product of velocity unit vectors to determine if they meet an angular alignment constraint
    //              of "AngleLimitDegrees" such that sin(angle) = V1hat x V2hat

    numer = fabs( Vcol1 * Vrow2  -  Vrow1 * Vcol2 );

	norm1 = sqrt( Vcol1 * Vcol1  +  Vrow1 * Vrow1 ) + 1.0e-30;

	norm2 = sqrt( Vcol2 * Vcol2  +  Vrow2 * Vrow2 ) + 1.0e-30;

	if( numer / norm1 / norm2 > sin( AngleLimitDegrees / 57.296 ) )  return( 0 );  //... Not a duplicate


	//======== Check that the relative speeds are is within the RelVelocityPercentage

	norm1 = sqrt( Vrow1 * Vrow1  +  Vcol1 * Vcol1 ) + 1.0e-30;

	norm2 = sqrt( Vrow2 * Vrow2  +  Vcol2 * Vcol2 ) + 1.0e-30;

	if( 2.0 * 100.0 * fabs( norm1 - norm2 ) / ( norm1 + norm2 ) > RelVelocityPercentage )  return( 0 );  //... Not a duplicate


	//======== Motion velocity vectors are at least parallel, check co-alignment by testing the second track's
	//             starting position distance offset "DistanceLimitPixels" relative to the first track's projected line

	numer = fabs( Vrow1 * Xcol2  -  Vcol1 * Xrow2  +  Vcol1 * Xrow1  -  Vrow1 * Xcol1 );

	if( numer / norm1 > DistanceLimitPixels )  return( 0 );  //... Not a duplicate


	//======== Tracks are duplicates, pick the one with the largest Metric

	if( Metric1 > Metric2 )  return( 1 );
	else                     return( 2 );


}

//######################################################################
