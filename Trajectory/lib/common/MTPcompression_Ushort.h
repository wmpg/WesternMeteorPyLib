
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%                                                                         %%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%       Unsigned Short Image Compression Functions for 2-Byte Data        %%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%                                                                         %%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%             Uses Maximum Temporal Pixel MTP compression                 %%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%                                                                         %%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//  MTPcompression_UShort is the 16-bit version of compressed file generation, storage, and retrieval 
//    interface with contents of #rows, #cols, #frames/block, 1st_frame#, camera#, frame rate in Hz,
//    and arrays for maxpixel, frame#_of_maxpixel, mean_with_max_removed, stddev_with_max_removed. 
//    The mean and stddev are designed to exclude the NHIGHVAL highest temporal values per pixel to 
//    avoid contamination from bright meteors with long lasting trains on a given pixel. The file 
//    naming convention is HHcamera_yymmdd_hhmmss_msc_frameno.bin whose content is unsigned short imagery.
//
//  Date         Change Comment
//  ----------   ---------------------------------------------------------------------------------------
//  2016-07-25   Final implementation as evolved from ArchiveFTP2
//  2016-07-30   Modified to have the compress product generation as a separately called function
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%                                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%              Structure Definitions             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%                                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#define  NHIGHVAL  4  // Specifies the number of highest value pixels removed from mean and stddev calc

struct  pixelinfo_US
{                   
	//------------- Pixel level work structure during capture/compression
    long            sumpixel;           // Temporal pixel sum 
    float           ssqpixel;           // Temporal pixel squared_ptr sum (float avoids overflow)
	unsigned short  highsort[NHIGHVAL]; // Sorted high pixel values
	unsigned short  maxframe;           // Frame number of temporal maximum pixel 
    unsigned short  numequal;           // Number of equal maximum pixel values 
};

//======================================================================================================

struct  rowcolval_US
{                   
	//------------- Pixel content info for maxpixel values of a specified frame number
    unsigned short  row;           // row index 
    unsigned short  col;           // col index 
	unsigned short  val;           // maxpixel minus mean value
};

//======================================================================================================

struct  MTP_US_compressor
{                   
	//------------- Compressed image and file descriptive parameters
	long            HHversion;          // Version number of the HH file
    long            cameranumber;       // Integer camera designation
	long            firstframe;         // First frame number of this HH data block
    long            totalframes;        // Total frames in this compression sequence <= 65535
    long            nrows;              // Number of rows in the image
    long            ncols;              // Number of columns in the image
	long            ndecim;             // Decimation factor
	long            interleave;         // Interleave: 0 = progressive, 1 = odd/even, 2 = even/odd
	double          framerateHz;        // Frame rate in Hz
	char            camerafolder[256];  // Full pathname of the compressed data folder
    char            HHpathname[256];    // Full pathname of HH file
    char            HHfilename[256];    // File name only of HH file

	//------------- Information to hold for HH file write during next block capture/compress
	//              and after ingest from an HH data file. Includes compressed image array 
	//              components where in each array, the pixels are memory sequential
    char            WritHHpathname[256];// Output full pathname spec'd by 1st frame in sequence
	unsigned short *maxpixel_ptr;       // Pointer to the temporal maximum pixel array
    unsigned short *maxframe_ptr;       // Pointer to the frame number of the max pixel array
    unsigned short *avepixel_ptr;       // Pointer to the temporal average pixel array
    unsigned short *stdpixel_ptr;       // Pointer to the temporal standard deviation array

    //............. Work storage buffers at the pixel level for current block of imagery
	struct pixelinfo_US  *pixel_ptr;       // Pointer to the pixelinfo structure (nrows*ncols)
	                                       // Note the structure is sequential in memory and
	                                       //   thus any specific pixel attribute is not. For
	                                       //   example the maxframe value jumps by the
	                                       //   sizeof(pixelinfo) to get to the next pixel.
	long                  Nrcval;          // Number of maxpixels in the rowcolmax structure
	struct rowcolval_US  *rcval_ptr;       // Pointer to the rowcolmax structure (variable length)

	//------------- Pre-computed LUTs for efficient compression processing
	float          *squared_ptr;        // Square of integer numbers from 0 to 65535
    unsigned short *randomN_ptr;        // Inverse of 65536 uniform random numbers between 0 and 1
    long            randcount;          // Position index in the random number vector

	//------------- Reconstructed image support arrays and scaler components
    unsigned short *imgframe_ptr;       // Pointer to a reconstructed image frame
    long           *mpoffset_ptr;       // Pointer to the pointer offset into the maxpixel array
    long           *mpcumcnt_ptr;       // Pointer to the cumulative sum of the frame count
    long           *framecnt_ptr;       // Pointer to the frame counter histogram of maxpixel
    unsigned short  global_ave;         // Global median of the FOV mean
    unsigned short  global_std;         // Global median of the FOV standard deviation

};



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%                                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%        Function Prototype Declarations         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%                                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int     MTPcompression_UShort_GetTimeofDay( long *seconds_jan1st1970,            //millisecond accuracy date/time (Windows)
								            long *microseconds            );


void        MTPcompression_UShort_MemSetup( long                        image_nrows,    //allocate memory and set parameters
                                            long                        image_ncols,
                                            long                        nframes,
                                            long                        cameranumber,
											long                        ndecim,
											long                        interleave,
								            double                      framerateHz,
                                            char                       *camerafolder,
                                            struct MTP_US_compressor   *imagecmp );
                              
                     
void        MTPcompression_UShort_Compress( long                        framenumber,     //compress newest frame into sequence
                                            unsigned short             *image_ptr,
                                            long                        image_nrows,
                                            long                        image_ncols,
                                            struct MTP_US_compressor   *imagecmp );

void        MTPcompression_UShort_Products( struct MTP_US_compressor   *imagecmp );      //post block's last frame compression product generation

                                                            
void        MTPcompression_UShort_Filename( long                        framenumber,     //generate HH filename from cam#, date/time, frm#
                                            struct MTP_US_compressor   *imagecmp );
                              
void   MTPcompression_UShort_FilenameParse( char                       *HHpathname,      //extract date/time, cam#, frm# from pathname or filename
                                            long                       *year,
                                            long                       *month,
                                            long                       *day,
                                            long                       *hour,
                                            long                       *minute,
                                            long                       *second,
                                            long                       *milliseconds,
							                long                       *cameranumber, 
								            long                       *framenumber );
                              
void        MTPcompression_UShort_DateTime( long                       *year,
                                            long                       *month,
                                            long                       *day,
                                            long                       *hour,
                                            long                       *minute,
                                            long                       *second,
                                            long                       *milliseconds );  //get the UT date and time to the millisecond

void       MTPcompression_UShort_FileWrite( struct MTP_US_compressor   *imagecmp );      //max,frame#,mean,stddev,aftermax to file
                             
int     MTPcompression_UShort_FileValidity( char                       *HHpathname );    //checks validity of file parameters

void        MTPcompression_UShort_FileRead( char                       *HHpathname,      //read max,frame#,mean,stddev from file
                                            struct MTP_US_compressor   *imagecmp );      //   and populate imagecmp structure

void       MTPcompression_UShort_PrepBuild( struct MTP_US_compressor   *imagecmp );      //finds pointer offsets to maxpixel per frame#

void       MTPcompression_UShort_RowColVal( long                        fileframenumber, //fills rowcolval structure for user frame number
                                            struct MTP_US_compressor   *imagecmp );
                                                   
void      MTPcompression_UShort_FrameBuild( long                        fileframenumber, //build imgframe from the mean and fill-in
								            struct MTP_US_compressor   *imagecmp );      //   with the frame number requested
                              
void      MTPcompression_UShort_GlobalMean( struct MTP_US_compressor   *imagecmp );      //find the global median of the mean image
                              
void     MTPcompression_UShort_GlobalSigma( struct MTP_US_compressor   *imagecmp );      //find the global median of sigma (std dev)
                              
                              
void         MTPcompression_UShort_Flatten( short                      *flattened_ptr,   //flattened = 128*(prebuilt_imgframe-mean)/sigma
                                            struct MTP_US_compressor   *imagecmp );
                              
void    MTPcompression_UShort_BuildFlatten( short                      *flattened_ptr,   //flattened = 128*(build_imgframe-mean)/sigma
                                            long                        fileframenumber,
                                            struct MTP_US_compressor   *imagecmp );
                              
                              
void      MTPcompression_UShort_MemCleanup( struct MTP_US_compressor   *imagecmp );      //free all allocated memory

void    MTPcompression_UShort_NullPointers( struct MTP_US_compressor   *imagecmp );      //NULL all pointers



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%                                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%   Include Statements for C Library Functions   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%                                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#pragma warning(disable: 4996)  // disable warning on strcpy, fopen, ...

#include <stdio.h>         // fopen, fread, fwrite, fclose, getchar, printf 
#include <math.h>          // intrinsic math functions: sqrt
#include <string.h> 
#include <stdlib.h> 
                     


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%                                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%         MTP Compression Functions              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%                                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


//#######################################################################################################
//                             MTPcompression_UShort_MemSetup
//
//  Compression structure initialization called a single time after program startup, once the 
//  image dimensions, number of frames per block, camera number, framerate in Hertz, and the
//  output destination folder for the compressed files are known. Recommend calling 
//  MTPcompression_UShort_NullPointers first (immediately after program startup), followed 
//  later by MTPcompression_UShort_MemSetup once the argument list parameters are all available.
//
//  The function allocates memory for all compression structure arrays and vectors. Call the
//  function MTPcompression_UShort_MemCleanup once ALL compression processing is completed, to  
//  free memory at the end of the program.
//
//#######################################################################################################


void    MTPcompression_UShort_MemSetup( long                        image_nrows,
                                        long                        image_ncols,
                                        long                        totalframes,
                                        long                        cameranumber,
										long                        ndecim,
										long                        interleave,
							            double                      framerateHz,
                                        char                       *camerafolder,
                                        struct MTP_US_compressor   *imagecmp )
{
long  k, nrc, arand;


   //----- Make sure memory is released for this imagecmp structure and set version

   MTPcompression_UShort_MemCleanup( imagecmp );

   imagecmp->HHversion = -101;   // CAMS HH file format designation for 16 bit


   //----- Set the output destination folder for Windows or Linux

   #ifdef _WIN32
      if( camerafolder == NULL ) strcpy( imagecmp->camerafolder, "./"         );
      else                       strcpy( imagecmp->camerafolder, camerafolder );
   #else //LINUX
      if( camerafolder == NULL ) strcpy( imagecmp->camerafolder, "."          );
      else                       strcpy( imagecmp->camerafolder, camerafolder );
   #endif


   //----- Check and save the number of frames for compressing the imagery block
   
   if( totalframes < 8  ||  totalframes > 65535 )  {
       printf(" ERROR:  Totalframes must be between 8 and 65535 in MTPcompression_UShort_MemSetup\n");
       Delay_msec(10000);
	   exit(1);
   }
   
   imagecmp->totalframes = totalframes;
 

   //----- Set the framerate, image dimensions, and camera number

   imagecmp->framerateHz  = framerateHz;

   imagecmp->nrows        = image_nrows;
   
   imagecmp->ncols        = image_ncols;
   
   imagecmp->cameranumber = cameranumber;

   imagecmp->ndecim       = ndecim;

   imagecmp->interleave   = interleave;


   //-----Allocate memory for all arrays and vectors

   nrc = (long)image_nrows * (long)image_ncols;
    
   imagecmp->maxpixel_ptr = (unsigned short*)  malloc( nrc * sizeof(unsigned short)  );  
   imagecmp->maxframe_ptr = (unsigned short*)  malloc( nrc * sizeof(unsigned short)  );  
   imagecmp->avepixel_ptr = (unsigned short*)  malloc( nrc * sizeof(unsigned short)  );  
   imagecmp->stdpixel_ptr = (unsigned short*)  malloc( nrc * sizeof(unsigned short)  );
      
   imagecmp->imgframe_ptr = (unsigned short*)  malloc( nrc * sizeof(unsigned short)  );
   imagecmp->mpoffset_ptr = (          long*)  malloc( nrc * sizeof(          long)  );
   imagecmp->mpcumcnt_ptr = (          long*)  malloc( imagecmp->totalframes * sizeof( long )  );
   imagecmp->framecnt_ptr = (          long*)  malloc( imagecmp->totalframes * sizeof( long )  );  
   
   imagecmp->randomN_ptr  = (unsigned short*)  malloc( 65536L * sizeof(unsigned short)  );
   imagecmp->squared_ptr  = (float*         )  malloc( 65536L * sizeof(float         )  );

   imagecmp->pixel_ptr    = (struct pixelinfo_US*) malloc( nrc * sizeof( struct pixelinfo_US ) );


   if( imagecmp->pixel_ptr          == NULL ||
	   imagecmp->maxpixel_ptr       == NULL ||
	   imagecmp->maxframe_ptr       == NULL ||
	   imagecmp->avepixel_ptr       == NULL ||
	   imagecmp->stdpixel_ptr       == NULL ||
	   imagecmp->imgframe_ptr       == NULL ||   
	   imagecmp->mpoffset_ptr       == NULL ||
	   imagecmp->framecnt_ptr       == NULL ||
	   imagecmp->mpcumcnt_ptr       == NULL ||
	   imagecmp->randomN_ptr        == NULL ||
	   imagecmp->squared_ptr        == NULL    )  {
       printf(" ERROR:  Memory not allocated in MTPcompression_UShort_MemSetup\n");
       Delay_msec(10000);
	   exit(1);
   }


   //----- Set the look-up-table (LUT) values for squared_ptr and square roots of integers

   for( k=0; k<65536L; k++ )  imagecmp->squared_ptr[k] = (float)k * (float)k;
   
  
   //----- Set the random number selection to ensure repeated maxpixel value get maxframe numbers
   //        distributed evenly in time (frame number). Uses a very simple random number generator.

   arand = 1;
   
   for( k=0; k<65536L; k++ )  {  //... equivalent to (double)RAND_MAX / (double)rand()
   
        arand = ( arand * 32719L + 3L ) % 32749L;
   
        imagecmp->randomN_ptr[k] = (unsigned short)( 32749.0 / (double)(1 + arand) );
   }
   
   imagecmp->randcount = 0;

   //----- Set the startup file and path names to blank

   strcpy( imagecmp->HHpathname, " " );
   strcpy( imagecmp->HHfilename, " " );

   	
   //-------- Clear the entire memory contents of the per pixel work structure and maxpixel offsets

   memset( imagecmp->pixel_ptr, 0, image_nrows * image_ncols * sizeof( struct pixelinfo_US ) );

   memset( imagecmp->mpoffset_ptr, 0, image_nrows * image_ncols * sizeof(unsigned long) );
   
}


//#######################################################################################################
//                              MTPcompression_UShort_Compress
//
//  Primary user function to compress an imagery block of "totalframes" images on-the-fly. 
//  The function is called once per image frame in temporally (frame number) increasing order. 
//  The compression works through a block of imagery of duration "totalframes" and then 
//  restarts on the next block of data. The start is triggered when the "framenumber" modulo
//  "totalframes" equals zero (Note that prior to the modulo zero call to this function,
//  one must call MTPcompression_UShort_Products to have the previous block's filename copied to 
//  the "WritHHpathname", and the previous block's compression products computed and copied off,  
//  at which point the products can be written out using MTPcompression_UShort_FileWrite or used
//  directly in processing).
//  The sequence continues compressing a given block of imagery through "framenumber" modulo
//  "totalframes" equals totalframes-1. Then once the modulo cycles back to zero all the 
//  working arrays are reset. Having the separate function MTPcompression_UShort_Products
//  allows the user to hold off starting a new compression sequence until the writing of
//  a previous sequence is ensured to be completed (that is wait for the last HH file to be
//  written and closed BEFORE the next compression call where modulo = 0). This provides 
//  totalframes worth of time to elapse to write the file plus any additional time if paused 
//  at the completion of the last block of data (AFTER the call when modulo = totalframes-1).
//
//         IMPORTANT NOTE: Do not start writing the latest complete block until 
//                         MTPcompression_UShort_Poducts is called. This same holds
//                         true for processing data from a previous block.
//
//  Note that the image_ptr array must be dimensioned exactly as the compression structure's
//  internal allocation of nrows x ncols in column preference order.
//
//  The compression keeps track of the NHIGHVAL highest values per pixel to remove them from
//  the mean and standard deviation calculation to minimize their contamination by meteors. It
//  employs an insertion sort technique for very fast sorting of short lists. The compression 
//  also checks for repeated instances of the maxpixel value of a given pixel and uses a 
//  randomization formula to evenly distribute those in time (frame number).
//
//#######################################################################################################


void    MTPcompression_UShort_Compress( long                        framenumber,  // starts at zero and increments externally
                                        unsigned short             *image_ptr,    
                                        long                        image_nrows,
                                        long                        image_ncols,
                                        struct MTP_US_compressor   *imagecmp )
{
long                  kthrow, kthcol, randcount, ksort;
unsigned short        datum, framemodulo;
unsigned short       *datum_ptr;
struct pixelinfo_US  *pixel_ptr;


  //======== Test for valid dimensionality of incoming image frame

  if( image_nrows != imagecmp->nrows  ||  image_ncols != imagecmp->ncols )  {
  
      printf("Mismatch of number of rows or columns in call to MTPcompression_UShort_Compress \n");
      Delay_msec(10000);
	  exit(1);
  }
    
  
  //======== Set the framecount to fall between 0 (first frame in the block) 
  //         and 2^nbits - 1 (last frame in the block)
    
  framemodulo = (unsigned short)( framenumber % imagecmp->totalframes );
  
  
  //======== First call for the next block sequence of images

  if( framemodulo == 0 )  {

      //-------- Build the new block's filename from the camera number, date/time, and framenumber
      
      MTPcompression_UShort_Filename( framenumber, imagecmp );


	  //-------- Assign the working pixelinfo structure pointer.

	  pixel_ptr    = imagecmp->pixel_ptr;

	  datum_ptr    = image_ptr;

	   
      //-------- Initialize the intermediate products for the next block
	  //           given the first image frame.
    
      for( kthrow=0; kthrow<image_nrows; kthrow++ )  {          // Loop over image frame rows
      
           for( kthcol=0; kthcol<image_ncols; kthcol++ )  {     // Loop over image frame columns

 				//-------- Initialize the pixelinfo contents for the new block's first frame

			    datum = *datum_ptr++;                           // Pull pixel value, increment to next pixel
       
				pixel_ptr->sumpixel = (long)datum;              // Temporal pixel sum
            
                pixel_ptr->ssqpixel = imagecmp->squared_ptr[ datum ];  // Temporal pixel-squared_ptr sum

 				pixel_ptr->highsort[0] = datum;                 // Set highest pixel value thus far

				for( ksort=1; ksort<NHIGHVAL; ksort++ )  pixel_ptr->highsort[ksort] = 0;

				pixel_ptr->maxframe = 0;                        // Maxpixel frame number is first frame = 0

                pixel_ptr->numequal = 1;                        // Set number of repeated highest pixel values

				pixel_ptr++;                                    // Increment the pixelinfo structure pointer

           } //... end of column loop
       
      } //... end of row loop

  }

  
  //======== Second thru last call for the block sequence of images
  
  else  {
      
	  //-------- Compute running sums and check for new maxpixels and/or sort high pixel values

      randcount = imagecmp->randcount;  // Load the random counter for repeated highest pixel frame# selection
  
	  pixel_ptr = imagecmp->pixel_ptr;

	  datum_ptr = image_ptr;


      for( kthrow=0; kthrow<image_nrows; kthrow++ )  {        // Loop over image frame rows
      
           for( kthcol=0; kthcol<image_ncols; kthcol++ )  {   // Loop over image frame columns
       
			    //-------- Extract the pixel value and compute sums

			    datum = *datum_ptr++;                               // Pull pixel value, increment to next pixel 

                pixel_ptr->sumpixel += (long)datum;                 // Temporal pixel sum
            
                pixel_ptr->ssqpixel += imagecmp->squared_ptr[ datum ];  // Temporal pixel-squared_ptr sum


			    //-------- Check if pixel is greater than smallest value in the sorted high-value list

			    if( datum >= pixel_ptr->highsort[NHIGHVAL-1] )  {

				    //...... Check if pixel greater than the current maxpixel

                    if( datum > pixel_ptr->highsort[0] )  {

						//- Slide all the insertion sorted values down
                    
 					    for( ksort=NHIGHVAL-1; ksort>0; ksort-- )  pixel_ptr->highsort[ksort] = pixel_ptr->highsort[ksort - 1];
					    
						pixel_ptr->highsort[0] = datum;
						pixel_ptr->maxframe    = framemodulo; 
						pixel_ptr->numequal    = 1;

                    } //... end of new maxpixel value


					//...... Check if pixel is equal to the current maxpixel

                    else if( datum == pixel_ptr->highsort[0] )  {  
                    
 						//- Slide all the insertion sorted values down

						for( ksort=NHIGHVAL-1; ksort>0; ksort-- )  pixel_ptr->highsort[ksort] = pixel_ptr->highsort[ksort - 1];
					    
						//- Set maxframe based on random draw to avoid biasing early frame #s

                        randcount = (randcount + 1) % 65536L;

                        pixel_ptr->numequal += 1;

                        if( pixel_ptr->numequal <= imagecmp->randomN_ptr[randcount] )  pixel_ptr->maxframe = framemodulo;                                          

                    }  //... end of repeated maxpixel value


					//...... Pixel value falls between sorted highest values so perform an insertion sort

					else if( datum > pixel_ptr->highsort[NHIGHVAL-1] )  {

					    for( ksort=NHIGHVAL-1; ksort>0; ksort-- )  {

						     if( datum <= pixel_ptr->highsort[ksort - 1] )  break;

						     pixel_ptr->highsort[ksort] = pixel_ptr->highsort[ksort - 1];

						}

					    pixel_ptr->highsort[ksort] = datum;

					}  //... end of between maxpixel and smallest value on high sorted list


                } //... end of IF test for pixel value must be greater than smallest pixel in high sorted list


				//-------- Increment the pixelinfo structure pointer to work on next pixel

				pixel_ptr++;


           } //... end of column loop
       
      } //... end of row loop

      imagecmp->randcount = randcount;    // Save the latest random index prior to function exit

  
  } //... end of IF first or subsequent calls to compress function

  
}

//#######################################################################################################
//                              MTPcompression_UShort_Products
//
//  User function to perform the compression "product" generation after a complete block of 
//  "totalframes" has been passed through the function MTPcompression_UShort_Compress. The 
//  computation of the products for maxpixel, maxframe, avepixel, and stdpixel is performed
//  and the user can make immediate follow-up calls to MTPcompression_UShort_Compress without
//  overwritting these products. The previous block's filename is copied to "WritHHpathname", 
//  the previous block's compression arrays are computed and can then be written out using 
//  MTPcompression_UShort_FileWrite or used directly in processing.
//
//  This function allows the user to hold off starting a new compression sequence until the 
//  writing of a previous sequence is ensured to be completed (that is wait for the last HH 
//  file to be written and closed BEFORE the next compression call where modulo = 0). This  
//  provides totalframes worth of time to elapse to write the file plus any additional time if  
//  paused at the completion of the last block of data (AFTER the call to the function
//  MTPcompression_UShort_Compress when modulo = totalframes-1).
//
//         IMPORTANT NOTE: Do not start writing the latest complete block until 
//                         MTPcompression_UShort_Products is called. The same holds 
//                         true for processing data from a previous block.
//
//#######################################################################################################


void    MTPcompression_UShort_Products( struct MTP_US_compressor   *imagecmp )
{
long                  kthrow, kthcol, ksort, sumpix;
float                 ssqpix;
double                avepix, varpix, N, N1;
unsigned short        datum;
unsigned short       *maxpixel_ptr;
unsigned short       *maxframe_ptr;
unsigned short       *avepixel_ptr;
unsigned short       *stdpixel_ptr;
struct pixelinfo_US  *pixel_ptr;

 

	  //-------- Move the last block's full pathname to the write pathname buffer

	  strcpy( imagecmp->WritHHpathname, imagecmp->HHpathname );


	  //-------- Assign the previous block's data array output pointers to save content for writing
	  //           to the HH file as well as the working pixelinfo structure pointer.

	  maxpixel_ptr = imagecmp->maxpixel_ptr; 
      maxframe_ptr = imagecmp->maxframe_ptr;
      avepixel_ptr = imagecmp->avepixel_ptr;
      stdpixel_ptr = imagecmp->stdpixel_ptr;

	  pixel_ptr    = imagecmp->pixel_ptr;


      //-------- Compute and copy out the last block's compressed products

	  N  = (double)( imagecmp->totalframes - NHIGHVAL     );
	  N1 = (double)( imagecmp->totalframes - NHIGHVAL - 1 );

    
      for( kthrow=0; kthrow<imagecmp->nrows; kthrow++ )  {          // Loop over image frame rows
      
           for( kthcol=0; kthcol<imagecmp->ncols; kthcol++ )  {     // Loop over image frame columns

				//-------- Compute the previous block's temporal mean and standard deviation by
			    //           removing the sorted list of highest values for this pixel

				sumpix = pixel_ptr->sumpixel; 

			    ssqpix = pixel_ptr->ssqpixel;

				for( ksort=0; ksort<NHIGHVAL; ksort++ )  {

					 datum = pixel_ptr->highsort[ksort];
					
					 sumpix -= (long)datum; 

					 ssqpix -= imagecmp->squared_ptr[ datum ];

				}

				avepix = (double)sumpix / N;

				varpix = ( (double)ssqpix - avepix * (double)sumpix ) / N1;


 				//-------- Copy the compressed content to the output arrays

 				*maxpixel_ptr++ = pixel_ptr->highsort[0];

				*maxframe_ptr++ = pixel_ptr->maxframe;

				*avepixel_ptr++ = (unsigned short)( avepix + 0.5 );  //... round to nearest integer
			   
				*stdpixel_ptr++ = (unsigned short)sqrt( varpix + 1.0 ) ;  //... add 1 to avoid divide by zero later


				pixel_ptr++;                                    // Increment the pixelinfo structure pointer

           } //... end of column loop
       
      } //... end of row loop

  
}


//#######################################################################################################
//                           MTPcompression_UShort_Filename
//
//  Constructs a HH filename with format \path\HH######_YYYYMMDD_HHMMSS_MSC_FRAMENO.bin and 
//  places it in the compression structure "HHpathname" for full path name, and "HHfilename"
//  for only the HH filename itself.
//
//#######################################################################################################

void   MTPcompression_UShort_Filename( long framenumber,  struct MTP_US_compressor *imagecmp )
{
long  utyear, utmonth, utday, uthour, utminute, utsecond, utmilliseconds;


   //====== Get time from onboard clock - preferably in universal time (UT)
   
   MTPcompression_UShort_DateTime( &utyear, &utmonth, &utday, &uthour, &utminute, &utsecond, &utmilliseconds );
 
            
   //====== Build filename

   sprintf( imagecmp->HHpathname, "%sHH_%06li_%04li%02li%02li_%02li%02li%02li_%03li_%07li.bin",
            imagecmp->camerafolder,
            imagecmp->cameranumber,
            utyear, utmonth, utday, uthour, utminute, utsecond, utmilliseconds,
            framenumber  );

   //====== Pull out HH name only just in case someone wants it after this call (legacy)

   strncpy( imagecmp->HHfilename, strrchr( imagecmp->HHpathname, 72 )-1, 41 );  // 41 characters from HH to .bin

   imagecmp->HHfilename[41] = '\0';  // Null terminate the string
               
}


//#######################################################################################################
//                          MTPcompression_UShort_FilenameParse
//
//  Parses the HH pathname or filename to extract date, time, camera number, and starting framenumber.
//#######################################################################################################

void   MTPcompression_UShort_FilenameParse( char *HHpathname, 
	                                        long *utyear, long *utmonth,  long *utday, 
											long *uthour, long *utminute, long *utsecond, long *utmilliseconds, 
											long *cameranumber, 
											long *framenumber )
{
char *H_ptr, substring[255];


   //  directory_pathname... HH_CAMNO_YYYYMMDD_HHMMSS_MSC_FRAMENO.bin

   H_ptr = strrchr( HHpathname, 72 );  //... last occurance of H = ASCII 72 
   
   strncpy( substring, H_ptr-1, 41 );  // 41 characters from HH to .bin

   substring[41] = '\0';  // Null terminate the string
   
   sscanf( substring, "HH_%6ld_%4ld%2ld%2ld_%2ld%2ld%2ld_%3ld_%7ld.bin", 
                        cameranumber,
                        utyear, utmonth, utday, 
                        uthour, utminute, utsecond, utmilliseconds,
                        framenumber );

}


//#######################################################################################################
//                            MTPcompression_UShort_DateTime
//
//  Populates the date and time structure given the current value read off the system clock.
//#######################################################################################################


void   MTPcompression_UShort_DateTime( long *utyear, long *utmonth,  long *utday, 
									   long *uthour, long *utminute, long *utsecond, long *utmilliseconds )
{
long       seconds_jan1st1970, microseconds;
time_t     currtime;
struct tm  gmt;


   //======== Get time from onboard clock in universal time (UT)

#ifdef _WIN32

   MTPcompression_UShort_GetTimeofDay( &seconds_jan1st1970, &microseconds );
   currtime = seconds_jan1st1970;

#else  //LINUX

   time( &currtime );
   microseconds = 0;

#endif // WINDOWS or LINUX // 


   gmt = *gmtime( &currtime );
          
   *utyear         = gmt.tm_year + 1900;    
   *utmonth        = gmt.tm_mon  + 1;       
   *utday          = gmt.tm_mday;           
   *uthour         = gmt.tm_hour;
   *utminute       = gmt.tm_min;
   *utsecond       = gmt.tm_sec;
   *utmilliseconds = microseconds / 1000;
               
}


//#######################################################################################################
//                         MTPcompression_UShort_GetTimeofDay
//
//  Windows specific date and time retrieval from the system clock to millisecond accuracy.
//#######################################################################################################
 

#ifdef _WIN32

int  MTPcompression_UShort_GetTimeofDay( long *seconds_jan1st1970, long *microseconds )
{
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
  //======== Define a structure to receive the current Windows filetime

  FILETIME ft;

  //======== Initialize the present time to 0 and the timezone to UTC

  unsigned __int64 tmpres = 0;
 
  GetSystemTimeAsFileTime(&ft);
 
  //======== The GetSystemTimeAsFileTime returns the number of 100 nanosecond 
  //           intervals since Jan 1, 1601 in a structure. Copy the high bits to 
  //           the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
  
  tmpres |= ft.dwHighDateTime;
  tmpres <<= 32;
  tmpres |= ft.dwLowDateTime;
 
  //======== Convert to microseconds by dividing by 10
  
  tmpres /= 10;
 
  //======== The Unix epoch starts on Jan 1 1970.  Need to subtract the difference 
  //           in seconds from Jan 1 1601.

  tmpres -= DELTA_EPOCH_IN_MICROSECS;
 
  //======== Finally change microseconds to seconds and place in the seconds value. 
  //           The modulus picks up the microseconds.

  *seconds_jan1st1970 = (long)(tmpres / 1000000UL);
  *microseconds       = (long)(tmpres % 1000000UL);

  return( 0 );

}

#endif  // WINDOWS or LINUX //



//#######################################################################################################
//                           MTPcompression_UShort_FileWrite
//
//  Writes the UShort CAMS HH file format (version = -4) given the values contained in
//  maxpixel, maxframe, avepixel, and the stdpixel arrays. Note that the HH filename is
//  set at the START of any given block's compression processing and represents the time of 
//  the first frame collected in the block sequence, but the filename is not available in the
//  WritHHpathname string until AFTER MTPcompression_UShort_Compress is called for the start of 
//  the NEXT block of data. The same holds true for the availability of the arrays listed above.

//  Thus this write function should not be called until AFTER the next block's compression has
//  started (framecount modulo totalframes equals zero on a call to MTPcompression_UShort_Compress). 
//  Also note that the arrays to be written to the HH file during this write call will NOT get 
//  overwritten until a call to MTPcompression_UShort_Compress again occurs for the framecount modulo 
//  totalframes equals zero. This allows the user to hold off compressing subsequent blocks
//  until the write of the previous block is determined to be completed. One can set this up
//  using signaling in a multi-threaded environment without changing these functions.
//
//#######################################################################################################

void   MTPcompression_UShort_FileWrite( struct MTP_US_compressor *imagecmp )
{
long   firstframe, cameranumber, framerate1000;
long   utyear, utmonth, utday, uthour, utminute, utsecond, utmilliseconds;
FILE  *WriteFile;


   MTPcompression_UShort_FilenameParse( imagecmp->WritHHpathname, 
	                                    &utyear, &utmonth, &utday, 
									    &uthour, &utminute, &utsecond, &utmilliseconds, 
									    &cameranumber, &firstframe );

   framerate1000 = (long)( imagecmp->framerateHz * 1000.0 );


   //======== Write the entire HH file contents

   if( (WriteFile = fopen( imagecmp->WritHHpathname, "wb")) == NULL )  {
       printf(" Cannot open output file %s for writing \n", imagecmp->WritHHpathname );
       Delay_msec(10000);
	   exit(1);
   }
   fwrite( &imagecmp->HHversion,   sizeof(long), 1, WriteFile ); 
   fwrite( &imagecmp->nrows,       sizeof(long), 1, WriteFile ); 
   fwrite( &imagecmp->ncols,       sizeof(long), 1, WriteFile ); 
   fwrite( &imagecmp->totalframes, sizeof(long), 1, WriteFile ); 
   fwrite( &firstframe,            sizeof(long), 1, WriteFile ); 
   fwrite( &cameranumber,          sizeof(long), 1, WriteFile );
   fwrite( &imagecmp->ndecim,      sizeof(long), 1, WriteFile );
   fwrite( &imagecmp->interleave,  sizeof(long), 1, WriteFile );
   fwrite( &framerate1000,         sizeof(long), 1, WriteFile );

   fwrite( imagecmp->maxpixel_ptr, sizeof(unsigned short), imagecmp->nrows*imagecmp->ncols, WriteFile ); 
   fwrite( imagecmp->maxframe_ptr, sizeof(unsigned short), imagecmp->nrows*imagecmp->ncols, WriteFile ); 
   fwrite( imagecmp->avepixel_ptr, sizeof(unsigned short), imagecmp->nrows*imagecmp->ncols, WriteFile ); 
   fwrite( imagecmp->stdpixel_ptr, sizeof(unsigned short), imagecmp->nrows*imagecmp->ncols, WriteFile ); 

   fclose( WriteFile );

}

//#######################################################################################################
//                           MTPcompression_UShort_FileValidity
//
//  Reads any CAMS HH file format, reads through the file contents, and checks for valid 
//  ranges of parameters and data section sizes. Error codes on return:
//    0 = normal HH file
//    1 = Cannot open file for reading
//    2 = Number of rows out of range
//    3 = Number of columns out of range
//    4 = Number of frames out of range
//    5 = First frame mismatch in filename
//    6 = Camera number mismatch in filename
//    7 = Decimation factor or interleave out of range
//    8 = Frame rate <0 or >10000 Hz
//    9 = Total file size incorrect
//   10 = HH file version not handled in MTPcompression_UShort_FileValidity
//#######################################################################################################

int    MTPcompression_UShort_FileValidity( char *HHpathname )
{
long   HHversion, nrows, ncols, totalframes, framerate1000, nbytes, ndecim, interleave;
long   firstframe, firstframe2, cameranumber, cameranumber2;
long   utyear, utmonth, utday, uthour, utminute, utsecond, utmilliseconds;
FILE  *ReadFile;



   //======== Open the HH file for reading as a binary file

   if( (ReadFile = fopen( HHpathname, "rb")) == NULL )  {
       return(1);
   }
      
   MTPcompression_UShort_FilenameParse( HHpathname, 
	                                    &utyear, &utmonth, &utday, 
									    &uthour, &utminute, &utsecond, &utmilliseconds, 
									    &cameranumber2, &firstframe2 );


   //======== Read the first integer in file and determine version

   fread( &HHversion, sizeof(long), 1, ReadFile );

   if( HHversion != -101 )  return(10); // version not implemented in this function


   //======== Skip the header entries for all CAMS formats

   fread( &nrows, sizeof(long), 1, ReadFile );

       if( nrows <= 0  ||  nrows >= 32768L )  {
		   fclose( ReadFile );
		   return(2);
	   }


   fread( &ncols, sizeof(long), 1, ReadFile ); 

       if( ncols <= 0  ||  ncols >= 32768L )  {
		   fclose( ReadFile );
		   return(3);
	   }


   fread( &totalframes, sizeof(long), 1, ReadFile );

       if( totalframes < 8  ||  totalframes > 65535 )  {
		   fclose( ReadFile );
		   return(4);
	   }


   fread( &firstframe, sizeof(long), 1, ReadFile );

       if( firstframe != firstframe2 )  {
		   fclose( ReadFile );
		   return(5);
	   }


   fread( &cameranumber, sizeof(long), 1, ReadFile );

       if( cameranumber != cameranumber2 )  {
		   fclose( ReadFile );
		   return(6);
	   }


   fread( &ndecim, sizeof(long), 1, ReadFile );

       if( ndecim < 1  ||  ndecim > 128 )  {
		   fclose( ReadFile );
		   return(7);
	   }


   fread( &interleave, sizeof(long), 1, ReadFile );

       if( interleave < 0  ||  interleave > 2 )  {
		   fclose( ReadFile );
		   return(7);
	   }


   fread( &framerate1000, sizeof(long), 1, ReadFile );

	   if( framerate1000 < 0  ||  framerate1000 > 10000000 )  {  // <=10000 Hz
		   fclose( ReadFile );
		   return(8);
	   }


   //---- final test on size of the data area

   fseek( ReadFile, 0, SEEK_END );

       nbytes = 36L + 8L * nrows * ncols;

	   if( nbytes != ftell( ReadFile ) )  {
		   fclose( ReadFile );
		   return(9);
	   }
           

   fclose( ReadFile );

   return(0);

}


//#######################################################################################################
//                             MTPcompression_UShort_FileRead
//
//  Reads any CAMS HH file format, allocates memory to and fills in the "imagecmp" structure.
//  Note that this MTPcompression_UShort_FileRead function calls the MTPcompression_UShort_MemSetup function
//  and the user must call MTPcompression_UShort_MemCleanup when done with "imagecmp" (special note that
//  the function MTPcompression_UShort_MemSetup does do a MTPcompression_UShort_MemCleanup call on entry in the
//  event the user forgot to make the memory cleanup call - it tests the structure pointer
//  for a NULL value - see beginning of MTPcompression_UShort_MemSetup). Thus the user need
//  only bother to clean up memory at the end of the program.
//
//  This reader also computes the global mean, standard deviation and fills out the pointer 
//  offsets needed for fast reconstruction of image frames. The reader populates the maxpixel,
//  maxframe, avepixel, stdpixel arrays as well as all the associated header info such as image 
//  dimensions, totalframes, frame rate, ...
//
//#######################################################################################################

void   MTPcompression_UShort_FileRead( char *HHpathname, struct MTP_US_compressor *imagecmp )
{
long   nrows, ncols, totalframes, HHversion, framerate1000, ndecim, interleave;
long   firstframe, cameranumber, framenumberdummy;
long   utyear, utmonth, utday, uthour, utminute, utsecond, utmilliseconds;
double framerateHz;
char  *H_ptr, camerafolder[256];
FILE  *ReadFile;


   //======== Open the HH file for reading as a binary file

   if( (ReadFile = fopen( HHpathname, "rb")) == NULL )  {
       printf("ERROR ===> Cannot open input file %s for reading \n", HHpathname );
       Delay_msec(10000);
	   exit(1);
   }


   //======== Read the header information according to the CAMS file format specified

   fread( &HHversion,     sizeof(long), 1, ReadFile ); 
   fread( &nrows,         sizeof(long), 1, ReadFile ); 
   fread( &ncols,         sizeof(long), 1, ReadFile ); 
   fread( &totalframes,   sizeof(long), 1, ReadFile );
   fread( &firstframe,    sizeof(long), 1, ReadFile );
   fread( &cameranumber,  sizeof(long), 1, ReadFile );
   fread( &ndecim,        sizeof(long), 1, ReadFile );
   fread( &interleave,    sizeof(long), 1, ReadFile );
   fread( &framerate1000, sizeof(long), 1, ReadFile );


   //..... Fill in and check parameters

   framerateHz = (double)framerate1000 / 1000.0;

   if( imagecmp->HHversion != -101 )  {
	   printf(" HHversion = %d  header read NOT implemented in MTPcompression_UShort_FileRead\n", imagecmp->HHversion );
	   Delay_msec(10000);
	   exit(1);
   }

 
   //======== Overwrite the cameranumber with the value in the HH filename (rather than the header)
      
   MTPcompression_UShort_FilenameParse( HHpathname, 
	                                    &utyear, &utmonth, &utday, 
									    &uthour, &utminute, &utsecond, &utmilliseconds, 
									    &cameranumber, &framenumberdummy );


   //======== Get the folder pathname, filename, and full pathname

   H_ptr = strrchr( HHpathname, 72 );  //...ASCII 72 is H
   
   strncpy( camerafolder, HHpathname, H_ptr-HHpathname-1 );
  
   camerafolder[H_ptr-HHpathname-1] = '\0';   // Null terminate the string

   
   strncpy( imagecmp->HHfilename, H_ptr-1, 41 );  // 41 characters from HH to .bin

   imagecmp->HHfilename[41] = '\0';  // Null terminate the string


   strcpy( imagecmp->HHpathname, HHpathname );


   //======== Set up the compression structure memory BEFORE reading data block arrays
   //           and move scalers into "imagecmp" structure
   
   MTPcompression_UShort_MemSetup( nrows, ncols, totalframes, cameranumber, ndecim, interleave, framerateHz, camerafolder, imagecmp );
   
   imagecmp->firstframe = firstframe;


   //======== Read the array information according to the CAMS file format specified

   fread( imagecmp->maxpixel_ptr, sizeof(unsigned short), nrows*ncols, ReadFile ); 
   fread( imagecmp->maxframe_ptr, sizeof(unsigned short), nrows*ncols, ReadFile ); 
   fread( imagecmp->avepixel_ptr, sizeof(unsigned short), nrows*ncols, ReadFile ); 
   fread( imagecmp->stdpixel_ptr, sizeof(unsigned short), nrows*ncols, ReadFile );                      


   //======== Close the HH file

   fclose( ReadFile );


   //======== Compute the global mean, sigma and populate the pointer offsets 
   //           of the maxpixel values specific to each frame number.

   MTPcompression_UShort_GlobalMean( imagecmp );          

   MTPcompression_UShort_GlobalSigma( imagecmp );          

   MTPcompression_UShort_PrepBuild( imagecmp );

}


//#######################################################################################################
//                           MTPcompression_UShort_PrepBuild
//
//  Populates the pointer offsets of the maxpixel values specific to a given frame number.
//  This preps the build functions for faster frame-by-frame image reconstruction from the 
//  HH file compressed data contents. 
//      mpfrmcnt = number of times a given maxframe number appears in the image block
//      mpcumcnt = specific frame# starting index in the maxpixel pointer offset array
//      mpoffset = rcindex pointer offset into the maxpixel array
//  This function is always called by MTPcompression_UShort_FileRead.
//
//#######################################################################################################

void   MTPcompression_UShort_PrepBuild( struct MTP_US_compressor *imagecmp )
{
long            k, kf, rcindex;
unsigned short *maxframe_ptr;


   //------ Compute the histogram of frame numbers in the maxframe array

   for( k=0; k<imagecmp->totalframes; k++ )  imagecmp->framecnt_ptr[k] = 0;


   maxframe_ptr = imagecmp->maxframe_ptr;

   for( rcindex=0; rcindex<imagecmp->nrows*imagecmp->ncols; rcindex++ )  {

	   k = (long)*maxframe_ptr++;

       imagecmp->framecnt_ptr[k] += 1;  //...increment the frame count for this frame number
   }


   //------ Compute the histogram's cumulative count

   imagecmp->mpcumcnt_ptr[0] = 0;

   for( k=1; k<imagecmp->totalframes; k++ )  imagecmp->mpcumcnt_ptr[k] = imagecmp->mpcumcnt_ptr[k-1] 
	                                                                   + imagecmp->framecnt_ptr[k-1];


   //------ Determine the pointer offsets for each specific frame number and group them
																	   
   for( k=0; k<imagecmp->totalframes; k++ )  imagecmp->framecnt_ptr[k] = 0;


   maxframe_ptr = imagecmp->maxframe_ptr;

   for( rcindex=0; rcindex<imagecmp->nrows*imagecmp->ncols; rcindex++ )  {

	   k = (long)*maxframe_ptr++;

	   kf = imagecmp->mpcumcnt_ptr[k] + imagecmp->framecnt_ptr[k];

       imagecmp->mpoffset_ptr[kf] = rcindex;

	   imagecmp->framecnt_ptr[k] += 1; //... this will eventually restore frame count

   }


}

//#######################################################################################################
//                          MTPcompression_UShort_RowColVal
//
//  Fills a rowcolval structure with the maxpixel minus mean values associated with a user   
//  specified frame number. Assumes MTPcompression_UShort_PrepBuild has been called, typically
//  done on an HH file read. Upon each MTPcompression_UShort_RowColVal call, the rowcolval
//  structure's pointer is checked for a NULL value. If not NULL, then the program cleans up
//  memory and reallocates a replacement rowcolval structure. The function returns the number 
//  of maxpixels for the given frame number as "Nrcval" in the imagecmp structure as well as
//  the pointer to the rowcolval structure containing row, col, and maxpixel-mean value.
//
//  For a program to later retrieve the row, col, or max-mean value respectively use:
//       imagecmp->rcval_ptr[k]->row
//       imagecmp->rcval_ptr[k]->col
//       imagecmp->rcval_ptr[k]->val       for k=0 to imagecmp->Nrcmax-1
//
//#######################################################################################################

void   MTPcompression_UShort_RowColVal( long fileframenumber, struct MTP_US_compressor *imagecmp )
{
long            kf, kflo, kfhi, rcindex;
unsigned short *maxpixel_ptr;
unsigned short *avepixel_ptr;

struct  rowcolval_US  *rcval_ptr;


     //------ Free memory rowcolmax structure if its pointer is not NULL, to ensure that
     //         the last call's memory allocation is freed.

     if( imagecmp->rcval_ptr != NULL )  free( imagecmp->rcval_ptr );


	 //------ Check for valid frame number

	 imagecmp->Nrcval = 0;

	 if( fileframenumber <  0                     )  return;
     if( fileframenumber >= imagecmp->totalframes )  return;


     //------ Determine the number of maxpixels to extract

     kflo = imagecmp->mpcumcnt_ptr[fileframenumber];

     if( fileframenumber < imagecmp->totalframes - 1 )  kfhi = (long)imagecmp->mpcumcnt_ptr[fileframenumber+1];
     else                                               kfhi = imagecmp->nrows * imagecmp->ncols;

     imagecmp->Nrcval = kfhi - kflo;


     //------ Allocate memory based on number of maxpixels for this frame number

     imagecmp->rcval_ptr = (struct rowcolval_US*) malloc( imagecmp->Nrcval * sizeof(struct rowcolval_US) );

     if( imagecmp->rcval_ptr == NULL )  {
         printf(" ERROR:  Memory not allocated in MTPcompression_UShort_RowColVal\n");
         Delay_msec(10000);
	     exit(1);
     }


	  //------ Extract only those pixel locations where the "fileframenumber" matches the 
	  //         maxframe number. The pixel locations are given by pointer offsets previously
	  //         calculated in the function MTPcompression_UShort_PrepBuild.

	  maxpixel_ptr = imagecmp->maxpixel_ptr;
	  avepixel_ptr = imagecmp->avepixel_ptr;
	  rcval_ptr    = imagecmp->rcval_ptr;

      for( kf=kflo; kf<kfhi; kf++ )  {

           rcindex = imagecmp->mpoffset_ptr[kf];

           rcval_ptr->row = (unsigned short)( rcindex / imagecmp->ncols );
           rcval_ptr->col = (unsigned short)( rcindex % imagecmp->ncols );
           rcval_ptr->val = maxpixel_ptr[rcindex] - avepixel_ptr[rcindex];

		   rcval_ptr++;
      }


}


//#######################################################################################################
//                            MTPcompression_UShort_FrameBuild
//
//  Reconstructs a full image frame given a specific frame number desired. The frame consists
//  of the mean pixel image overwritten with maxpixel values only where the maxframe matches 
//  the "fileframenumber". Call MTPcompression_UShort_PrepBuild once BEFORE 
//  MTPcompression_UShort_FrameBuild is used but note that if the function 
//  MTPcompression_UShort_FileRead was called to obtain the compressed data block, then
//  the function MTPcompression_UShort_PrepBuild has already been called. Final output
//  image is placed in imagecmp->imgframe_ptr.
//
//#######################################################################################################

void   MTPcompression_UShort_FrameBuild( long fileframenumber, struct MTP_US_compressor *imagecmp )
{
long            kf, kflo, kfhi, rcindex;
unsigned short *maxpixel_ptr;
unsigned short *imgframe_ptr;


      //------ First fill with the mean image and check for out of bounds frame number request

	  memcpy( imagecmp->imgframe_ptr, imagecmp->avepixel_ptr, sizeof(unsigned short) * imagecmp->nrows * imagecmp->ncols ); 

      if( fileframenumber <  0                     )  return;
      if( fileframenumber >= imagecmp->totalframes )  return;


	  //------ Determine the range of maxpixels that need to get mapped onto the image frame

	  kflo = (long)imagecmp->mpcumcnt_ptr[fileframenumber];

	  if( fileframenumber < imagecmp->totalframes - 1 )  kfhi = (long)imagecmp->mpcumcnt_ptr[fileframenumber+1];
	  else                                               kfhi = imagecmp->nrows * imagecmp->ncols;


	  //------ Overwrite those pixel locations with maxpixel values where the "fileframenumber" 
	  //         matches the maxframe number. The pixel locations are given by pointer offsets 
	  //         previously calculated in the function MTPcompression_UShort_PrepBuild.

	  maxpixel_ptr = imagecmp->maxpixel_ptr;
      imgframe_ptr = imagecmp->imgframe_ptr;

      for( kf=kflo; kf<kfhi; kf++ )  {

           rcindex = imagecmp->mpoffset_ptr[kf];

           imgframe_ptr[rcindex] = maxpixel_ptr[rcindex];
      }
           
}


//#######################################################################################################
//                       MTPcompression_UShort_GlobalMean
//
//  Assigns the median of the block mean to imagecmp->global_ave via gray level histogram
//#######################################################################################################

void   MTPcompression_UShort_GlobalMean( struct MTP_US_compressor *imagecmp )
{
long            *kount, kgray, kpixel, ksum, khalf;
unsigned short *avepixel_ptr;


      //------ Allocate memory for histogram

      kount = (long*) malloc( 65536L * sizeof(long) );
      
	  if( kount == NULL )  {
         printf(" ERROR:  Memory not allocated in MTPcompression_UShort_GlobalMean\n");
         Delay_msec(10000);
	     exit(1);
      }


      //------ Build histogram of mean values across image block

      for( kgray=0; kgray<65536L; kgray++ )  kount[kgray] = 0;

      avepixel_ptr = imagecmp->avepixel_ptr;
      
      for( kpixel=0; kpixel<imagecmp->nrows * imagecmp->ncols; kpixel++ )  {
  
           kgray = (long)*avepixel_ptr++;
                
           kount[kgray] += 1;           
      }
    
     
      //------ Find the median which equals the middle value in the cumulative histogram

	  khalf = imagecmp->nrows * imagecmp->ncols / 2;
      
      ksum = 0;
      
      for( kgray=0; kgray<65536L; kgray++ )  {

           ksum += kount[kgray];
           
           if( ksum > khalf )  break;
      }  

      imagecmp->global_ave = (unsigned short)kgray;
          
	  free( kount );

}


//#######################################################################################################
//                        MTPcompression_UShort_GlobalSigma
//
//  Assigns the median of the block standard deviation to imagecmp->global_std via histogram
//#######################################################################################################

void   MTPcompression_UShort_GlobalSigma( struct MTP_US_compressor *imagecmp )
{
long           *kount, kgray, kpixel, ksum, khalf;
unsigned short *stdpixel_ptr;
 

      //------ Allocate memory for histogram

      kount = (long*) malloc( 65536L * sizeof(long) );
      
	  if( kount == NULL )  {
         printf(" ERROR:  Memory not allocated in MTPcompression_UShort_GlobalSigma\n");
         Delay_msec(10000);
	     exit(1);
      }


	  //------ Build histogram of sigma values across image block      
      
      for( kgray=0; kgray<65536L; kgray++ )  kount[kgray] = 0;

      stdpixel_ptr = imagecmp->stdpixel_ptr;
  
      for( kpixel=0; kpixel<imagecmp->nrows * imagecmp->ncols; kpixel++ )  {
  
           kgray = (long)*stdpixel_ptr++;
                
           kount[kgray] += 1;           
      }

      
      //------ Find the median of sigma = middle value in the cumulative histogram
      
	  khalf = imagecmp->nrows * imagecmp->ncols / 2;
      
      ksum = 0;
      
      for( kgray=0; kgray<65536L; kgray++ )  {

           ksum += kount[kgray];
           
           if( ksum > khalf )  break;
      }  
      
      imagecmp->global_std = (unsigned short)kgray;

	  free( kount );

}


//#######################################################################################################
//                            MTPcompression_UShort_Flatten
//
//  Flat fielding of previously built "imgframe" comprised of mean removal, division by the 
//  standard deviation, and scaling up by 128. The sign preserving output uses the short
//  data type. Either a 0 or 1 is added to the entire frame on each call to toggle a small
//  noise contribution.
//
//  Call MTPcompression_UShort_FrameBuild before calling this function to populate the
//  array imagecmp->imgframe_ptr.
//
//#######################################################################################################

void   MTPcompression_UShort_Flatten( short *flattened_ptr, struct MTP_US_compressor *imagecmp )
{ 
long            unitybias, rcindex;
unsigned short *imgframe_ptr;
unsigned short *avepixel_ptr;
unsigned short *stdpixel_ptr;


      //------ Set pointers to the image frame, mean, and standard deviation

      imgframe_ptr = imagecmp->imgframe_ptr;
      avepixel_ptr = imagecmp->avepixel_ptr;
      stdpixel_ptr = imagecmp->stdpixel_ptr;
      
      
      //------ Add either zero or one to this image for well behaved noise filter
            
      imagecmp->randcount = (imagecmp->randcount + 1) % 65536L;
      
      unitybias = (short)( imagecmp->randomN_ptr[imagecmp->randcount] % 2 );
      
      
      //------ Mean remove and flat field each pixel = 128 * ( pixel - mean ) / sigma
        
      for( rcindex=0; rcindex<imagecmp->nrows*imagecmp->ncols; rcindex++ )  {
  
		  flattened_ptr[rcindex] = (short)( unitybias + ( labs( (long)(imgframe_ptr[rcindex] - avepixel_ptr[rcindex]) ) << 7 ) / (long)stdpixel_ptr[rcindex] );
                           
      }
      
}

//#######################################################################################################
//                              MTPcompression_UShort_BuildFlatten
//
//  Flat field image processing by first reconstructing the image frame and then performing  
//  mean removal, division by the standard deviation, and scaling up by 128. The sign  
//  preserving output is of the short data type. Either a 0 or 1 is added to the entire frame 
//  on each call to toggle a small noise contribution. Call MTPcompression_UShort_PrepBuild once before
//  MTPcompression_UShort_FrameBuild is used. Note that if the function MTPcompression_UShort_FileRead was called
//  to get the compressed data block, then the function MTPcompression_UShort_PrepBuild has already been
//  called. Currently only puts in the maxpixel values (no aftermax values).
//
//#######################################################################################################

void   MTPcompression_UShort_BuildFlatten( short *flattened_ptr, long fileframenumber, struct MTP_US_compressor *imagecmp )
{
long            rcindex, kf, kflo, kfhi;
unsigned short *maxpixel_ptr;
unsigned short *avepixel_ptr;
unsigned short *stdpixel_ptr;
short           unitybias;


      //------ Add either zero or one to this image for well behaved noise filter
            
      imagecmp->randcount = (imagecmp->randcount + 1) % 65536L;
      
      unitybias = (short)( imagecmp->randomN_ptr[imagecmp->randcount] % 2 );

	  for( rcindex=0; rcindex<imagecmp->nrows*imagecmp->ncols; rcindex++ )  flattened_ptr[rcindex] = unitybias;

      
      //------ Check if frame number is out of bounds

      if( fileframenumber <  0                     ) return;
      if( fileframenumber >= imagecmp->totalframes ) return;


	  //------ Determine the range of maxpixels that need to get mapped onto the image frame

	  kflo = imagecmp->mpcumcnt_ptr[fileframenumber];

	  if( fileframenumber < imagecmp->totalframes - 1 )  kfhi = (long)imagecmp->mpcumcnt_ptr[fileframenumber+1];
	  else                                               kfhi = imagecmp->nrows * imagecmp->ncols;


	  //------ Set pointers to the image frame, mean, and standard deviation

	  maxpixel_ptr = imagecmp->maxpixel_ptr;
	  avepixel_ptr = imagecmp->avepixel_ptr;
	  stdpixel_ptr = imagecmp->stdpixel_ptr;


      //..... Flat field each pixel = 128 * ( pixel - mean ) / sigma 
	  //        only for the specified frame number (otherwise set to unitybias)
        
      for( kf=kflo; kf<kfhi; kf++ )  {

           rcindex = imagecmp->mpoffset_ptr[kf];

           flattened_ptr[rcindex] += (short)( ( labs( (long)(maxpixel_ptr[rcindex] - avepixel_ptr[rcindex]) ) << 7 ) / (long)stdpixel_ptr[rcindex] );
      }
      
}


//#######################################################################################################
//                                 MTPcompression_UShort_MeanRemoval
//
//  Remove the mean from the image frame array and preserve the sign using a short on output.
//       demean = imgframe - avepixel
//#######################################################################################################

void   MTPcompression_UShort_MeanRemoval( short *demean_ptr, struct MTP_US_compressor *imagecmp )
{
long            kthrow, kthcol;
unsigned short *imgframe_ptr;
unsigned short *avepixel_ptr;


      //------ Set pointers to the image frame and mean

      imgframe_ptr = imagecmp->imgframe_ptr;
      avepixel_ptr = imagecmp->avepixel_ptr;
            
      
      //------ Remove mean for each pixel
        
      for( kthrow=0; kthrow<imagecmp->nrows; kthrow++ )  {
  
           for( kthcol=0; kthcol<imagecmp->ncols; kthcol++ )  {
                           
                *demean_ptr++ = (short)( (long)*imgframe_ptr++ - (long)*avepixel_ptr++ );
                                                 
           }
      }
      
}



//#######################################################################################################
//                      MTPcompression_UShort_MemCleanup
//
//  Frees all allocated memory of the compression structure specified
//#######################################################################################################

void   MTPcompression_UShort_MemCleanup( struct MTP_US_compressor *imagecmp )
{

   if( imagecmp->maxpixel_ptr != NULL )  free( imagecmp->maxpixel_ptr );
   if( imagecmp->maxframe_ptr != NULL )  free( imagecmp->maxframe_ptr );
   if( imagecmp->avepixel_ptr != NULL )  free( imagecmp->avepixel_ptr );
   if( imagecmp->stdpixel_ptr != NULL )  free( imagecmp->stdpixel_ptr );
   if( imagecmp->imgframe_ptr != NULL )  free( imagecmp->imgframe_ptr );

   if( imagecmp->pixel_ptr    != NULL )  free( imagecmp->pixel_ptr    );
   if( imagecmp->rcval_ptr    != NULL )  free( imagecmp->rcval_ptr    );

   if( imagecmp->mpoffset_ptr != NULL )  free( imagecmp->mpoffset_ptr );
   if( imagecmp->mpcumcnt_ptr != NULL )  free( imagecmp->mpcumcnt_ptr );
   if( imagecmp->framecnt_ptr != NULL )  free( imagecmp->framecnt_ptr );

   if( imagecmp->randomN_ptr  != NULL )  free( imagecmp->randomN_ptr  );
   if( imagecmp->squared_ptr  != NULL )  free( imagecmp->squared_ptr  );

   MTPcompression_UShort_NullPointers( imagecmp );
   

}


//#######################################################################################################
//                                MTPcompression_UShort_NullPointers
//
//  Ensures all array and vector data pointers are set to NULL. Recommend calling this at
//  program startup to clear all the compressor structure's pointers
//#######################################################################################################

void   MTPcompression_UShort_NullPointers( struct MTP_US_compressor *imagecmp )
{

   imagecmp->maxpixel_ptr = NULL;
   imagecmp->maxframe_ptr = NULL;
   imagecmp->avepixel_ptr = NULL;
   imagecmp->stdpixel_ptr = NULL;
   imagecmp->imgframe_ptr = NULL;
   imagecmp->pixel_ptr    = NULL;
   imagecmp->rcval_ptr    = NULL;
   imagecmp->mpoffset_ptr = NULL;
   imagecmp->mpcumcnt_ptr = NULL;
   imagecmp->framecnt_ptr = NULL;
   imagecmp->randomN_ptr  = NULL;
   imagecmp->squared_ptr  = NULL;
    
} 

//#######################################################################################################
