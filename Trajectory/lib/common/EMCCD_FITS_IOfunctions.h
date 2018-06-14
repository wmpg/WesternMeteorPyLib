//#########################################################################
//#########################################################################
//
//  EMCCD I/O Functions for FITS files (Windows Only)
//
//#########################################################################
//#########################################################################

#ifdef _WIN32 /************* WINDOWS ******************************/

#pragma warning(disable: 4996)  // disable warning on strcpy, fopen, ...

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fitsio.h"
#include "longnam.h"


//*************************************************************************
//******************  FITS File Interface Functions  **********************
//*************************************************************************

//------------------------------- FITS file status
#define  FILEOK            0
#define  WRONGIMAGETYPE    1
#define  NOT2DIMIMAGE      2
#define  FILENOTFOUND      3
#define  MEMNOTALLOCATED   4



//=========================================================================
//                  FitsFileInfoMemAllocation
//-------------------------------------------------------------------------
//
// Purpose:   Function to inquire into existance of an image data file and 
//            retrieve header information for data set size. Assumes FITS 
//            format and checks for two-dimensional imagery. Also allocates
//            memory for an unsigned short array of dimension given in the
//            header information for later infill using FitsImageRead.
//
// Inputs:    pathname        Pathname of image data file
//
// Outputs:  *totalrows       Number of rows
//           *totalcols       Number of columns
//          **image_ptr_ptr   Allocated array of unsigned shorts (ptr to ptr)
//            status          Return argument - see mnemonics above 
//
//
//=========================================================================

int   FitsFileInfoMemAllocation( char *pathname, int *totalrows, int *totalcols, unsigned short **image_ptr_ptr )
{
  fitsfile *fptr;    // FITS file pointer
  int  status;       // CFITSIO status value MUST be initialized to zero
  int  openstatus;   // CFITSIO status value
  int  hdutype;      // Data type: image or table
  int  naxis;        // Number of dimensions in the image
  long naxes[2];     // column and row size of image
      
      
     
      //----------------- Open the image file to obtain the header info
      //                  CFITSIO status value MUST be initialized to zero
      
      status = 0;        
      if( !fits_open_file( &fptr, pathname, READONLY, &status ) )  {

		  openstatus = FILEOK;

          status = 0;
          if( fits_get_hdu_type (fptr, &hdutype, &status )  ||  hdutype != IMAGE_HDU ) { 
              fprintf( stdout, " ERROR ===>  This program only works on images, NOT tables in FitsFileInfoMemAllocation\n");
              fprintf( stderr, " ERROR ===>  This program only works on images, NOT tables in FitsFileInfoMemAllocation\n");
              openstatus = WRONGIMAGETYPE;
          }

          status = 0;           
          fits_get_img_dim( fptr, &naxis, &status );

          if( status || naxis != 2 ) { 
              fprintf( stdout, " ERROR ===>  NAXIS = %d. Only 2-D images are supported in FitsFileInfoMemAllocation\n", naxis );
              fprintf( stderr, " ERROR ===>  NAXIS = %d. Only 2-D images are supported in FitsFileInfoMemAllocation\n", naxis );
              openstatus = NOT2DIMIMAGE;
          }
           
          //-------------------- save number of rows and columns
           
          status = 0;          
          fits_get_img_size( fptr, 2, naxes, &status );
          
          *totalcols = (int)naxes[0];
          *totalrows = (int)naxes[1];
          
          status = 0;          
          fits_close_file( fptr, &status );

      }
      else  {
          fprintf( stdout, " ERROR ===>  Cannot open image file %s in FitsFileInfoMemAllocation\n", pathname );
          fprintf( stderr, " ERROR ===>  Cannot open image file %s in FitsFileInfoMemAllocation\n", pathname );
		  Delay_msec(10000);
          openstatus = FILENOTFOUND;
      }


	  if( openstatus == FILEOK )  {

		  *image_ptr_ptr = (unsigned short*) malloc( *totalrows * *totalcols * sizeof( unsigned short ) );

          if( image_ptr_ptr == NULL )  {
              fprintf( stdout, " ERROR ===>  Cannot allocate array memory in FitsFileInfoMemAllocation\n" );
              fprintf( stderr, " ERROR ===>  Cannot allocate array memory in FitsFileInfoMemAllocation\n" );
              openstatus = MEMNOTALLOCATED;
          }
	  }


	  return( openstatus );      
            
}

//=========================================================================
//                       FitsImageDimensions
//-------------------------------------------------------------------------
//
// Purpose:   Function to inquire into existance of an image data file and 
//            retrieve header information for data set size. Assumes FITS 
//            format and checks for two-dimensional imagery. 
//
// Inputs:    pathname        Pathname of image data file
//
// Outputs:  *totalrows       Number of rows
//           *totalcols       Number of columns
//            status          Return argument - see mnemonics above 
//
//
//=========================================================================

int   FitsImageDimensions( char *pathname, int *totalrows, int *totalcols )
{
  fitsfile *fptr;  // FITS file pointer
  int  status;     // CFITSIO status value MUST be initialized to zero
  int  openstatus; // CFITSIO status value 
  int  hdutype;    // Data type: image or table
  int  naxis;      // Number of dimensions in the image
  long naxes[2];   // column and row size of image
      
      
     
      //----------------- Open the image file to obtain the header info
      //                  CFITSIO status value MUST be initialized to zero
      
      status = 0;        
      if( !fits_open_file( &fptr, pathname, READONLY, &status ) )  {

          openstatus = FILEOK;
           
          status = 0;
          if( fits_get_hdu_type (fptr, &hdutype, &status )  ||  hdutype != IMAGE_HDU ) { 
              fprintf( stdout, " ERROR ===>  This program only works on images, NOT tables in FitsImageDimensions\n");
              fprintf( stderr, " ERROR ===>  This program only works on images, NOT tables in FitsImageDimensions\n");
              openstatus = WRONGIMAGETYPE;
          }

          status = 0;
          fits_get_img_dim( fptr, &naxis, &status );

          if( status || naxis != 2 ) { 
              fprintf( stdout, " ERROR ===>  NAXIS = %d. Only 2-D images are supported in FitsImageDimensions\n", naxis );
              fprintf( stderr, " ERROR ===>  NAXIS = %d. Only 2-D images are supported in FitsImageDimensions\n", naxis );
              openstatus = NOT2DIMIMAGE;
          }
           
          //-------------------- save number of rows and columns
           
          status = 0;
          fits_get_img_size( fptr, 2, naxes, &status );
          
          *totalcols = (int)naxes[0];
          *totalrows = (int)naxes[1];
          
          status = 0;
          fits_close_file( fptr, &status );

      }
      else  {
          fprintf( stdout, " ERROR ===>  Cannot open image file %s in FitsImageDimensions\n", pathname );
          fprintf( stderr, " ERROR ===>  Cannot open image file %s in FitsImageDimensions\n", pathname );
		  Delay_msec(60000);
          openstatus = FILENOTFOUND;
      }

	  return( openstatus );      
            
}

//*************************************************************************

void   FitsFreeMemory( unsigned short *image_ptr )
{
	free( image_ptr );
}


//=========================================================================
//                         FitsImageRead
//-------------------------------------------------------------------------
//
// Purpose:   Function to read a user defined image segment from the FITS
//            data file. Assumes unsigned short data stored in FITS file.
//
// Inputs:    pathname        Pathname of image data file
//            totalrows       Number of rows
//            totalcols       Number of columns
//
// Outputs:  *image_ptr       Pointer to unsigned short array pre-allocated to correct size
//
//=========================================================================

void   FitsImageRead( char *pathname, int totalrows, int totalcols, unsigned short *image_ptr )
{
  fitsfile *fptr;   // FITS file pointer
  int   status;     // CFITSIO status value MUST be initialized to zero     
  long  row, startpixel[2];
      

      //----------------- Open the image file and read the data
      //                  CFITSIO status value MUST be initialized to zero
           
      startpixel[0] = 1;  // starting column to read in the file
                          // Note that FITS starts with row=1,col=1

      status = 0;
      if( !fits_open_file( &fptr, pathname, READONLY, &status ) ) {
          
          for( row=1; row<=totalrows; row++ ) {

               startpixel[1] = row;  //... row to read in the FITS file
          
			   status = 0;
               fits_read_pix( fptr, TUSHORT, startpixel, (long)totalcols, 0, image_ptr, 0, &status );
                              
			   image_ptr += totalcols;
                                             
          } 

		  status = 0;
          fits_close_file( fptr, &status );
      }
      
      else  {
          fprintf( stdout, " ERROR ===>  Cannot open image file %s in FitsImageRead\n", pathname );
          fprintf( stderr, " ERROR ===>  Cannot open image file %s in FitsImageRead\n", pathname );
	      Delay_msec(60000);
	      exit(1);
      } 

  
}

//=========================================================================
//                         SetParametersFromFITSfile
//-------------------------------------------------------------------------
//
// Purpose:   Function to read some info from the FITS header to fill the
//            parameter structure.
//
// Inputs:    imagery_pathname   Pathname of image data file
//
// Outputs:  *params             Pointer to the EMCCDparameters structure
//               nrows              Number of rows
//               ncols              Number of columns
//               cameranumber     Camera number
//               sample_time_sec  Sample time spacing in seconds
//
//=========================================================================

void   SetParametersFromFITSfile( char* imagery_pathname, struct EMCCDparameters *params )
{
int    status, nrows, ncols, cameranumber;
char   folder_path[256];
char   filename[256];


    //======== Get the image dimensions

    status = FitsImageDimensions( imagery_pathname, &nrows, &ncols );

	if( status != 0 )  {
		fprintf( stdout, " ERROR ===> Function error %d FitsImageDimensions\n", status );
		fprintf( stderr, " ERROR ===> Function error %d FitsImageDimensions\n", status );
		Delay_msec(30000);
		exit(1);
	}


	//======== Get the camera number by parsing the filename

	SplitPathname( imagery_pathname, folder_path, filename );

	sscanf( filename, "EMCCD_%d", &cameranumber );


	//======== Assign values to the parameters

	params->nrows           = nrows;
	params->ncols           = ncols;

	if(      cameranumber == 3 )  params->cameranumber = (256 * 2)  +  64 + 4;  // 256 * site# + ascii code for cam# (1=A, 2=B, ...)
	else if( cameranumber == 4 )  params->cameranumber = (256 * 2)  +  64 + 6;
	else                          params->cameranumber = (256 * 2)  +  64 + 26; 

	// params->sample_time_sec = ?     // override value if available from FITS

}

//========================================================================


#endif   /************* WINDOWS ******************************/