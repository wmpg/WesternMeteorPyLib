//#########################################################################
//#########################################################################
//
//  EMCCD I/O Functions for UWO VID files (Windows or Linux System)
//
//#########################################################################
//#########################################################################

#ifdef _WIN32 /************* WINDOWS ******************************/

//========================================================================

#include <windows.h>
#include <stdio.h>


struct vidheader {              //==== Header from the FIRST frame only

	unsigned long  magic;		// { 'V', 'I', 'D', 0 }
	unsigned long  seqlen;		// Bytes for a single image
	unsigned long  headlen;		// Bytes for the embedded header
	unsigned long  flags;		// flags
	unsigned long  seq;			// Sequence number

	long           ts, tu;	    // UNIX time
	short          num;			// station number
	short          wid, ht;		// Image width and height in pixels
	short          depth;		// bit-depth

	unsigned short hx, hy;		// hwctrl pointing for centre of frame
	unsigned short str;			// stream number
	unsigned short reserved0;	// reserved
	unsigned long  expose;		// exposure time in milliseconds
	unsigned long  reserved2;	// reserved
	char           text[64];	// Text:  e.g. "CMOR_25mm_Fujinon_f/0.85"

	FILE          *vidfile;     // File pointer
};



int   VID_File_Open( char *pathname, struct vidheader *vidinfo );

int   VID_Frame_Read(struct vidheader *vidinfo, int kthframe, unsigned short *image_ptr, double *unixseconds, long *seqnum);

int   VID_File_Close(struct vidheader *vidinfo);


//=========================================================================
//                          VID_File_Open
//-------------------------------------------------------------------------
//
// Purpose:   Function to open a .vid file and read the first frame's
//            header content and store it in the vidheader structure.
//            Leaves file open for subsequent image reads.
//
// Inputs:    pathname        Pathname of the multi-frame imagery data file
//
// Outputs:   vidinfo         Structure vidheader of FIRST frame only
//            status          Return argument 0 = success, -1 = error "errno"
//
//
//=========================================================================

int   VID_File_Open(char *pathname, struct vidheader *vidinfo)
{
	int     openstatus;


	//----------------- Open the image file and obtain the header info

	openstatus = 0;
	vidinfo->vidfile = fopen(pathname, "rb");
	if ( vidinfo->vidfile == NULL ) {
		fprintf(stdout, " ERROR ===>  Cannot open image file %s in VID_File_Open\n", pathname);
		fprintf(stderr, " ERROR ===>  Cannot open image file %s in VID_File_Open\n", pathname);
		Sleep(10000);
		openstatus = -1;
	}

	//----------------- Read the hedaer lines

	fread(&vidinfo->magic,      4,  1, vidinfo->vidfile);
	fread(&vidinfo->seqlen,     4,  1, vidinfo->vidfile);
	fread(&vidinfo->headlen,    4,  1, vidinfo->vidfile);
	fread(&vidinfo->flags,      4,  1, vidinfo->vidfile);
	fread(&vidinfo->seq,        4,  1, vidinfo->vidfile);
	fread(&vidinfo->ts,         4,  1, vidinfo->vidfile);
	fread(&vidinfo->tu,         4,  1, vidinfo->vidfile);
	fread(&vidinfo->num,        2,  1, vidinfo->vidfile);
	fread(&vidinfo->wid,        2,  1, vidinfo->vidfile);
	fread(&vidinfo->ht,         2,  1, vidinfo->vidfile);
	fread(&vidinfo->depth,      2,  1, vidinfo->vidfile);
	fread(&vidinfo->hx,         2,  1, vidinfo->vidfile);
	fread(&vidinfo->hy,         2,  1, vidinfo->vidfile);
	fread(&vidinfo->str,        2,  1, vidinfo->vidfile);
	fread(&vidinfo->reserved0,  2,  1, vidinfo->vidfile);
	fread(&vidinfo->expose,     4,  1, vidinfo->vidfile);
	fread(&vidinfo->reserved2,  4,  1, vidinfo->vidfile);
	fread(&vidinfo->text,       1, 64, vidinfo->vidfile);

	/*
	printf(" header %d\n", vidinfo->magic);
	printf(" seqlen %d\n", vidinfo->seqlen);
	printf(" length %d\n", vidinfo->headlen);
	printf(" header %d\n", vidinfo->flags);
	printf(" seqnum %d\n", vidinfo->seq);
	printf(" header %d\n", vidinfo->ts);
	printf(" header %d\n", vidinfo->tu);
	printf(" header %d\n", vidinfo->num);
	printf(" width  %d\n", vidinfo->wid);
	printf(" height %d\n", vidinfo->ht);
	printf(" depth  %d\n", vidinfo->depth);
	printf(" header %d\n", vidinfo->hx);
	printf(" header %d\n", vidinfo->hy);
	printf(" header %d\n", vidinfo->str);
	printf(" header %d\n", vidinfo->reserved0);
	printf(" header %d\n", vidinfo->expose);
	printf(" header %d\n", vidinfo->reserved2);
	printf(" string %s\n", vidinfo->text);
	printf(" -------------------------------\n");
	Sleep(10000);
	*/

	fseek( vidinfo->vidfile, 0, SEEK_SET ); //... move back to beginning of file

	return(openstatus);

}

//=========================================================================
//                             VID_Frame_Read
//-------------------------------------------------------------------------
//
// Purpose:   Function to read the next frame in the .vid file or a user
//            specified frame starting with 0. Use -1 for sequential reads
//
// Inputs:    vidinfo         Structure vidheader of file metadata for the 1st frame
//            kthframe        Frame to read starting at 0, 1, 2, ...
//                               Use -1 for sequential reads
//           *image_ptr       Pointer to the user's unsigned short imagery array
//
// Outputs:   unixseconds     Time of THIS frame in seconds since UNIX start Jan 1, 1970  0:00 UT
//            seqnumber       Image sequence number of THIS frame
//            status          Return value #frames read (should be 1 as configured here)
//                                0 for EOF,  -1 for error
//
//=========================================================================

int    VID_Frame_Read(struct vidheader *vidinfo, int kthframe, unsigned short *image_ptr, double *unixseconds, long *seqnum)
{
	long       kpixel, npixels, nread, ts, tu;
	long            *header;
	unsigned char   *pixmap8;
	unsigned short  *pixmap16;


	//======== Allocate memory

	npixels = vidinfo->wid * vidinfo->ht;

	if (vidinfo->depth <= 8)  pixmap8  = (unsigned char* )malloc(npixels * sizeof(unsigned char ));
	else                      pixmap16 = (unsigned short*)malloc(npixels * sizeof(unsigned short));

	if (vidinfo->depth <= 8)  header = (long*)pixmap8;
	else                      header = (long*)pixmap16;


	//======== Position the file pointer to the desired frame (no reposition if sequential)

	if( kthframe >= 0 )  {

		if (vidinfo->depth <= 8)  fseek( vidinfo->vidfile, 1 * npixels * kthframe, SEEK_SET );
		else                      fseek( vidinfo->vidfile, 2 * npixels * kthframe, SEEK_SET );

	}


	//======== Read the next image frame and embedded header "stamp" from the file

	if (feof(vidinfo->vidfile)) return(0);   //  EOF = 0

	if (vidinfo->depth <= 8)  nread = (long)fread(pixmap8,  1, npixels, vidinfo->vidfile);
	else                      nread = (long)fread(pixmap16, 2, npixels, vidinfo->vidfile);

	if (nread != npixels)  return(-1);  //  ERROR = -1


	//======== Get the frame's sequence number (frame#) and frame time in seconds since unix start

	*seqnum = header[4];

	ts      = header[5];

	tu      = header[6];

	*unixseconds = (double)ts + (double)tu / 1000000.0;

    //printf("seqnum  %d    unixseconds %lf\n", *seqnum, *unixseconds );


	//======== Clear the header "stamp" that is embedded in the first row of the image

	memset( header, 0, vidinfo->headlen );


	//======== Assign the appropriate pointer and copy image to a user array based on bit depth

	if (vidinfo->depth <= 8)

		for (kpixel = 0; kpixel<npixels; kpixel++)  image_ptr[kpixel] = (unsigned short)pixmap8[kpixel];

	else

		for (kpixel = 0; kpixel<npixels; kpixel++)  image_ptr[kpixel] = (unsigned short)pixmap16[kpixel];


	//---------- free memory

	if (vidinfo->depth <= 8)  free(pixmap8);
	else                      free(pixmap16);


	return(1);

}


//=========================================================================
//                          VID_File_Close
//-------------------------------------------------------------------------
//
// Purpose:   Function to close the .vid file
//
// Inputs:    pathname        Pathname of the multi-frame imagery data file
//
// Outputs:   status          Return argument 0 closed, -1 if vidfile was NULL
//
//
//=========================================================================

int   VID_File_Close(struct vidheader *vidinfo)
{
	int   closestatus;


	//----------------- Close the image file

	closestatus = fclose( vidinfo->vidfile );

	vidinfo->vidfile = NULL;

	return( closestatus );

}

//=========================================================================

#endif


//###########################################################################
//###########################################################################
//###########################################################################


#ifdef __linux  /********************  LINUX  ******************************/

//=========================================================================

int   VID_File_Open( char *pathname, struct dumpvid_s **vidinfo );

int   VID_Frame_Read( struct dumpvid_s *vidinfo, unsigned short *image_ptr, double *unixseconds, long *seqnum );

int   VID_File_Close( struct dumpvid_s *vidinfo );

void  SetParametersFromVIDfile( struct dumpvid_s *vidinfo, struct EMCCDparameters *params, double *jdtstart );

void  GetParametersFromVIDfile( struct dumpvid_s *vidinfo,
	                             int *nrows, int *ncols, int *cameranumber,
								 double *sample_time_sec, double *jdtstart );


//=========================================================================
//                          VID_File_Open
//-------------------------------------------------------------------------
//
// Purpose:   Function to open a .vid file
//
// Inputs:    pathname        Pathname of the multi-frame imagery data file
//
// Outputs:   vidinfo         Structure dumpvid_s of file metadata
//            status          Return argument 0 = success, -1 = error "errno"
//
//
//=========================================================================

int   VID_File_Open( char *pathname, struct dumpvid_s **vidinfo )
{
int     openstatus;


      //----------------- Open the image file and obtain the header info

      openstatus = 0;
      if( dumpvid_open( vidinfo, pathname, 0 ) != 0 )  {
          fprintf( stdout, " ERROR ===>  Cannot open image file %s in VID_File_Open\n", pathname );
          fprintf( stderr, " ERROR ===>  Cannot open image file %s in VID_File_Open\n", pathname );
		  Delay_msec(10000);
          openstatus = -1;
      }

	  return( openstatus );

}

//=========================================================================
//                             VID_Frame_Read
//-------------------------------------------------------------------------
//
// Purpose:   Function to read the next frame in the .vid file
//
// Inputs:    vidinfo         Structure dumpvid_s of file metadata
//
//
// Outputs:  *image_ptr       Pointer to the user's unsigned short imagery array
//            seqnumber       Image sequence number (frame#)
//            status          Return value #frames read (should be 1 as configured here)
//                                0 for EOF,  -1 for error
//
//=========================================================================

int    VID_Frame_Read( struct dumpvid_s *vidinfo, unsigned short *image_ptr, double *unixseconds, long *seqnum )
{
int        status, Nframes2read;
long       kpixel, npixels;
struct     dumpvid_frame_s  vidframe;
uint8_t   *pixmap8;
uint16_t  *pixmap16;


    //======== Allocate memory for the frame read buffer

    vidframe.buf = malloc( vidinfo->seqlen );

    if( vidframe.buf == NULL )  {
		dumpvid_close( vidinfo );
		fprintf( stdout, "ERROR ===> Memory not aloocated in VID_Frame_Read\n" );
		fprintf( stderr, "ERROR ===> Memory not aloocated in VID_Frame_Read\n" );
		Delay_msec(10000);
		exit(1);
	}


	//======== Read the next frame and and associated header "stamp" from the file

	Nframes2read = 1;

	status = dumpvid_frame_read( vidinfo, &vidframe, Nframes2read );

	if( status <= 0 )  return( status );  //  EOF = 0  or  ERROR = -1


	//======== Get the frame's sequence number (frame#) and frame time in seconds since unix start

	*seqnum = (long)vidframe.st.seq;

	*unixseconds = (double)vidframe.st.ts + (double)vidframe.st.tu / 1000000.0;


	//======== Clear the header "stamp" that is embedded in the first row of the image

	memset( vidframe.buf, 0, vidframe.st.headlen );


	//======== Assign the appropriate pointer and copy image to a user array based on bit depth

	npixels = vidframe.st.ht * vidframe.st.wid;

	if( vidframe.st.depth <= 8 )  {

		pixmap8 = (uint8_t*)vidframe.buf;

		for( kpixel=0; kpixel<npixels; kpixel++ )  image_ptr[kpixel] = (unsigned short)pixmap8[kpixel];

	}

	else  {

		pixmap16 = (uint16_t*)vidframe.buf;

		for( kpixel=0; kpixel<npixels; kpixel++ )  image_ptr[kpixel] = (unsigned short)pixmap16[kpixel];

	}


	//======== Free the image read buffer

	free( vidframe.buf );

	return( status );


}


//=========================================================================
//                          VID_File_Close
//-------------------------------------------------------------------------
//
// Purpose:   Function to close a .vid file
//
// Inputs:    pathname        Pathname of the multi-frame imagery data file
//
// Outputs:   status          Return argument 0 closed, -1 if vidinfo is NULL
//
//
//=========================================================================

int   VID_File_Close( struct dumpvid_s *vidinfo )
{
int   closestatus;


      //----------------- Close the image file

      closestatus = dumpvid_close( vidinfo );

	  return( closestatus );

}


//=========================================================================
//                       SetParametersFromVIDfile
//-------------------------------------------------------------------------
//
// Purpose:   Function to read some info from the VID header for the next frame
//            to fill the parameter structure. Usually used for the first frame
//            called from the VID file.
//
// Inputs:    vidinfo         Structure dumpvid_s of file metadata
//
// Outputs:  *params          Pointer to the EMCCDparameters structure
//               nrows            Number of rows
//               ncols            Number of columns
//               cameranumber     Camera number
//               sample_time_sec  Sample time spacing in seconds
//            jdtstart        Julian date of collection start (very first frame)
//
//=========================================================================

void   SetParametersFromVIDfile( struct dumpvid_s *vidinfo, struct EMCCDparameters *params, double *jdtstart )
{
int     status, nrows, ncols, cameranumber;
double  jdt_unixstart, jdt_sinceunix;
struct  dumpvid_stamp_s  stamp;


    //======== Get the "stamp" header which sould be the same dimensions for all frames

    if( dumpvid_stamp_peek( vidinfo, &stamp ) != 0 )  {
		dumpvid_close( vidinfo );
		fprintf( stdout, "ERROR ===> Attempting to peek at first .vid stamp failed in SetParametersFromVIDfile\n" );
		fprintf( stderr, "ERROR ===> Attempting to peek at first .vid stamp failed in SetParametersFromVIDfile\n" );
		Delay_msec(10000);
		exit(1);
	}

	//======== Get the Julian date of the starting first frame plus the unix start date such that
	//             ts is in seconds since January 1, 1970 0:00:00 UTC and tu is the additive microseconds.

	jdt_unixstart = 2440587.5;  //... Julian date for January 1, 1970  0:00:00 UTC

	jdt_sinceunix = ( (double)stamp.ts + (double)stamp.tu / 1000000.0 ) / 86400.0;

	*jdtstart = jdt_unixstart + jdt_sinceunix;


	//======== Assign values to the EMCCDparameters structure

	params->nrows           = stamp.ht;
	params->ncols           = stamp.wid;
	params->cameranumber    = 256 * stamp.num + 65 + stamp.str;  // 256 * site#  +  ascii(stream#),  582 = 02F

    //params->sample_time_sec = (double)stamp.expose / 1000.0;  //... exposure time in milliseconds not implemented in stamp header
	                                                            // override value if available for .vid

}


//=========================================================================
//                       GetParametersFromVIDfile
//-------------------------------------------------------------------------
//
// Purpose:   Function to read some info from the VID header info for the
//            next available frame in the file and pass the parameters
//            out via an argument list.
//
// Inputs:    vidinfo         Structure dumpvid_s of file metadata
//
// Outputs:   nrows            Number of rows
//            ncols            Number of columns
//            cameranumber     Camera number
//            sample_time_sec  Sample time spacing in seconds
//            jdtstart         Julian date of collection start (very first frame)
//
//=========================================================================

void   GetParametersFromVIDfile( struct dumpvid_s *vidinfo,
	                             int *nrows, int *ncols, int *cameranumber,
								 double *sample_time_sec, double *jdtstart )
{
int     status;
double  jdt_unixstart, jdt_sinceunix;
struct  dumpvid_stamp_s  stamp;


    //======== Get the "stamp" header which sould be the same dimensions for all frames

    if( dumpvid_stamp_peek( vidinfo, &stamp ) != 0 )  {
		dumpvid_close( vidinfo );
		fprintf( stdout, "ERROR ===> Attempting to peek at next .vid stamp failed in GetParametersFromVIDfile\n" );
		fprintf( stderr, "ERROR ===> Attempting to peek at next .vid stamp failed in GetParametersFromVIDfile\n" );
		Delay_msec(10000);
		exit(1);
	}

	//======== Get the Julian date of the starting first frame plus the unix start date such that
	//             ts is in seconds since January 1, 1970 0:00:00 UTC and tu is the additive microseconds.

	jdt_unixstart = 2440587.5;  //... Julian date for January 1, 1970  0:00:00 UTC

	jdt_sinceunix = ( (double)stamp.ts + (double)stamp.tu / 1000000.0 ) / 86400.0;

	*jdtstart = jdt_unixstart + jdt_sinceunix;


	//======== Assign values to the EMCCDparameters structure

	*nrows           = stamp.ht;
	*ncols           = stamp.wid;
	*cameranumber    = 256 * stamp.num + 65 + stamp.str;  // 256 * site#  +  ascii(stream#),  582 = 02F

    *sample_time_sec = (double)stamp.expose / 1000.0;  //... exposure time in milliseconds (not implemented in VID yet)

}

#endif  /*****************************************************************************/