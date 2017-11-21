
//======================================================================================================================================
//  The WriteBMPfile_RGB function creates a very basic Win3 bit mapped file given an image passed as red, green and blue unsigned
//  char arrays. If the red, green and blue pointers are identicle, then the file is a gray scale 8-bit image, otherwise it will be
//  a 24-bit RGB image. This function flips the image upside-down internally before writing, thus adhering to standard BMP format.
//  There is no compression.
//======================================================================================================================================

void  WriteBMPfile_RGB( char *filename, unsigned char *red_ptr, unsigned char *grn_ptr, unsigned char *blu_ptr, long nrows, long ncols )
{
long          padSize, dataSize, paletteSize, fileSize, krow, kcol, krc, colorsperpixel;

unsigned char bmpfileheader[14] = { 'B','M',       // BMP file identifier 
	                                0,0,0,0,       // Size in bytes
									0,0,           // App data
									0,0,           // App data
									54,0,0,0 };    // Data offset from beginning of file

unsigned char bmpinfoheader[40] = { 40,0,0,0,      // info hd size
	                                0,0,0,0,       // width
									0,0,0,0,       // heigth
									1,0,           // number color planes
									24,0,          // bits per pixel (default RGB)
	                                0,0,0,0,       // compression is none
									0,0,0,0,       // padded image size in bytes
									0x13,0x0B,0,0, // horz resoluition in pixel / m 
									0x13,0x0B,0,0, // vert resolutions (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
									0,0,0,0,       // #colors in pallete
									0,0,0,0 };     // #important colors

unsigned char bmppad[3] = { 0, 0, 0 };
unsigned char BGRA[4]   = { 0, 0, 0, 0 };

FILE  *BMPFile;


    //======== Check if this to be a gray scale or RGB image

    if( red_ptr == grn_ptr  &&  red_ptr == blu_ptr )  colorsperpixel = 1;  //... gray only
	else                                              colorsperpixel = 3;  //... red/green/blue



    //======== Overwrite the file size, width and height dimensions in the headers
    //             making sure the column count is padded to a multiple of four

    padSize = ( 4 - ( ( ncols * colorsperpixel ) % 4 ) ) % 4; 

	dataSize = colorsperpixel * nrows * ncols  +  nrows * padSize;

	if( colorsperpixel == 1 )  paletteSize = 1024; // 256 gray levels in RGBA format
	else                       paletteSize = 0;

    fileSize = 54  +  dataSize  +  paletteSize;

    bmpfileheader[ 2] = (unsigned char)( fileSize       );
    bmpfileheader[ 3] = (unsigned char)( fileSize >>  8 );
    bmpfileheader[ 4] = (unsigned char)( fileSize >> 16 );
    bmpfileheader[ 5] = (unsigned char)( fileSize >> 24 );

    bmpinfoheader[ 4] = (unsigned char)( ncols       );
    bmpinfoheader[ 5] = (unsigned char)( ncols >>  8 );
    bmpinfoheader[ 6] = (unsigned char)( ncols >> 16 );
    bmpinfoheader[ 7] = (unsigned char)( ncols >> 24 );

    bmpinfoheader[ 8] = (unsigned char)( nrows       );
    bmpinfoheader[ 9] = (unsigned char)( nrows >>  8 );
    bmpinfoheader[10] = (unsigned char)( nrows >> 16 );
    bmpinfoheader[11] = (unsigned char)( nrows >> 24 );

	bmpinfoheader[14] = (unsigned char)( 8 * colorsperpixel );  //... bits per pixel

    bmpinfoheader[20] = (unsigned char)( dataSize       );
    bmpinfoheader[21] = (unsigned char)( dataSize >>  8 );
    bmpinfoheader[22] = (unsigned char)( dataSize >> 16 );
    bmpinfoheader[23] = (unsigned char)( dataSize >> 24 );


	//======== Open and write the header

    if( ( BMPFile = fopen( filename, "wb" ) ) == NULL )  {
        fprintf( stdout, "ERROR ===> BMP file %s could not be opened\n\n", filename );
        fprintf( stderr, "ERROR ===> BMP file %s could not be opened\n\n", filename );
        exit(1);
    }

	fwrite( bmpfileheader, sizeof(unsigned char), 14, BMPFile );
	fwrite( bmpinfoheader, sizeof(unsigned char), 40, BMPFile );


	//======== If grayscale, insert the color palette

	if( colorsperpixel == 1 )  {
		
		for ( krow=0; krow<256; krow++ )  {

			BGRA[0] = (unsigned char)krow;
			BGRA[1] = (unsigned char)krow;
			BGRA[2] = (unsigned char)krow;
			BGRA[3] = (unsigned char)0;

			fwrite( BGRA, sizeof(unsigned char), 4, BMPFile );

		}

	}


	//======== Write the imagery data portion, flipping the image up/down

    for ( krow=nrows-1; krow>=0; krow-- )  {

		krc = krow * ncols;

        for ( kcol=0; kcol<ncols; kcol++ )  {

			BGRA[0] = blu_ptr[krc];
			BGRA[1] = grn_ptr[krc];
			BGRA[2] = red_ptr[krc];

			fwrite( BGRA, sizeof(unsigned char), colorsperpixel, BMPFile );

			krc++;

        }

		if( padSize > 0 )  fwrite( bmppad, sizeof(unsigned char), padSize, BMPFile );

	}


	//========= Close the BMP file

	fclose( BMPFile );

}

//=========================================================================

