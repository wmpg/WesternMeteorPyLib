

//#######################################################################################################################
//  Multi-dimensional dynamic array generation for all data types (short,int,long,float,double,...) using a single
//     contiguous chunk of memory which makes releasing memory a simple operation.
//
//     1D Memory allocation:      user_array = (type*)   allocate_1d_array( dim1, sizeof(type) );
//     2D Memory allocation:      user_array = (type**)  allocate_2d_array( dim1, dim2, sizeof(type) );
//     3D Memory allocation:      user_array = (type***) allocate_3d_array( dim1, dim2, dim3, sizeof(type) );
//
//     Assignment and usage:      user_array[i]       = ...;   for 1D arrays
//     Assignment and usage:      user_array[i][j]    = ...;   for 2D arrays
//     Assignment and usage:      user_array[i][j][k] = ...;   for 3D arrays
//
//     Such that indexing is as follows:  [dim1][dim2][dim3]
//
//     Freeing memory:            delete [] user_array;     One simple call for all arrays types and sizes
//
//#######################################################################################################################

/*
void*** allocate_3d_array( const int dim1, const int dim2, const int dim3, const size_t nsize )
{
int  i, j;

  void*** array3d = (void***) malloc( (dim1 * sizeof(void**)) + (dim1*dim2 * sizeof(void*)) + (dim1*dim2*dim3 * nsize) );

  if( array3d == NULL )  {
	  fprintf( stdout, "ERROR ===> Memory not allocated for 3D array in function allocate_3d_array\n" );
	  fprintf( stderr, "ERROR ===> Memory not allocated for 3D array in function allocate_3d_array\n" );
	  Delay_msec(10000);
	  exit(1);
  }

  for(i = 0; i < dim1; ++i) {
      array3d[i] = (void**)( array3d + dim1 * sizeof(void**) + i * dim2 * sizeof(void*) );
      for(j = 0; j < dim2; ++j) {      PROBLEM WITH actual sizes of void* and void**
          array3d[i][j] = (void*)( array3d + dim1 * sizeof(void**) + dim1 * dim2 * sizeof(void*) + i * dim2 * dim3 * nsize + j * dim3 * nsize );
      }
  }

  return array3d;
}

*/

//====================================================================

void** allocate_2d_array( const int dim1, const int dim2, const size_t nsize )
{
int i;

  void** array2d = (void**) malloc( (dim1 * sizeof(void*)) + (dim1 * dim2 * nsize) );

  if( array2d == NULL )  {
	  fprintf( stdout, "ERROR ===> Memory not allocated for 2D array in function allocate_2d_array\n" );
	  fprintf( stderr, "ERROR ===> Memory not allocated for 2D array in function allocate_2d_array\n" );
	  Delay_msec(10000);
	  exit(1);
  }


  for(i = 0; i < dim1; ++i) {
      ////array2d[i] = (void*)( (array2d + dim1) * sizeof(void*) + i * dim2 * nsize );
      array2d[i] = (void*)( (array2d + dim1) + i * dim2 * nsize / sizeof(void*) );
  }

  return array2d;
}

//====================================================================

void* allocate_1d_array( const int dim1, const size_t nsize )
{
  void* array1d = (void*) malloc( dim1 * nsize );  

  if( array1d == NULL )  {
	  fprintf( stdout, "ERROR ===> Memory not allocated for 1D array in function allocate_1d_array\n" );
	  fprintf( stderr, "ERROR ===> Memory not allocated for 1D array in function allocate_1d_array\n" );
	  Delay_msec(10000);
	  exit(1);
  }

  return array1d;
}

//====================================================================

