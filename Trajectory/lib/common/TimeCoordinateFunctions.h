
//=========================================================================

double  JulianDateAndTime( long year, long month,  long day, 
                           long hour, long minute, long second, long milliseconds )
{
long    aterm, bterm, jdate;
double  jdt;

              
      //.................. compute julian date terms

      if( (month == 1L)  ||  (month == 2L) )  {
          year = year - 1L;
          month = month + 12L;
      }
      aterm = (long)( year / 100L );
      bterm = 2L - aterm + (long)( aterm / 4L );

      //.................. julian day at 12hr UT
            
      jdate = (long)(  365.25 * (double)(year + 4716L) )
            + (long)( 30.6001 * (double)(month +    1L) )
            + day + bterm - 1524L;
         
      //................... add on the actual UT
         
      jdt = (double)jdate - 0.5 
          + (double)hour   / (24.0)
          + (double)minute / (24.0 * 60.0)
          + (double)second / (24.0 * 60.0 * 60.0)
	      + (double)milliseconds / (24.0 * 60.0 * 60.0 * 1000.0);
                       
      return( jdt );                          
}


//====================================================================


void  CalendarDateAndTime( double jdt,
					       long *year, long *month,  long *day, 
                           long *hour, long *minute, long *second,
						   long *milliseconds )
{
long        jyear, jmonth, jday, jhour, jminute, jsecond, jmilliseconds;
long        aterm, bterm, cterm, dterm, eterm, alpha, intpart;
double      fracpart;

      //.................. JD integer and fractional part
      
      intpart  = (long)(jdt + 0.5);
      fracpart = (jdt + 0.5) - (double)intpart;

      alpha = (long)( ( (double)intpart - 1867216.25 ) / 36524.25 );       
      aterm = intpart + 1 + alpha - (long)( alpha / 4L );
      bterm = aterm + 1524L;
      cterm = (long)( ( (double)bterm - 122.1 ) / 365.25 );
      dterm = (long)( 365.25 * (double)cterm );
      eterm = (long)( ( (double)bterm - (double)dterm ) / 30.6001 );

      //................... date calculation
      
      jday = bterm - dterm - (long)( 30.6001 * (double)eterm );
      
      if( eterm < 14L )  jmonth = (long)eterm -  1;
      else               jmonth = (long)eterm - 13;
      
      if( jmonth > 2L )  jyear  = (long)cterm - 4716;
      else               jyear  = (long)cterm - 4715; 
            
         
      //................... time calculation

      fracpart     += 0.0001 / ( 24.0 * 60.0 * 60.0 ); // Add 0.1 msec for rounding
      fracpart     *= 24.0;      
      jhour         = (long)fracpart;
      fracpart     -= (double)jhour;
      fracpart     *= 60.0;      
      jminute       = (long)fracpart;
      fracpart     -= (double)jminute;
      fracpart     *= 60.0;      
      jsecond       = (long)fracpart;
      fracpart     -= (double)jsecond;
      fracpart     *= 1000.0;      
      jmilliseconds = (long)fracpart;
      
          
      //................... put into output pointers
      
      *year         = jyear;
      *month        = jmonth;
      *day          = jday;
      *hour         = jhour;
      *minute       = jminute;
      *second       = jsecond;
	  *milliseconds = jmilliseconds;
      
                          
}

//=========================================================================

double  LocalSiderealTime( double jdt, double longitude_radians /* +East */ )
{
double  pi, tim, stg, localst_radians;

        //.................. Sidereal time at Greenwich
        
        tim = ( jdt - 2451545.0 ) / 36525.0;
                
        stg = 280.46061837 
            + 360.98564736629 * ( jdt - 2451545.0 )
            + tim * tim * 0.000387933
            - tim * tim * tim / 38710000.0;
                    
        //.................. Local sidereal time

		pi = 4.0 * atan(1.0);
            
        localst_radians = stg * pi / 180.0  +  longitude_radians;
        
        //.................. Set value between 0 and 2*pi radians
        
        while( localst_radians >= +2.0 * pi )  localst_radians -= 2.0 * pi;
        while( localst_radians <   0.0      )  localst_radians += 2.0 * pi;
       
        return( localst_radians ); 
}


//=============================================================

double  LongitudeFromLST( double jdt, double localst_radians )
{
double  pi, tim, stg, longitude_radians;  // longitude returned as +east radians

        //.................. Sidereal time at Greenwich
        
        tim = ( jdt - 2451545.0 ) / 36525.0;
                
        stg = 280.46061837 
            + 360.98564736629 * ( jdt - 2451545.0 )
            + tim * tim * 0.000387933
            - tim * tim * tim / 38710000.0;
                    
        //.................. Longitude given LST
            
		pi = 4.0 * atan(1.0);
            
		longitude_radians = localst_radians - stg * pi / 180.0;
        
        //.................. Set value between -pi and +pi radians
        
        while( longitude_radians >  +pi )  longitude_radians -= 2.0*pi;
        while( longitude_radians <= -pi )  longitude_radians += 2.0*pi;
       
        return( longitude_radians ); 
}

//====================================================================

void     RADec2ThetaPhi( double  RA,        // Note: All angles in radians
                         double  Dec,
                         double  latitude,  // Geodetic
                         double  LST,
						 double *Theta,     // Zenith angle
						 double *Phi  )     // Azimuth CCW +north of east
{
double  pi, hourangle, sinelev, sinlat, coslat;


       sinlat = sin(latitude);
       coslat = cos(latitude);

       hourangle = LST - RA;

	   pi = 4.0 * atan(1.0);

       while( hourangle < -pi )  hourangle += 2.0 * pi;
       while( hourangle > +pi )  hourangle -= 2.0 * pi;

	   //-------- Compute phi defined as azimuth measured +north of east
       
       *Phi = -0.5 * pi - atan2( sin(hourangle), cos(hourangle) * sinlat - tan(Dec) * coslat );
                                    
       if( *Phi < 0.0 )  *Phi += 2.0 * pi;

	   //-------- Compute theta defined as zenith angle
       
       sinelev = sinlat * sin(Dec) + coslat * cos(Dec) * cos(hourangle);
               
       if( sinelev > +1.0 )  sinelev = +1.0;
       if( sinelev < -1.0 )  sinelev = -1.0;

       *Theta = 0.5 * pi - asin( sinelev );
       
} 

//====================================================================

void     ThetaPhi2RADec( double  Theta,     // Zenith angle
                         double  Phi,       // Azimuth +north of east
                         double  latitude,  // Geodetic
                         double  LST,       
						 double *RA,
						 double *Dec    )   // Note: All angles in radians
{
double  pi, hourangle, sinlat, coslat, elev, azim;


	   pi = 4.0 * atan(1.0);

	   sinlat = sin(latitude);
       coslat = cos(latitude);

	   elev = +0.5 * pi - Theta;  //... convert zenith angle to elevation

	   azim = +0.5 * pi - Phi;    //... convert +north of east to +east of north

       //--- The formulae signs assume azimuth measured +east of north

	   *Dec = asin( sinlat * sin(elev) + coslat * cos(elev) * cos(azim) );
                        
       hourangle = atan2( -sin(azim), tan(elev)*coslat - cos(azim)*sinlat );                
       
       *RA = LST - hourangle;
       
       while( *RA < 0.0      )  *RA += 2.0 * pi;
       while( *RA > 2.0 * pi )  *RA -= 2.0 * pi;

} 


//=========================================================================
//                     Convert_ColRow2AzimZA
//-------------------------------------------------------------------------
//
// Purpose:   Function to convert column and row measurements to azimuth
//            "phi" and zenith angle "theta" using an astrometric plate 
//            solution. Phi is defined as +north of east. 
//            All angles in radians.
//=========================================================================

void  Convert_ColRow2ThetaPhi( struct plate_s *plate_solution, 
	                           double col, double row,
							   double *zenith_angle, double *azimuth_nofe )
{

#ifdef _WIN32 /************* WINDOWS ******************************/

  *azimuth_nofe = 0.0;
  *zenith_angle = 0.0;

#else /********************** LINUX *******************************/

  plate_map( plate_solution, col, row, zenith_angle, azimuth_nofe );

#endif /***********************************************************/

}