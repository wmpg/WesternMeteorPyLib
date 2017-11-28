
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
