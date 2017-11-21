
//#######################################################################################
//               Cluster and Tracker Attribute Structure Definition
//#######################################################################################

struct  attributeinfo
{
	double    time;          //  Time in user defined units (sec or frame# or...)
	double    rowcentroid;   //  Row centroid in pixels
	double    colcentroid;   //  Col centroid in pixels
	long      nexceed;       //  Number of exceedance pixels in region
	long      nsaturated;    //  Number of saturated pixels in region
	long      npixels;       //  Number of total pixels in region
	double    sumsignal;     //  Summed signal over image chip region
	double    sumnoise;      //  Summed noise over image chip region
	double    SsqNdB;        //  Signal squared over noise detection metric in dB
	double    entropy;       //  Region entropy
	double    PCD_angle;     //  Phase coded disk angle estimate in deg
	double    PCD_width;     //  Phase coded disk line width estimate in pixels
	double    Hueckel_angle; //  Hueckel kernel angle estimate in deg
};
