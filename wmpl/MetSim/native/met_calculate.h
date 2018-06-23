#include <fstream>
#include <iostream>
#include <math.h>
#include <cstring>

using namespace std;

const double sigma_b=5.67036713e-8;		//Stephan-boltzmann constant (W/(m^2K^4))
const double k_B	= 1.3806485279e-23;	//boltzmann constant (J/K)
const double P_sur = 101325;		//Standard atmospheric pressure
const double g0 = 9.81 ;           	//earth acceleration in m/s^2
const double re = 6378000.0;      	//earth radius in m
//const double e = 2.718281828;     	//Euler number
const double e = exp(1.0);     	//Euler number
const double R_gas = 287.2;       	//specific gas constant in J/kg*K
//const double pi=3.14159265358;
const double pi=M_PI;
const int ncmps_max=20;
const int nszs_max=20;

class meteor
{
public:
	double t;
	double s;
	double h;
	double v;
	double p2;
	double m;
	double vh;
 	double vv;
	double temp;
	double m_kill;
	double T_lim;
	double T_int;
	int ncmp;
	double mcmp[ncmps_max];
	double rho_meteor[ncmps_max]; 		//;meteoroid density
	double Vtot;
	double T_boil[ncmps_max];		//boiling point
	double T_fus[ncmps_max];		//melting point
	double c_p[ncmps_max];		//specific heat
	double q[ncmps_max];		//heat of ablation
	double psi[ncmps_max];		//coefficient of condensation
	double m_mass[ncmps_max];		//average molar mass of meteor
	double therm_cond[ncmps_max];		//thermal conductivity of meteoroid
	double lum_eff[ncmps_max];
	double poros;
	double T_sphere;
	double n_frag;	//number of fragments of same mass ablating at same height
	double Fl_frag_cond;	//1 if the particle should fragment now
	double Fl_frag_ever;	//1 if the particle will fragment sometime
	double Fl_ablate; 						//1 if the particle is still actively ablating
};

class constscl
{  
public:
	double emiss;			//emissivity
	double shape_fact;		//Shape factor of meteor
	double z;				//zenith angle (degrees)
	double zr;				//zenith angle in radians
	double dt;				//time step for integration
	double h_obs;			//normalize magnitude to this height
	double sigma;			//arbitrary angle for hypersonic flight
	double kappa;			//ratio of specific heats
	double alph;
	double c_s;	
	double T_a;				//atmospheric temperature
	double dens_co[6];			//coefficients for atm. density
	double Press_co[6];			//coefficients for atm. pressure
	double nrec;						//total number of records
	double t_max;				//largest time reached
	double h_min;				//smallest height reached (by any particle)
	double ion_eff[ncmps_max];
	double P_sur;
	double maxht_rhoa; //largest height for which atm fit is valid
	double minht_rhoa; //smallest height for atm fit
};

void rm_item(double arr[],long ind,long Nrem,long &maxind);

void rm_iteml(long arr[],long ind,long &maxind);

double atm_dens(double h,double dens_co[]);

double scaleht(double h,double dens_co[]);

double mass_loss(meteor &met,double rho_atmo,constscl &cs,double Lambda,double m_dotcmp[]);

double temp_ch(meteor &met,double rho_atmo,double scale_ht,constscl &cs,double Lambda,
	double m_dot,double m_dotcm[]);

void Lum_intens(constscl &cs,double &lum,double m_dot,double& q_ed,double vdot,meteor &met,double m_dotcm[]);

void ablate(meteor &met,constscl& cs,ofstream &outstream,ofstream &accelout);

double gauss(double a,double s,long idum);

double fgauss(double p);

double ran0(long *idum);

void hpsort(unsigned long n, double ra[], double rb[]);

int  Met_calculate(char input_dir[],char result_dir[],char metsimID[], char F[]);