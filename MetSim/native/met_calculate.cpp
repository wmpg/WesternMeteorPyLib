#include <iostream>
#include <math.h>
#include <fstream>
#include <stdlib.h>

#include "met_calculate.h"

#include <cstring>
#include <ctime>

using namespace std;

double mkill;

	
int  Met_calculate(char input_dir[],char result_dir[],char metsimID[],
	char F[])
{

	ifstream datain;
	ofstream dataout,times,sizes,acceldat;
	meteor large_frag;//,tmp,*main_array;
	constscl consts;
	char inputfile[300],str[300],c,binfn[300],szsfn[300],timfn[300],accelfn[300];
	double m_init,h_init,s_init,v_init,vv_init,vh_init,MetTemp_init;
	double r_init, T_sphere;//,T_lim_init;
	int ncmp; //cmp[ncmps_max],
	double mcmp[ncmps_max];	//mass of grains (in each grain size)
	double rho_meteor[ncmps_max],q[ncmps_max],T_boil[ncmps_max],T_fus[ncmps_max],c_p[ncmps_max],Psi[ncmps_max],
		m_mass[ncmps_max],therm_cond[ncmps_max],lum_eff[ncmps_max],poros,metvol;
	long idum,i;//,j,k,N_tot,ma_imax,n_sizes[nszs_max],;
	
	idum=1235; //for the random number generator, used for fundamental grain size
	strcpy(inputfile,input_dir);
	strcat(inputfile,F);
	
	//read initial and default variables/constants required for calculation from a metsim input file
	datain.open(inputfile);
	if (datain.fail()) 
	{
		cout << "Could not find file " << inputfile << endl;
		return(1);
	}
	//read in parameters
	datain.getline(str,300);
	for (i=0;i<6;i++)
		datain.get(c);
	datain >> ncmp;
	if (ncmp>ncmps_max)
	{
		cout << "Too many compositions: no more than " << ncmps_max << " are allowed." << endl;
		return(1);
	}
	m_init=0;
	datain.getline(str,300);

	for (i=0;i<ncmp;i++)
	{
		datain >> mcmp[i];// >> cmp[i];
		datain.getline(str,300);
		m_init+=mcmp[i];
	}
	datain.getline(str,300);

	//meteoroid properties
	for(i=0;i<ncmp;i++)
	{	datain >> rho_meteor[i];datain.getline(str,300);}
	datain.getline(str,300);
	datain >> poros;

	datain.getline(str,300);
	metvol=0;
	for (i=0;i<ncmp;i++)
		metvol+=mcmp[i]/rho_meteor[i];
	metvol*=1+poros;
	r_init=pow((metvol*3/4/pi),(1.0/3));

	datain.getline(str,300);
	for(i=0;i<ncmp;i++)
	{	datain >> q[i];datain.getline(str,300);}
	datain.getline(str,300);
	for(i=0;i<ncmp;i++)
	{	datain >> T_boil[i];datain.getline(str,300);}
	datain.getline(str,300);
	for(i=0;i<ncmp;i++)
	{	datain >> T_fus[i];datain.getline(str,300);}
	datain.getline(str,300);

	for(i=0;i<ncmp;i++)
	{	datain >> c_p[i];datain.getline(str,300);}
	datain.getline(str,300);
	for(i=0;i<ncmp;i++)
	{	datain >> Psi[i];datain.getline(str,300);}
	datain.getline(str,300);
	for(i=0;i<ncmp;i++)
	{	datain >> m_mass[i];datain.getline(str,300);m_mass[i]=m_mass[i]*1.6726231e-27;}
	datain.getline(str,300);
	for(i=0;i<ncmp;i++)
	{	datain >> therm_cond[i];datain.getline(str,300);}
	datain.getline(str,300);
	for(i=0;i<ncmp;i++)
	{	datain >> lum_eff[i];datain.getline(str,300);}
	datain.getline(str,300);
	datain >> T_sphere;
	datain.getline(str,300);
	datain.getline(str,300);	
	datain >> consts.shape_fact;
	datain.getline(str,300);
	datain.getline(str,300);
	datain >> consts.emiss;

	//initial conditions
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
	datain >> h_init;
	datain.getline(str,300);
	datain.getline(str,300);
	datain >> s_init;
	datain.getline(str,300);
	datain.getline(str,300);
	datain >> v_init;
	datain.getline(str,300);
	datain.getline(str,300);
	datain >> consts.z;
	datain.getline(str,300);
	datain.getline(str,300);
	consts.zr=consts.z*pi/180;
	vv_init=-v_init*cos(consts.zr);
	vh_init=v_init*sin(consts.zr);
	datain >> MetTemp_init;
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
	datain >> consts.h_obs;
	datain.getline(str,300);
	datain.getline(str,300);
	datain >> consts.dt;
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
	for(i=0;i<6;i++)
	{	datain >> consts.dens_co[i];datain.getline(str,300);}
	datain.getline(str,300);
	datain >> consts.maxht_rhoa; datain.getline(str,300);
	datain >> consts.minht_rhoa; datain.getline(str,300);
	
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
	datain.getline(str,300);
		for(i=0;i<6;i++)
	{	datain >> consts.Press_co[i];datain.get(c);}
	datain.close();
	consts.T_a=280;
	consts.nrec=0;
	consts.h_min=200000;
	consts.t_max=0;
	consts.P_sur=1.01325e5;
	consts.kappa=1.39;
	
	//write parameters to one file, spectral data to another

	strcpy(accelfn,result_dir);
	strcat(accelfn,"Metsim");
	strcat(accelfn,metsimID);
	strcat(accelfn,"_accel2.txt");
	acceldat.open(accelfn);
	
	strcpy(binfn,result_dir);
	strcat(binfn,"Metsim");
	strcat(binfn,metsimID);
	strcat(binfn,"_results.bin");
    dataout.open(binfn);
    
    //currently, fits for pressure and density of the atmosphere are only valid between 200 and
	//60 km. Results may not be valid outside this range
	if (h_init > consts.maxht_rhoa){
		cout << "**** WARNING: Beginning height exceeds " << consts.maxht_rhoa << " km ****" << endl;
		cout << "Results of this integration may not be valid" << endl;
		cout << "Proceeding with integration" << endl;
	}
	
	//n = 1; //number of fragments

	clock_t begin = clock();
        
    mkill=1e-14;//-7.627e-15*v_init/1000+1.034e-12+(9e-18)*consts.z*consts.z;//5e-13;
    if (mkill>m_init) mkill=m_init;

	//define parameters of original meteoroid
	large_frag.t=0;
	large_frag.s=s_init;               		
	large_frag.h=h_init;               		
	large_frag.v=v_init;               		
	large_frag.p2=0;             		
	large_frag.m=m_init;              		
 	large_frag.vh=vh_init;             		
	large_frag.vv=vv_init;             		
	large_frag.temp=MetTemp_init;		  		
	large_frag.m_kill=mkill;				
	large_frag.T_int=MetTemp_init;
	large_frag.ncmp=ncmp;
	for (i=0;i<ncmp;i++)
	{	
		large_frag.mcmp[i]=mcmp[i];			
		large_frag.rho_meteor[i]=rho_meteor[i];		
		large_frag.T_boil[i]=T_boil[i];				
		large_frag.T_fus[i]=T_fus[i];					
		large_frag.c_p[i]=c_p[i];						
		large_frag.q[i]=q[i];							
		large_frag.psi[i]=Psi[i];						
		large_frag.m_mass[i]=m_mass[i];				
		large_frag.therm_cond[i]=therm_cond[i];
		large_frag.lum_eff[i]=lum_eff[i];
	}
	large_frag.Vtot=metvol;
	large_frag.poros=poros;	
	large_frag.T_sphere=T_sphere;
	large_frag.n_frag=1;						
	large_frag.Fl_frag_cond=0;					//1 if the particle should fragment now
	large_frag.Fl_frag_ever=0;					//1 if the particle will fragment sometime
	large_frag.Fl_ablate=1;					//1 if the particle is still actively ablating
	

 	//ablate large_frag until all remaining mass is in main_array
 	while ((large_frag.Fl_ablate==1)&&(large_frag.m>mkill)&&(large_frag.h/1000.0<200))
 	//for (int i = 0; i < 2; ++i)
 	{
 		ablate(large_frag,consts,dataout,acceldat);
 	}

 	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
 	
	dataout.close();
	cout << "Calculation done" << endl;

	cout << "Runtime:" << elapsed_secs << endl;

	//write dt and t_max into file for procedure 'evaluate'
	strcpy(timfn,result_dir);
	strcat(timfn,"METSIM");
	strcat(timfn,metsimID);
	strcat(timfn,"_times.txt");
	strcpy(szsfn,result_dir);
	strcat(szsfn,"METSIM");
	strcat(szsfn,metsimID);
	strcat(szsfn,"_sizes.txt");
	sizes.open(szsfn);
	times.open(timfn);
	times <<  "dt=" <<  consts.dt << endl;
	times << "t_max=" <<  consts.t_max << endl;
	times.close();
	
	sizes.close();
	cout << "Minimum height was " <<  consts.h_min/1000.0 << " km" << endl;
	if (consts.h_min < consts.minht_rhoa)
	{
		cout << "**** WARNING: Final height of some fragments was below " << consts.minht_rhoa << " km ****" << endl;
		cout << "Atmospheric model not valid in this range: results may not be valid" << endl;
		cout << "Proceeding to evaluate" << endl;
	}
	acceldat.close();

	// TEST
	printf("ATM: %f", atm_dens(100000, consts.dens_co)*1000000000);
	printf("Scale ht: %f", scaleht(100000, consts.dens_co));

	return(0);
}

double atm_dens(double h,double dens_co[])
{
	//uses the coefficients from the input file to calculate atmospheric density
	double rho_a;
	
	rho_a=pow(10,(dens_co[0]+dens_co[1]*h/1000+dens_co[2]*pow((h/1000),2)
       		+dens_co[3]*pow((h/1000),3)+ 
       		dens_co[4]*pow((h/1000),4)+dens_co[5]*pow((h/1000),5)))*1000;
	return(rho_a);
}

double scaleht(double h,double dens_co[])
{
	//calculates the scale height, based on the density at h and 2km away
	double scl_ht;
	scl_ht=2000.0/log(atm_dens(h-2000.0,dens_co)/atm_dens(h,dens_co));
	return(scl_ht);
}

double mass_loss(meteor &met,double rho_atmo,constscl &cs,double Lambda,double m_dotcmp[])
{
	//Evaporation, using the Clausius-Clapeyron equation for vapour pressure (external pressure
	//neglected for now) and the Knudsen-Langmuir formula for evaporation rate.
	//assume that the fraction of energy absorbed by each component of the meteoroid is proportional to its share of the volume
	double  qb1,qb2,qb3,qb4,m_dot;//Pspall,
	double incase;
	int i;
	
	incase=rho_atmo*Lambda; //might need later, get rid of warning.
	m_dot=0;
	for (i=0;i<met.ncmp;i++)
	{

		//printf("mass temp %e\n", met.temp);
		printf("met.q %f\n", met.q[i]);
		printf("met.m_mass %e\n", met.m_mass[i]);
		printf("met.T_boil %e\n", met.T_boil[i]);
		
		qb1=cs.dt * cs.shape_fact * pow((met.mcmp[i]*(1+met.poros)/met.rho_meteor[i]),(2.0/3)) * met.psi[i] * exp(met.q[i]*met.m_mass[i]
			/(k_B*met.T_boil[i]))*cs.P_sur*exp(-met.q[i]*met.m_mass[i]/(k_B*met.temp))/sqrt(2*pi*k_B*met.temp/met.m_mass[i]);

		printf("qb1 %e\n", qb1);

		if (qb1/2>met.mcmp[i]) qb1=met.mcmp[i]*2;
 		qb2=cs.dt * cs.shape_fact * pow(((met.mcmp[i]-qb1/2.0)*(1+met.poros)/met.rho_meteor[i]),(2.0/3)) * met.psi[i] * 
 			exp(met.q[i]*met.m_mass[i]/(k_B*met.T_boil[i]))*cs.P_sur*exp(-met.q[i]*met.m_mass[i]/(k_B*met.temp))
 			/sqrt(2*pi*k_B*met.temp/met.m_mass[i]);
		if (qb2/2>met.mcmp[i]) qb2=met.mcmp[i]*2;
		qb3=cs.dt * cs.shape_fact * pow(((met.mcmp[i]-qb2/2.0)*(1+met.poros)/met.rho_meteor[i]),(2.0/3)) * met.psi[i] * 
			exp(met.q[i]*met.m_mass[i]/(k_B*met.T_boil[i]))*cs.P_sur*exp(-met.q[i]*met.m_mass[i]/(k_B*met.temp))
			/sqrt(2*pi*k_B*met.temp/met.m_mass[i]);
		if (qb3>met.mcmp[i]) qb3=met.mcmp[i];
    	qb4=cs.dt * cs.shape_fact * pow(((met.mcmp[i]-qb3)*(1+met.poros)/met.rho_meteor[i]),(2.0/3)) * met.psi[i] * 
    		exp(met.q[i]*met.m_mass[i]/(k_B*met.T_boil[i]))*cs.P_sur*exp(-met.q[i]*met.m_mass[i]/(k_B*met.temp))
    		/sqrt(2*pi*k_B*met.temp/met.m_mass[i]);
    	
    	m_dotcmp[i] = (qb1/6.0 + qb2/3.0 + qb3/3.0 + qb4/6.0) / cs.dt;  //mass loss in kg/s due to ablation
    	if (m_dotcmp[i]*cs.dt>met.mcmp[i]) 
    		m_dotcmp[i]=met.mcmp[i]/cs.dt;
    	m_dot+=m_dotcmp[i];
    }
    
	return(m_dot);
}

double temp_ch(meteor &met,double rho_atmo,double scale_ht,constscl &cs,double Lambda,
	double m_dot,double m_dotcm[])
{
	//calculates the change in temperature
	double qc1,qc2,qc3,qc4,T_dot;//m_therm,r_met,x_0,x_eq,
	double sumcpm,sumqmdot,incase;
	int i;
	
	incase=scale_ht*m_dot; //might need these in future and don't want the warning message

	//change in temperature is calculated from the kinetic energy of air molecules, blackbody
	//radiation and mass ablation
	sumcpm=0;
	sumqmdot=0;
	for (i=0;i<met.ncmp;i++)
	{
		sumcpm+=met.c_p[i]*met.mcmp[i]; // the total thermal inertia
		sumqmdot+=met.q[i]*m_dotcm[i]; // all the energy lost to ablating the components
	}
	
	//Three terms: fraction Lambda of kinetic energy of air, blackbody radiation, energy to ablate mass
	qc1=cs.dt * 1/(sumcpm)* (cs.shape_fact * pow(met.Vtot,(2.0/3))*Lambda * rho_atmo * (pow(met.v,3))/2.0
		-4 * sigma_b * cs.emiss * (pow((met.temp),4)-pow(cs.T_a,4))*pow(met.Vtot,(2.0/3))- sumqmdot);
	qc2=cs.dt * 1/(sumcpm)* (cs.shape_fact * pow(met.Vtot,(2.0/3))*Lambda * rho_atmo * (pow(met.v,3))/2.0
		-4 * sigma_b * cs.emiss * (pow((met.temp+qc1/2.0),4)-pow(cs.T_a,4))*pow(met.Vtot,(2.0/3))- sumqmdot);
	qc3=cs.dt * 1/(sumcpm)* (cs.shape_fact * pow(met.Vtot,(2.0/3))*Lambda * rho_atmo * (pow(met.v,3))/2.0
		-4 * sigma_b * cs.emiss * (pow((met.temp+qc2/2.0),4)-pow(cs.T_a,4))*pow(met.Vtot,(2.0/3))- sumqmdot);
	qc4=cs.dt * 1/(sumcpm)* (cs.shape_fact * pow(met.Vtot,(2.0/3))*Lambda * rho_atmo * (pow(met.v,3))/2.0
		-4 * sigma_b * cs.emiss * (pow((met.temp+qc3),4)-pow(cs.T_a,4))*pow(met.Vtot,(2.0/3))- sumqmdot);	

	T_dot= (qc1/6.0 + qc2/3.0 + qc3/3.0 + qc4/6.0) / cs.dt;
	return(T_dot);
}

void Lum_intens(constscl &cs,double &lum,double m_dot,double& q_ed,double vdot,meteor &met,double m_dotcm[])
{
	double beta,incase;
	int i;
	
	incase=m_dot*vdot;

	//ionization:
	beta=4.0e-18*pow(met.v,3.5); //v in m/s
	q_ed=0;
	for (i=0;i<met.ncmp;i++)
		q_ed=beta*m_dotcm[i]/(met.v*met.m_mass[i]);

	//luminosity
	lum=0;
	for (i=0;i<met.ncmp;i++)
		lum+=0.5*met.v*met.v*m_dotcm[i]*met.lum_eff[i]*1e10/(pow(cs.h_obs,2))+met.lum_eff[i]*met.m*met.v*vdot* 1e10 / (pow(cs.h_obs,2));
	
}

void ablate(meteor &met,constscl &cs,ofstream &outstream,ofstream &accelout)
{
	//Updates  the mass, position, velocity etc and finds the luminosity produced
	//in one time step. Writes results to a file.
	double lum,q_ed,rho_atmo,Scale_ht,MFP,Kn,Gamma,Lambda;
	double p1,Ma,qa1,qa2,qa3,qa4,a_current,gv,av,ah,T_dot,m_dot,v_dot;//sg,
	double AAA,ht,m_dotcm[ncmps_max],Vmax,dVmax,dV,smV;
	int i;
	double dEair,dEdm,dErad;
	meteor tmp;

	//calculate atmospheric parameters 
	
	i=0;
	ht=met.h;

	
	rho_atmo=atm_dens(ht,cs.dens_co);
	Scale_ht=scaleht(met.h,cs.dens_co);
	MFP=pow(10,(-7.1-log10(rho_atmo)));	//mean free path
	Kn=(MFP/pow((3*met.Vtot/(4*pi)), 1.0/3.0));	//Knudsen number

	printf("\n");
	printf("h %e\n", met.h);
	printf("rho_atmo %e\n", rho_atmo);
	printf("Scale_ht %e\n", 	Scale_ht);
	printf("MFP %e\n", 	MFP);
	printf("Kn %e\n", 	Kn);

	//separate fragments in the free molecular flow, transition and continuum flow regime
	//Gamma is the drag coefficient, Lambda the heat transfer coefficient
	if (Kn >=10) //free molecular flow
	{
		Gamma=1.0;
		Lambda=0.5;
	}
	else if (Kn>=1) //transition flow
	{
		Gamma=1.0;
		Lambda=0.5;
	}
	else //continuum flow
	{
		Gamma=1.0;
		Lambda=0.5;
	}


    //calculation of pressure acting on meteor: atmospheric pressure
    p1=pow(10,(cs.Press_co[0]+cs.Press_co[1]*met.h/1000+cs.Press_co[2]*pow((met.h/1000),2)+cs.Press_co[3]*
    	pow((met.h/1000),3)	+cs.Press_co[4]*pow((met.h/1000),4)+cs.Press_co[5]*pow((met.h/1000),5)));
    Ma = met.v/(sqrt(cs.kappa*R_gas*cs.T_a));      //calculation of Mach-number

    printf("p1 %e\n", p1);
    printf("Ma %e\n", Ma);

	//the following calculation of pressure is valid only for continuum flow
	if (Kn>=1) Ma=0;  //sets pressure to 0 for transition and free molecular flow
    met.p2 = p1 * ((2*cs.kappa/(cs.kappa+1))*Ma*Ma * pow((sin(cs.sigma)),2));   //calculation of pressure

    printf("met.p2 %e\n", met.p2);

    //MCB 20030120 - integrations are a 4th order Runge-Kutta
	m_dot=mass_loss(met,rho_atmo,cs,Lambda,m_dotcm);

	printf("m_dot %e\n", m_dot);

	met.m_kill = m_dot * cs.dt; //anything smaller than this will ablate in less than one time step
		//anything smaller than mkill kg will not ablate, since it will cool faster from radiation
	if (met.m_kill < mkill)  met.m_kill=mkill;
	T_dot=temp_ch(met,rho_atmo,Scale_ht,cs,Lambda,m_dot,m_dotcm);

	printf("T_dot %e\n", T_dot);
	
	//Three terms: fraction Lambda of kinetic energy of air, blackbody radiation, energy to ablate mass
	dEair= (cs.shape_fact * pow(met.Vtot,(2.0/3))*Lambda * rho_atmo * (pow(met.v,3))/2.0);
	dErad= (-4 * sigma_b * cs.emiss * (pow((met.temp),4)-pow(cs.T_a,4))*pow(met.Vtot,(2.0/3)));
	dEdm=  (-met.q[0]*m_dotcm[0]);

	printf("dEair %e\n", dEair);
	printf("dErad %e\n", dErad);
	printf("dEdm %e\n", dEdm);
	
	//change in velocity
        //the change in velocity is calculated from the exchange of momentum with
        //atmospheric molecules (McKinley 1961, Bronshten 1983)
        qa1=cs.dt * Gamma * cs.shape_fact * pow(met.Vtot,2/3.0)* rho_atmo * pow((met.v),2)
        	/ met.m;
        qa2=cs.dt * Gamma * cs.shape_fact * pow(met.Vtot,2/3.0)* rho_atmo * pow((met.v+qa1/2.0),2)
        	/ met.m;
        qa3=cs.dt * Gamma * cs.shape_fact * pow(met.Vtot,2/3.0)* rho_atmo * pow((met.v+qa2/2.0),2)
        	/ met.m;
       	qa4=cs.dt * Gamma * cs.shape_fact * pow(met.Vtot,2/3.0)* rho_atmo * pow((met.v+qa3),2)
        	/ met.m;
	a_current = (qa1/6.0 + qa2/3.0 + qa3/3.0 + qa4/6.0) / cs.dt;   //decelaration in m/(s*s)

	printf("a_current %e\n", a_current);

    gv = g0 / pow((1+met.h/re),2);                      //g at height h in m/s^2
    av = -gv-a_current*met.vv/met.v + met.vh*met.v / (re + met.h);  //- gv //vertical component of a
    ah = -a_current * met.vh/met.v - met.vv*met.v / (re + met.h);    //;horizontal component of a
    v_dot=sqrt(av*av+ah*ah);

    printf("v_dot %e\n", v_dot);


    //Check to make sure dm is less than m for each component, otherwise nonexistant mass will be ablated
    for (i=0;i<met.ncmp;i++)
		if ((m_dotcm[i]*cs.dt) > met.mcmp[i]) 
			m_dotcm[i] = met.mcmp[i]/cs.dt;		//ablate only what's there

	//update parameters
	tmp=met;	//keep original parameters until all updates are finished
    tmp.s = met.s + met.v * cs.dt;// * re / (re + met.h);
    tmp.h = met.h + met.vv * cs.dt;
    tmp.m = met.m - m_dot * cs.dt;
    for (i=0;i<met.ncmp;i++)
    	tmp.mcmp[i]=met.mcmp[i]-m_dotcm[i]*cs.dt;

    //calculate the new volume and porosity
    //Use component with largest volume
    Vmax=0;
    dV=0;
    for (i=0;i<met.ncmp;i++)
    	if (met.mcmp[i]/(met.rho_meteor[i]*(1-met.poros))>Vmax)
    	{
    		Vmax=met.mcmp[i]/(met.rho_meteor[i]*(1-met.poros));
    		dVmax=m_dotcm[i]*cs.dt/(met.rho_meteor[i]*(1-met.poros)); 
    	}
    for (i=0;i<met.ncmp;i++)
    	dV+=m_dotcm[i]*cs.dt/(met.rho_meteor[i]*(1-met.poros)); 
    if (met.ncmp*dVmax<dV)
    	tmp.Vtot-=met.ncmp*dVmax; //subtract smaller of two: porosity increases in this case, is constant in the next.
    else tmp.Vtot-=dV;

    printf("tmp.Vtot %e\n", tmp.Vtot);

    //porosity
    smV=0;
    for (i=0;i<met.ncmp;i++)
    	smV+=tmp.mcmp[i]/tmp.rho_meteor[i]; //volume with no porosity
    tmp.poros=(tmp.Vtot-smV)/tmp.Vtot;

    printf("tmp.poros %e\n", tmp.poros);
    
    //if temperature is high enough, the particle starts to consolidate into a sphere.
    if ((met.temp>met.T_sphere)&&((tmp.poros-0.002)>=0))
    {
    	tmp.poros-=0.002;
    	tmp.Vtot=tmp.Vtot*(1-(tmp.poros+0.002))/(1-tmp.poros);
    	//shape factor here
    }
    if (tmp.poros<0)
    {
    	if (tmp.poros<-1e-7)
    		cout << "negative porosity" << endl;
    	tmp.poros=0;
    }
    
    tmp.temp = met.temp + T_dot * cs.dt;
    AAA=tmp.h;

    //this should not occur, but if negative temperatures are produced, they must be dealt with
    if (tmp.temp < 100)
    {
    	tmp.temp = 100;
    	cout << "Numerical instability in temperature at " << tmp.h/1000 << " km" << endl;
    }



    tmp.vv = met.vv + av * cs.dt;
    tmp.vh = met.vh + ah * cs.dt;
    tmp.v = sqrt (tmp.vh * tmp.vh + tmp.vv * tmp.vv);
    tmp.t = met.t+cs.dt;

	//here the visual luminosity is the kinetic energy lost in this step times lum_eff,
	//the coefficient of luminous intensity.It is also corrected to a range of 100 km.
	//added 07/2004: ionization produced in this step as well
    Lum_intens(cs,lum,m_dot,q_ed,v_dot,met,m_dotcm);

    printf("q_ed %e\n", q_ed);
    printf("lum %e\n", lum);
    printf("met.m %e\n", met.m);


    accelout << met.t << '	' << a_current << '	' << v_dot << '	' << lum << '	' << rho_atmo << '	' << met.v
    	<< '	' << m_dot << '	' << met.m << '	' << T_dot << '	' << met.poros << '	' << met.temp << '	' << met.h/1000.0 
    	<< '	' << dEair << '	' << -dEdm << '	' << -dErad << endl;
	met=tmp;	//put results back in met

	//check fragmentation condition and ablation condition
	if (met.temp>=met.T_lim) met.Fl_frag_cond=1; //will fragment before next step
	if ((met.h<85000)&&(met.temp<(cs.T_a+200))) 
		met.m_kill=met.m;
	if (met.m <=met.m_kill) met.Fl_ablate=0; //no longer ablates

	//write results
	if ((met.h/1000>60)&&(met.h/1000<200))
	{
		outstream << endl;
		outstream << met.t << '	' ;
		outstream.precision(8);
		outstream << met.s/1000.0 << '	' << met.h/1000 << '	' << met.v/1000 << '	';
		outstream << lum << '	';
		outstream  << q_ed << '	' << met.temp << '	' << met.n_frag << '	' << met.m;
	}
	if ((met.h/1000>60)&&(met.t > cs.t_max)) cs.t_max=met.t;
	if ((met.h/1000>60)&&(met.h < cs.h_min)) cs.h_min=met.h;
	cs.nrec = cs.nrec+1;
}