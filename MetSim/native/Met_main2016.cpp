#include <iostream>
#include <math.h>
#include <fstream>
#include <stdlib.h>

#include "met_calculate.h"

#include <cstring>

using namespace std;

//this version reads in given grain numbers and sizes, rather than generating them from a distribution
extern int Met_calculate(char input_dir[],char result_dir[],char metsimID[],
	char F[]);
extern int Met_evaluate(char result_dir[],char metsimID[]);

int main()
{
	char input_dir[256], result_dir[256],F[50],IDS[5];
	int TIF,ID1,ID2,Err;
	int i,j,k,ii,l;//,n_f;
	
	strcpy(input_dir,"./");//".\\MetSimGen2016\\");
	strcpy(result_dir,"./");//".\\MetSimGen2016\\");

	TIF=0;         //input files must start with "metsim"
	ID1=1;
	ID2=1;
	TIF=ID2-ID1+1; //number of files
	for (ii=0;ii<TIF;ii++)
	{
		//create file name
		if (ID2>=1000) 
			l=(ii+ID1)/1000;
		else l=0;
		if (ID2>=100) 
			k=((ii+ID1)-l*1000)/100;
		else k=0;
		i=((ii+ID1)-100*k-1000*l)/10;
		j=(ii+ID1) % 10;
		//if (ID2>100)
		{
			IDS[0]=l+48;
			IDS[1]=k+48;
			IDS[2]=i+48;
			IDS[3]=j+48;
			IDS[4]='\0';
		}
		//else
		//{
		//	IDS[0]=i+48;
		//	IDS[1]=j+48;
		//	IDS[2]='\0';
		//}
		strcpy(F,"Metsim");
		strcat(F,IDS);
		strcat(F,"_input.txt");
		
		cout << "Simulation = " << ii << ", metsimID = " << IDS << endl;
        cout << "Input file = " << F << endl;
        cout << "Result dir = " << result_dir << endl;

        Err=Met_calculate(input_dir,result_dir,IDS,F);
        //if (Err==0)
		//	Met_evaluate(result_dir,IDS);
	}

	return (0);
}