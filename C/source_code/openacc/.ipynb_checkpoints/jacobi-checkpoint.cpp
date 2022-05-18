#include <stdio.h>

#include "jacobi.h"

void jacobistep(double *psinew, double *psi, int m, int n)
{
  int i, j;

// #pragma acc data copyin(psi[:(m+2)*(n+2)]) copyout(psinew[:(m+2)*(n+2)])
#pragma acc parallel loop private(i,j) collapse(2) async(1) 
  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  psinew[i*(m+2)+j]=0.25*(psi[(i-1)*(m+2)+j]+psi[(i+1)*(m+2)+j]+psi[i*(m+2)+j-1]+psi[i*(m+2)+j+1]);
        }
    }
#pragma acc wait
  
}

void jacobistepvort(double *zetnew, double *psinew,
		    double *zet, double *psi,
		    int m, int n, double re)
{
  int i, j;
    
// #pragma acc data copyin(psi[:(m+2)*(n+2)],zet[:(m+2)*(n+2)]) copyout(psinew[:(m+2)*(n+2)],zetnew[:(m+2)*(n+2)])
    {
//#pragma acc parallel loop private(i,j) async(1) 
        {
#pragma acc parallel loop private(i,j) collapse(2) async(1) 
  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  psinew[i*(m+2)+j]=0.25*(  psi[(i-1)*(m+2)+j]+psi[(i+1)*(m+2)+j]+psi[i*(m+2)+j-1]+psi[i*(m+2)+j+1]
			     - zet[i*(m+2)+j] );
	}
    }
  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  zetnew[i*(m+2)+j]=0.25*(zet[(i-1)*(m+2)+j]+zet[(i+1)*(m+2)+j]+zet[i*(m+2)+j-1]+zet[i*(m+2)+j+1])
	    - re/16.0*(
		       (  psi[i*(m+2)+j+1]-psi[i*(m+2)+j-1])*(zet[(i+1)*(m+2)+j]-zet[(i-1)*(m+2)+j])
		       - (psi[(i+1)*(m+2)+j]-psi[(i-1)*(m+2)+j])*(zet[i*(m+2)+j+1]-zet[i*(m+2)+j-1])
		       );
	}
    }
    }
    }
    #pragma acc wait
}

double deltasq(double *newarr, double *oldarr, int m, int n)
{
  int i, j;

  double dsq=0.0;
  double tmp;

#pragma acc parallel loop private(i,j) collapse(2) async(1) 
  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  tmp = newarr[i*(m+2)+j]-oldarr[i*(m+2)+j];
      #pragma acc atomic
      {
          dsq += tmp*tmp;
      }
        }
    }
#pragma acc wait
  return dsq;
}
