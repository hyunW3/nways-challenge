#include <stdio.h>

#include "jacobi.h"

void jacobistep(double *psinew, double *psi, int m, int n)
{
  int i, j;
    int size = (m+2)*(n+2);
    #pragma target team distribute data map(to:psi[:size]) map(from:psinew[:size]) default(shared) private(i,j,psi,zet)
    {
  // #pragma omp parallel for collapse(2) 
  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  psinew[i*(m+2)+j]=0.25*(psi[(i-1)*(m+2)+j]+psi[(i+1)*(m+2)+j]+psi[i*(m+2)+j-1]+psi[i*(m+2)+j+1]);
        }
    }
    }
}

void jacobistepvort(double *zetnew, double *psinew,
		    double *zet, double *psi,
		    int m, int n, double re)
{
  int i, j;
    int size = (m+2)*(n+2);
#pragma target team distribute  data map(to:psi[:size],zet[:size]) map(from:psinew[:size],zetnew[:size])  //default(shared) private(i,j,psi,zet)  
    {
 // #pragma omp parallel for collapse(2) 
  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  psinew[i*(m+2)+j]=0.25*(  psi[(i-1)*(m+2)+j]+psi[(i+1)*(m+2)+j]+psi[i*(m+2)+j-1]+psi[i*(m+2)+j+1]
			     - zet[i*(m+2)+j] );
         
      
	}
    }
  #pragma omp parallel for collapse(2)      
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
    } // pragma target end
    
}

double deltasq(double *newarr, double *oldarr, int m, int n)
{
  int i, j;

  double dsq=0.0;
  double tmp;

  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  tmp = newarr[i*(m+2)+j]-oldarr[i*(m+2)+j];
	  dsq += tmp*tmp;
        }
    }

  return dsq;
}
