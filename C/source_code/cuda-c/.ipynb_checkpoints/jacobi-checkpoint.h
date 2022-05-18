#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
    
__global__ void jacobistep(double *psinew, double *psi, int m, int n);

__global__ void jacobistepvort(double *zetnew, double *psinew,
		    double *zet,    double* psi,
		    int m, int n, double re);

double deltasq(double *newarr, double *oldarr, int m, int n);
