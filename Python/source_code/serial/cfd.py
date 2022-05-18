# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
#!/usr/bin/env python
#
# CFD Calculation
# ===============
#
# Simulation of inviscid flow in a 2D box using the Jacobi algorithm.
#
# Python version - uses numpy and loops
#
# EPCC, 2014
#
import sys
import time

# Import numpy
import numpy as np
import sys
import cupy.cuda.nvtx as nvtx

def main(argv):

    # Test we have the correct number of arguments
    if len(argv) < 2:
        sys.stdout.write("Usage: cfd.py <scalefactor> <iterations>")
        sys.exit(1)
        
    # Get the systen parameters from the arguments
    scalefactor = int(argv[0])
    niter = int(argv[1])
    
    sys.stdout.write("\n2D CFD Simulation\n")
    sys.stdout.write("=================\n")
    sys.stdout.write("Scale factor = {0}\n".format(scalefactor))
    sys.stdout.write("Iterations   = {0}\n".format(niter))
    
    # Set the minimum size parameters
    mbase = 32
    nbase = 32
    bbase = 10
    hbase = 15
    wbase =  5
    
    # Set the dimensions of the array
    m = mbase*scalefactor
    n = nbase*scalefactor
    
    # Set the parameters for boundary conditions
    b = bbase*scalefactor 
    h = hbase*scalefactor
    w = wbase*scalefactor

    # Write the simulation details
    sys.stdout.write("\nGrid size = {0} x {1}\n".format(m, n))
    
    # Define the psi array of dimension [m+2][n+2] and set it to zero
    psi = np.zeros((m+2, n+2))

    # Set the boundary conditions on bottom edge
    for i in range(b+1, b+w):
        psi[i][0] = float(i-b)
    for i in range(b+w, m+1):
        psi[i][0] = float(w)

    # Set the boundary conditions on right edge
    for j in range(1, h+1):
        psi[m+1][j] = float(w)
    for j in range(h+1, h+w):
        psi[m+1][j] = float(w-j+h)
    
    # Call the Jacobi iterative loop (and calculate timings)
    sys.stdout.write("\nStarting main Jacobi loop ...\n\n")
    tstart = time.time()
    nvtx.RangePush("jacobi loop")
    jacobi(niter, psi)
    nvtx.RangePop()
    tend = time.time()

    sys.stdout.write("\n... finished\n")
    sys.stdout.write("\nCalculation took {0:.5f}s\n\n".format(tend-tstart))
    
    # Write the output files for subsequent visualisation
    nvtx.RangePush("output visualization")
    write_data(m, n, scalefactor, psi, "velocity.dat", "colourmap.dat")
    nvtx.RangePop()

    # Finish nicely
    sys.exit(0)



def jacobi(niter, psi):

    (m, n) = psi.shape
    m = m - 2
    n = n - 2

    tmp = np.zeros((m+2, n+2))
    for iter in range(1,niter+1):
        # Use index notation and offsets to compute the stream function
        tmp[1:m+1,1:n+1] = 0.25 * (psi[2:m+2,1:n+1]+psi[0:m,1:n+1]+psi[1:m+1,2:n+2]+psi[1:m+1,0:n])

        # Update psi
        np.copyto(psi[1:m+1,1:n+1], tmp[1:m+1,1:n+1])

        if iter%1000 == 0:
            sys.stdout.write("completed iteration {0}\n".format(iter))


def write_data(m, n, scale, psi, velfile, colfile):

    # Open the specified files
    velout = open(velfile, "w")
    velout.write("{0} {1}\n".format(m/scale, n/scale))
    colout = open(colfile, "w")
    colout.write("{0} {1}\n".format(m, n))

    # Loop over stream function array (excluding boundaries)
    for i in range(1, m+1):
        for j in range(1, n+1):

            # Compute velocities and magnitude
            ux =  (psi[i][j+1] - psi[i][j-1])/2.0
            uy = -(psi[i+1][j] - psi[i-1][j])/2.0
            umod = (ux**2 + uy**2)**0.5

            # We are actually going to output a colour, in which
            # case it is useful to shift values towards a lighter
            # blue (for clarity) via the following kludge...

            hue = umod**0.6
            colout.write("{0:5d} {1:5d} {2:10.5f}\n".format(i-1, j-1, hue))

            # Only write velocity vectors every "scale" points
            if (i-1)%scale == (scale-1)/2 and (j-1)%scale == (scale-1)/2:
                velout.write("{0:5d} {1:5d} {2:10.5f} {3:10.5f}\n".format(i-1, j-1, ux, uy))

    velout.close()
    colout.close()

# Function to create tidy way to have main method
if __name__ == "__main__":
        main(sys.argv[1:])
