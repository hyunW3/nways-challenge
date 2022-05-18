# System dependent definitions

FC=	nvfortran
FFLAGS=	-O3
LFLAGS= -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/11.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/lib64 -lnvToolsExt

# System independent definitions

MF=	Makefile

EXE=	cfd

SRC= \
        nvtx.f90 \
	boundary.f90 \
	cfd.f90 \
	cfdio.f90 \
	jacobi.f90

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .f90 .o

OBJ=	$(SRC:.f90=.o)

.f90.o:
	$(FC) $(FFLAGS) -c $<

all:	$(EXE)

$(EXE):	$(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)

cfd.o:	boundary.o jacobi.o cfdio.o

tar:
	tar cvf cfd.tar $(MF) $(INC) $(SRC)

clean:
	rm -f $(OBJ) $(EXE) *.mod velocity.dat colourmap.dat cfd.plt core
