# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

CC=	nvc++
CFLAGS=	-lm -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/lib64 -lnvToolsExt
LFLAGS=


# System independent definitions

MF=	Makefile

EXE=	cfd

INC= \
	arraymalloc.h \
	boundary.h \
	cfdio.h \
	jacobi.h

# Replace/Add .cu files to the below and make the change for the rest of the locations.
SRC= \
	arraymalloc.cpp \
	boundary.cpp \
	cfd.cpp \
	cfdio.cpp \
	jacobi.cpp

#
# Make the necessary changes (hint: replace .cpp with .cu so the "make clean" does not remove them)
#

.SUFFIXES:
.SUFFIXES: .cpp .o

OBJ=	$(SRC:.cpp=.o)

.cpp.o:
	$(CC) $(CFLAGS) -c $<

all:	$(EXE)

$(OBJ):	$(INC)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)

tar:
	tar cvf cfd.tar $(MF) $(INC) $(SRC)

clean:
	rm -f $(OBJ) $(EXE) velocity.dat colourmap.dat cfd.plt core
