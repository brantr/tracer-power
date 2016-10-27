EXEC   = athena4_binary.tracer_power

OBJS   = main.o grid_fft.o grid_pk.o rng.o read_athena_tracers.o read_athena_header.o

INCL   = grid_fft.h grid_pk.h rng.h read_athena_tracers.hpp read_athena_header.hpp

LIBS     = -lgsl -lgslcblas -lmpi -lfftw3_mpi -lfftw3 -lm -stdlib=libstdc++

CC       = mpicxx
CXX      = mpicxx
CFLAGS   = -fopenmp -stdlib=libstdc++
CPPFLAGS = -stdlib=libstdc++


$(EXEC): $(OBJS) 
	 $(CXX) $(OBJS) $(LIBS) -o $(EXEC)   
         

$(OBJS): $(INCL) 

.PHONY : clean

clean:
	 rm -f $(OBJS) $(EXEC)

