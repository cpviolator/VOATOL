#Your path to Eigen
<<<<<<< HEAD:IRAM/Makefile
EIGEN=../Eigen
OPENBLAS=../OpenBLAS-0.3.10
=======
EIGEN=../externals/eigen
EIGEN_COMPRESS=../externals/eigen-comp
>>>>>>> b760bdb14be0418e0d7d0edf8bcd685b3db95a96:Makefile
INCLUDE=../include

TARGET	 = iram 
SOURCES  = iram.cpp 
OBJS     = iram.o
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h lapack.h)
<<<<<<< HEAD:IRAM/Makefile
LIBS     = /usr/lib/x86_64-linux-gnu/libomp.so.5 ${OPENBLAS}/libopenblas.a -lpthread -lgfortran
#LIBS     = -L/usr/local/opt/libomp/lib -lomp -framework Accelerate /usr/local/Cellar/lapack/3.9.0_1/lib/liblapacke.dylib
=======
LIBS     = -L/usr/local/opt/libomp/lib -lomp 
>>>>>>> b760bdb14be0418e0d7d0edf8bcd685b3db95a96:Makefile

ERRS=-Wall

CXX = g++-9
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE} 
<<<<<<< HEAD:IRAM/Makefile
OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/lib/gcc/x86_64-linux-gnu/9/include
#OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
=======
OMPFLAGS = -Xpreprocessor -fopenmp 
>>>>>>> b760bdb14be0418e0d7d0edf8bcd685b3db95a96:Makefile
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: $(TARGET)

iram: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o iram $(OBJS) $(LIBS)

iram.o: iram.cpp ${INC}
	${CXX} ${CXXFLAGS} -c iram.cpp

ALL_SOURCES = Makefile $(SOURCES) $(INC)

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
