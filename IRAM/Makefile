#Your path to Eigen
EIGEN=../Eigen
OPENBLAS=../OpenBLAS-0.3.10
INCLUDE=../include

TARGET	 = iram 
SOURCES  = iram.cpp 
OBJS     = iram.o
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h lapack.h)
LIBS     = /usr/lib/x86_64-linux-gnu/libomp.so.5 ${OPENBLAS}/libopenblas.a -lpthread -lgfortran
#LIBS     = -L/usr/local/opt/libomp/lib -lomp -framework Accelerate /usr/local/Cellar/lapack/3.9.0_1/lib/liblapacke.dylib

ERRS=-Wall

CXX = g++-9
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE} 
OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/lib/gcc/x86_64-linux-gnu/9/include
#OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: $(TARGET)

iram: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o iram $(OBJS) $(LIBS)

iram.o: iram.cpp ${INC}
	${CXX} ${CXXFLAGS} -c iram.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
