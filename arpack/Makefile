#Your path to Eigen
EIGEN=../../../Eigen
INCLUDE=../include

#Your path to ARPACK
ARPACK_LIB='/scratch/CPviolator/ARPACK_SILENT/ARPACK/libarpack.a'

TARGET	 = arpack
SOURCES  = arpack_interface.cpp
OBJS     = arpack_interface.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h arpack_interface.h) #/Users/deanhowarth/ARPACK_NG/install/include
#LIBS     = /usr/local/lib/libmkl_intel.dylib /usr/local/lib/libmkl_intel_thread.dylib /usr/local/lib/libmkl_core.dylib /usr/local/lib/libiomp5.dylib ${ARPACK_LIB} 


ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE} -I/usr/include -fPIC

#============================================================

all: $(TARGET)

arpack: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o arpack ${OBJS} ${LIBS} ${ARPACK_LIB} -L/usr/local/gfortran/lib/ -lgfortran 

arpack_interface.o: arpack_interface.cpp ${INC}
	${CXX} ${CXXFLAGS} -c arpack_interface.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
