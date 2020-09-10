#Your path to Eigen
EIGEN=../Eigen
INCLUDE=../include

#Your path to ARPACK
ARPACK_LIB='/scratch/CPviolator/ARPACK_CLEAN/libarpack.a'

TARGET	 = arpack
SOURCES  = arpack_interface.cpp
OBJS     = arpack_interface.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h arpack_interface.h)
LIBS     = ${ARPACK_LIB}

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE} -I/usr/include -fPIC

#============================================================

all: $(TARGET)

arpack: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o arpack ${OBJS} ${ARPACK_LIB} -lgfortran

arpack_interface.o: arpack_interface.cpp ${INC}
	${CXX} ${CXXFLAGS} -c arpack_interface.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
