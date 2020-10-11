#Your path to Eigen
EIGEN=../externals/eigen
EIGEN_COMPRESS=../externals/eigen-comp
ARPACK=../externals/arpack-ng_install
INCLUDE=../include
LIB=../lib

SOURCES  = iram.cpp trlm.cpp blktrlm.cpp
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h lapack.h arpack_interface.h)
#LIBS     = -L/usr/local/opt/libomp/lib -lomp 
LGFORTRAN=/usr/lib/gcc/x86_64-linux-gnu/9/libgfortran.a
LARPACK  =${ARPACK}/lib/libarpack.a

ERRS=-Wall

CXX=g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}
#OMPFLAGS = -Xpreprocessor -fopenmp
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: iram trlm blktrlm arpack

iram: iram.o
	$(CXX) $(CXXFLAGS) -o iram_exe iram.o $(LIBS) 

iram.o: iram.cpp ${INC} 
	${CXX} ${CXXFLAGS} -c iram.cpp

trlm: trlm.o
	$(CXX) $(CXXFLAGS) -o trlm_exe trlm.o $(LIBS)

trlm.o: trlm.cpp ${INC} 
	${CXX} ${CXXFLAGS} -c trlm.cpp

blktrlm: blktrlm.o
	$(CXX) $(CXXFLAGS) -o blktrlm_exe blktrlm.o $(LIBS)

blktrlm.o: blktrlm.cpp ${INC} 
	${CXX} ${CXXFLAGS} -c blktrlm.cpp

arpack: arpack.o
	$(CXX) $(CXXFLAGS) -o arpack_exe arpack.o $(LIBS) ${LARPACK} ${LIBGFORTRAN}

arpack.o: arpack.cpp ${INC} 
	${CXX} ${CXXFLAGS} -c arpack.cpp

clean:
	rm -f *.o *_exe 

