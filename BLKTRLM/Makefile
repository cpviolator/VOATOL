#Your path to Eigen
EIGEN=/Users/deanhowarth/LaptopLanczos/Eigen
INCLUDE=../include

TARGET	 = block_trlm
SOURCES  = block_trlm.cpp 
OBJS     = block_trlm.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)
LIBS     = -L/usr/local/opt/libomp/lib -lomp

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}
#OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: $(TARGET)

block_trlm: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o block_trlm $(OBJS) $(LIBS)

block_trlm.o: block_trlm.cpp ${INC}
	${CXX} ${CXXFLAGS} -c block_trlm.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
