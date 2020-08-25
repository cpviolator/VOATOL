#Your path to Eigen
EIGEN=/Users/deanhowarth/LaptopLanczos/Eigen
INCLUDE=../include

TARGET	 = arnoldi 
SOURCES  = arnoldi.cpp 
OBJS     = arnoldi.o
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)
LIBS     = -L/usr/local/opt/libomp/lib -lomp

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}
OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: $(TARGET)

arnoldi: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o arnoldi $(OBJS) $(LIBS)

arnoldi.o: arnoldi.cpp ${INC}
	${CXX} ${CXXFLAGS} -c arnoldi.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
