#Your path to Eigen
EIGEN=/Users/deanhowarth/LaptopLanczos/Eigen
INCLUDE=../include

TARGET	 = lanczos 
SOURCES  = lanczos.cpp 
OBJS     = lanczos.o
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)
LIBS     = -L/usr/local/opt/libomp/lib -lomp

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}
OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: $(TARGET)

lanczos: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o lanczos $(OBJS) $(LIBS)

lanczos.o: lanczos.cpp ${INC}
	${CXX} ${CXXFLAGS} -c lanczos.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
