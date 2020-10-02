#Your path to Eigen
EIGEN=../../Eigen
INCLUDE=../include

TARGET	 = iram 
SOURCES  = iram.cpp 
OBJS     = iram.o
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h lapack.h)
LIBS     = -L/usr/local/opt/libomp/lib -lomp 

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE} 
OMPFLAGS = -Xpreprocessor -fopenmp 
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
