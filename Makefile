INCLUDES := -I/usr/local/cuda/include
LIBRARIES := -L/usr/local/cuda/lib64 -lOpenCL
COMPILER ?= g++

all: build

build: oclMatrixMul

matrixMul_gold.o: matrixMul_gold.cpp oclMatrixMul.cpp
	$(COMPILER) $(INCLUDES) -o $@ -c $<
oclMatrixMul.o: oclMatrixMul.cpp
	$(COMPILER) $(INCLUDES) -o $@ -c $<
oclMatrixMul: matrixMul_gold.o oclMatrixMul.o
	$(COMPILER) -o $@ $+ $(LIBRARIES)

clean:
	rm -f oclMatrixMul.o matrixMul_gold.o oclMatrixMul
