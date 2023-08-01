all: helloWorld vectorAdd julia

%: %.cu
	nvcc -std=c++17 -Xcompiler -Wall -g -O2 $< -o $@
