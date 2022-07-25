all: helloWorld vectorAdd julia

%: %.cu
	nvcc -std=c++17 -g -O2 $< -o $@
