all: helloWorld vectorAdd julia

ifeq ($(wildcard ${CPLUS_INCLUDE_PATH}/boost/gil),) 
    HAVE_GIL =
else 
    HAVE_GIL = -DUSE_BOOST_GIL -lpng
endif 

%: %.cu
	nvcc -std=c++17 -Xcompiler -Wall -g -O2 $< -o $@

julia: julia.cu *.h
	nvcc -std=c++17 -Xcompiler -Wall -g -O2 $< -o $@ ${HAVE_GIL}