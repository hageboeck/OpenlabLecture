all: helloWorld vectorAdd julia

ifeq ($(wildcard ${CPLUS_INCLUDE_PATH}/boost/gil),) 
    HAVE_GIL =
else 
    HAVE_GIL = -DUSE_BOOST_GIL -lpng
endif 

%: %.cu
	nvcc -arch sm_80 -std=c++17 -Xcompiler -Wall -g -O2 $< -o $@

julia: julia.cu *.h
	nvcc -arch sm_80 -std=c++17 -Xcompiler -Wall -g -O2 $< -o $@ ${HAVE_GIL}
%/julia: %/julia.cu *.h
	nvcc -arch sm_80 -std=c++17 -Xcompiler -Wall -g -O2 $< -o $@ ${HAVE_GIL}

clean:
	rm -f helloWorld julia{,.png,.ppm} vectorAdd solution/julia{,.ppm,.png}