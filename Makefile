NVCCFLAGS = -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcusolver -lcudart -lcublas -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -Xcompiler -fopenmp -lineinfo -O3 -std=c++17 `pkg-config --cflags --libs opencv4`


main: Else.cu Excuse.cu ElseGreedyI.cu ElseGreedyII.cu ExcuseGreedyI.cu ExcuseGreedyII.cu main.cpp 
	nvcc $(NVCCFLAGS) Else.cu Excuse.cu ElseGreedyI.cu ElseGreedyII.cu ExcuseGreedyI.cu ExcuseGreedyII.cu main.cpp -o build/pupil_tracking


clean:
	rm -rf build/*