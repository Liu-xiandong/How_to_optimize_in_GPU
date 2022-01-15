# How to optimize code in GPU
This is a series of GPU optimization topics. Here we will introduce  how to optimize the program on the GPU in detail. The reduce optimization has been completed. The optimization of GEMM has completed the CUDA C code. The assembler is currently being used to tune the program, and the code will be issued later.

# reduce
Seven optimization methods were used to optimize reduce operator, and the performance of different optimization methods was tested on V100. Finally, the bandwidth of 770GB/s is obtained, and the bandwidth utilization rate is: 770/900=85.6%. The figure below shows the performance of the 7 optimization techniques.
![](https://github.com/writerblack/test/blob/master/reduce8.png?raw=true)

# gemm
Optimized for Sgemm, it can reach 99.5% of cublas performance on 4096 matrices (M=N=K).

## License
All the source codes of this repo are released under [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).
