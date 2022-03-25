# How to optimize in GPU
This is a series of GPU optimization topics. Here we will introduce  how to optimize the program on the GPU in detail. Here, I will introduce several basic kernel optimizations, including: elementwise, reduce, sgemv, sgemm, etc. All the following performance data are run on V100 and tested by nsight. 

If you have any questions, you can directly contact: xiandong_liu@foxmail.com

# reduce
Seven optimization methods were used to optimize reduce operator, and the performance of different optimization methods was tested on V100. The bandwidth of 858GB/s is obtained, and the bandwidth utilization rate is: 858/900=95.3%. The figure below shows the performance of the 7 optimization techniques.
![](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/figure/reduce.png?raw=true)

# sgemm
The optimization of sgemm is divided into two levels, namely CUDA
C-level optimization and optimization of SASS code.

Regarding CUDA C-level optimizations, the final code is sgemm_v3. On a large matrix of 4096 (M=N=K), our sgemm can achieve 96.8% performance of cublas, with a peak floating point efficiency of 93.6%, basically reaching the limit of CUDA C code optimization. The figure below shows the performance of sgemm_v3.
![](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/figure/sgemm.png?raw=true)

Regarding the optimization of SASS code, CuAssembler is used for tuning. There are two main optimization techniques in this part, namely register remapping and instruction rearrangement to obtain better .reuse logo layout.

## License
All the source codes of this repo are released under [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).
