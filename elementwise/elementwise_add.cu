#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// transfer vector
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void add(float* a, float* b, float* c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
__global__ void vec2_add(float* a, float* b, float* c)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*2;
    //c[idx] = a[idx] + b[idx];
    float2 reg_a = FETCH_FLOAT2(a[idx]);
    float2 reg_b = FETCH_FLOAT2(b[idx]);
    float2 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    FETCH_FLOAT2(c[idx]) = reg_c;
}

__global__ void vec4_add(float* a, float* b, float* c)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*4;
    //c[idx] = a[idx] + b[idx];
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(c[idx]) = reg_c;
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    const int N=32*1024*1024;
    float *a=(float *)malloc(N*sizeof(float));
    float *b=(float *)malloc(N*sizeof(float));
    float *out=(float *)malloc(N*sizeof(float));
    float *d_a;
    float *d_b;
    float *d_out;
    cudaMalloc((void **)&d_a,N*sizeof(float));
    cudaMalloc((void **)&d_b,N*sizeof(float));
    cudaMalloc((void **)&d_out,N*sizeof(float));
    float *res=(float *)malloc(N*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1;
        b[i]=i;
        res[i]=a[i]+b[i];
    }

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( N/THREAD_PER_BLOCK/4, 1);
    dim3 Block( THREAD_PER_BLOCK, 1);

    int iter = 2000;
    for(int i=0; i<iter; i++){
        vec4_add<<<Grid,Block>>>(d_a, d_b, d_out);
    }

    cudaMemcpy(out,d_out,N*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,N))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<N;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}
