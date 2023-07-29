#include <bits/stdc++.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime_api.h> 
#include <cuda_runtime.h>
#include <cusparse.h> 
#include "sputnik/sputnik.h"

using namespace std;

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void ReadFile(std::string &file, int &row_num, int &col_num, int &nnz_num,
            std::vector<int> &A_row_offset, std::vector<int> &A_col_index,
            std::vector<float> &A_value, std::vector<float> &B,
            std::vector<float> &C)
{
    std::ifstream input;
    input.open("matrix/" + file + ".smtx");

    while (input.peek() == '%')
        input.ignore(2048, '\n');
    
    std::string s;
    getline(input, s);
    // parse the first line
    std::stringstream s_stream(s);
    std::string current_str;
    getline(s_stream, current_str, ',');
    row_num = atoi(current_str.c_str());
    getline(s_stream, current_str, ',');
    col_num = atoi(current_str.c_str());
    getline(s_stream, current_str, ',');
    nnz_num = atoi(current_str.c_str());

    A_row_offset.resize(row_num+1);
    A_col_index.resize(nnz_num);
    A_value.resize(nnz_num);
    for(int i=0; i<row_num+1; i++){
        input >> A_row_offset[i];
    }
    for(int i=0; i<nnz_num; i++){
        input >> A_col_index[i];
    }
    input.close();
    B.resize(col_num * row_num);
    C.resize(row_num * row_num);

    // init A
    for(int i=0; i<A_value.size(); i++){
        A_value[i]=i%17;
    }

    // init B
    for(int i=0; i<B.size(); i++){
        B[i]=i%13;
    }
}

template<typename T>
void vec_print(std::vector<T> array){
    for(auto x: array){
        cout<<x<<" ";
    }
    cout<<std::endl;
}

template <typename IndexType, typename ValueType>
void spmm_cpu_kernel(std::vector<IndexType> &row_offset,
                std::vector<IndexType> &col_index,
                std::vector<ValueType> &value,
                std::vector<ValueType> &B,
                std::vector<ValueType> &C,
                IndexType row_num,
                IndexType col_num)
{
    for(int i=0; i<row_num; i++){
        for(int j=0; j<row_num; j++){
            ValueType res = 0;
            IndexType num = row_offset[i+1] - row_offset[i];
            for(int k=0; k<num; k++){
                IndexType index = row_offset[i] + k;
                IndexType current_col = col_index[index];
                res += value[index]* B[current_col*row_num + j];
            }
            C[i*row_num+j] = res;
        }
    }
}

// dim3 dimBlock(THREAD_NUM_PER_BLOCK);
// dim3 dimGrid(row_num/THREAD_NUM_PER_BLOCK, row_num);
template <unsigned int THREAD_NUM_PER_BLOCK>
__global__ void My_spmm_csr_vector_kernel_v0(const int num_rows,
    const int * A_row_offset,
    const int * A_col_index,
    const float * A_value,
    const float * B,
    float * C,
    const int ldb,
    const int ldc){
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;

    // matrix C row_index
    int C_row_index = by;
    int C_col_index = bx * THREAD_NUM_PER_BLOCK + tx;

    if(C_row_index < num_rows && C_col_index < ldc){
        int row_start = A_row_offset[C_row_index];
        int row_end = A_row_offset[C_row_index + 1];
        int iter_num = row_end - row_start;
        float sum = 0.0;
        for(int i=0; i<iter_num; i++){
            int index = row_start + i;
            int current_col = A_col_index[index];
            float current_val = A_value[index];
            float reg_B = B[ current_col * ldb + C_col_index];
            sum += current_val * reg_B;
        }

        C[C_row_index * ldc + C_col_index] = sum;
    }
}

// dim3 dimBlock(THREAD_NUM_PER_BLOCK);
// dim3 dimGrid(row_num/THREAD_NUM_PER_BLOCK, row_num);
// useless optimize
template <
    const int BLOCK_SIZE_X,   
    const int BLOCK_SIZE_K,
    const int THREAD_NUM_PER_BLOCK
    > 
__global__ void My_spmm_csr_vector_kernel_v1(const int num_rows,
    const int * A_row_offset,
    const int * A_col_index,
    const float * A_value,
    const float * B,
    float * C,
    const int M,
    const int N,
    const int K){
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;

    // matrix C row_index
    int C_row_index = by;
    int C_col_index = bx * THREAD_NUM_PER_BLOCK + tx;

    // shared mem for A 
    __shared__ int As_col[BLOCK_SIZE_K];
    __shared__ float As_value[BLOCK_SIZE_K];

    int NUM_A_PER_THREAD = BLOCK_SIZE_K/THREAD_NUM_PER_BLOCK;

    if(C_row_index < num_rows && C_col_index < N){
        int row_start = A_row_offset[C_row_index];
        int row_end = A_row_offset[C_row_index + 1];
        int iter_num = row_end - row_start;
        float sum = 0.0;

        for(int k=0; k<iter_num; k+=BLOCK_SIZE_K){
            // store A to shared mem
            int global_index = row_start + k*BLOCK_SIZE_K;
            int local_index = NUM_A_PER_THREAD * tx;
            for(int i=0; i< NUM_A_PER_THREAD; i++){
                if(global_index + local_index + i < row_end){
                    As_col[local_index + i] = A_col_index[global_index + local_index +i];
                    As_value[local_index + i] = A_value[global_index + local_index +i];
                }
                else{
                    As_col[local_index + i] = -1;
                    As_value[local_index + i] = 0.0;
                }
            }
            __syncthreads();
            // load A from shared mem
            for(int i=0; i< BLOCK_SIZE_K; i++){
                int current_col = As_col[i];
                float current_val = As_value[i];
                if(current_col != -1){
                    float reg_B = B[ current_col * N + C_col_index];
                    sum += current_val * reg_B;
                }
            }
        }

        C[C_row_index * N + C_col_index] = sum;
    }
}


// A(row_num,col_num)
// B(col_num,row_num)
// C(row_num,row_num)
int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("usage: ./spmm -f [matrix]\n");
        exit(0);
    }
    string file;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
            file = argv[i + 1];
        }
    }

    // load csr data from .smtx file
    int row_num = 0;
    int col_num = 0;
    int nnz_num = 0;
    std::vector<int> A_row_offset;
    std::vector<int> A_col_index;
    std::vector<float> A_value;
    std::vector<float> B;
    std::vector<float> C;
    ReadFile(file, row_num, col_num, nnz_num, A_row_offset, A_col_index, A_value, B, C);
    std::vector<float> C_cusparse(C.size());

    // used in sputnik
    // TODO: it's useless?
    std::vector<int> row_indices(row_num);
    // init row_indices
    for(int i=0; i<row_num; i++){
        row_indices[i] = A_row_offset[i+1] - A_row_offset[i];
    }

    //debug case
    /*
    int row_num = 4;
    int col_num = 4;
    int nnz_num = 9;
    int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f };
    float hB[]            = { 1.0f, 5.0f, 9.0f, 0.0f,
                              2.0f, 6.0f, 10.0f, 0.0f,
                              3.0f, 7.0f, 11.0f, 0.0f,
                              4.0f, 8.0f, 12.0f, 0.0f};
    std::vector<int> A_row_offset(hA_csrOffsets, hA_csrOffsets + sizeof(hA_csrOffsets));
    std::vector<int> A_col_index(hA_columns, hA_columns + sizeof(hA_columns));
    std::vector<float> A_value(hA_values, hA_values + sizeof(hA_values));
    std::vector<float> B(hB, hB + sizeof(hB));
    std::vector<float> C(16, 0);
    std::vector<float> C_cusparse(16, 0);
    */

    // check input
    std::cout<<"The row_num is:" <<row_num <<std::endl;
    std::cout<<"The col_num is:" <<col_num <<std::endl;
    std::cout<<"The nnz_num is:" <<nnz_num <<std::endl;

    // allocate memory in GPU device
    int* d_A_row_offset;
    int* d_A_col_index;
    float* d_A_value;
    float* d_B;
    float* d_C;
    float* d_C_cusparse;
    int* d_row_indices;
    int B_num = B.size();
    int C_num = C.size();

    checkCudaErrors(cudaMalloc(&d_A_row_offset, (row_num + 1)*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_A_col_index, nnz_num*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_A_value, nnz_num*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, B_num*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, C_num*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C_cusparse, C_num*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_row_indices, row_num*sizeof(int)));
    checkCudaErrors(cudaMemcpy( d_A_row_offset, A_row_offset.data(), (row_num + 1)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_A_col_index, A_col_index.data(), nnz_num*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_A_value, A_value.data(), nnz_num*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, B.data(), B_num*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_row_indices, row_indices.data(), row_num*sizeof(int), cudaMemcpyHostToDevice));
    
    int iter = 2000;
    // My spmm
    // cpu version
    // spmm_cpu_kernel<int,float>(A_row_offset, A_col_index, A_value, B, C, row_num, col_num);

    constexpr unsigned int THREAD_NUM_PER_BLOCK  = 128;
    
    dim3 dimBlock(THREAD_NUM_PER_BLOCK);
    dim3 dimGrid(row_num/THREAD_NUM_PER_BLOCK, row_num);

    for(int i=0; i<iter; i++){
        My_spmm_csr_vector_kernel<128, 512, THREAD_NUM_PER_BLOCK> <<< dimGrid, dimBlock >>> 
            (row_num, d_A_row_offset, d_A_col_index, d_A_value, d_B, d_C, row_num, row_num, col_num);
    }
    //checkCudaErrors(cudaMemcpy(C.data(), d_C, C_num*sizeof(float), cudaMemcpyDeviceToHost));

    // sputnik
    cudaStream_t s0 = 0;
    for(int i=0; i<iter; i++){
        sputnik::CudaSpmm(row_num, row_num, col_num, 
                            nnz_num, d_row_indices, 
                            d_A_value, d_A_row_offset, d_A_col_index, 
                            d_B, d_C, s0);
    }
    cudaStreamSynchronize(s0);
    checkCudaErrors(cudaMemcpy(C.data(), d_C, C_num * sizeof(float), cudaMemcpyDeviceToHost));
    

    // cusparse spmm
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    int ldb = row_num;
    int ldc = row_num;
    float alpha           = 1.0f;
    float beta            = 0.0f;
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, row_num, col_num, nnz_num,
                                      d_A_row_offset, d_A_col_index, d_A_value,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, col_num, row_num, ldb, d_B,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, row_num, row_num, ldc, d_C_cusparse,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    for(int i=0; i<iter; i++){
        CHECK_CUSPARSE( cusparseSpMM(handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    }
    
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(C_cusparse.data(), d_C_cusparse, C_num * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    bool check_result = true;
    for(int i=0; i<C.size(); i++){
        if(fabs(C[i]-C_cusparse[i])>1e-6){
            std::cout<<"The result is error!"<<std::endl;
            printf("The error case is (%d %d %f %f)\n", i/row_num, i%row_num, C[i], C_cusparse[i]);
            check_result = false;
            break;
        }
    }
    if(check_result){
        std::cout<<"The result is right!"<<std::endl;
    }

    // Free Memory
    cudaFree(d_A_row_offset);
    cudaFree(d_A_col_index);
    cudaFree(d_A_value);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_cusparse);

    return 0;
}