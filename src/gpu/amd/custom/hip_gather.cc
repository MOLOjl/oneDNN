/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <hip/hip_runtime.h>
#include "hip_customs.h"
#include <hip/hip_fp16.h>

#define MAX_DIM 8
#define blockSize 256

// Kernel for gather op. input, output and index should all have the same number of dimensions, no broadcast.
__global__ void gather_kernel_f32(float* input, float* output, int* index, size_t* dims, size_t* strides, int gather_dim, int num_dims, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int coordinate[MAX_DIM];
  
  __shared__ size_t s_strides[MAX_DIM];
  __shared__ size_t s_dims[MAX_DIM];

  if(idx < num_dims)
  {
    s_strides[threadIdx.x] = strides[threadIdx.x];
    s_dims[threadIdx.x] = dims[threadIdx.x];
  }

  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t idx_offset = idx + Scope * i;
    if(idx_offset >= total_elements)
      continue;
    
    // read
    int gather_idx = index[idx_offset];

    // colculate coordinate of element.
    size_t margin = idx_offset;
    for(int j = 0; j < num_dims; j++)
    {
      coordinate[j] = margin / s_strides[j];
      margin = margin % s_strides[j];
    }
    
    coordinate[gather_dim] = gather_idx;

    // output offset
    int input_offset = 0;
    for(int j=0; j<num_dims; j++){
      input_offset += coordinate[j]*strides[j];
    }

    float gather_value = input[input_offset];
    output[idx_offset] = gather_value;
  }
}

// Kernel to mask the input, can be done in place. the mask tensor shall be broadcast to the input tensor.
__global__ void gather_kernel_f16(half* input, half* output, int* index, size_t* dims, size_t* strides, int gather_dim, int num_dims, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int coordinate[MAX_DIM];
  
  __shared__ size_t s_strides[MAX_DIM];
  __shared__ size_t s_dims[MAX_DIM];

  if(idx < num_dims)
  {
    s_strides[threadIdx.x] = strides[threadIdx.x];
    s_dims[threadIdx.x] = dims[threadIdx.x];
  }

  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t idx_offset = idx + Scope * i;
    if(idx_offset >= total_elements)
      continue;
    
    int gather_idx = index[idx_offset];

    // colculate coordinate of element.
    size_t margin = idx_offset;
    for(int j = 0; j < num_dims; j++)
    {
      coordinate[j] = margin / s_strides[j];
      margin = margin % s_strides[j];
    }
    
    coordinate[gather_dim] = gather_idx;

    // output offset
    int input_offset = 0;
    for(int j=0; j<num_dims; j++){
      input_offset += coordinate[j]*strides[j];
    }

    half gather_value = input[input_offset];
    output[idx_offset] = gather_value;
  }
}

inline void prepare_utils(int num_dims, const size_t *dims, size_t **d_strides, size_t **d_dims, size_t* total_elements, int* numBlocks){
  size_t* strides = (size_t*)malloc(num_dims * sizeof(size_t));
  strides[num_dims-1] = 1;
  for (int i = num_dims-2; i >= 0; i--)
    strides[i] = strides[i+1] * dims[i+1];
  
  hipMalloc(d_strides, num_dims * sizeof(size_t));
  hipMalloc(d_dims, num_dims * sizeof(size_t));
  hipMemcpy(*d_strides, strides, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(*d_dims, dims, num_dims*sizeof(size_t), hipMemcpyHostToDevice);

  *total_elements = strides[0] * dims[0];

  // ceiling div
  *numBlocks = *total_elements / blockSize;
  if(*total_elements % blockSize != 0)
    *numBlocks += 1;
  
  free(strides);
}

namespace hip_custom {

// float tensor mask
void gather(void *input, void *output, void *index, const size_t *dims, int num_dims, int gather_dim, int dtype)
{
  size_t *d_strides;
  size_t *d_dims;
  size_t total_elements;
  int numBlocks;

  prepare_utils(num_dims, dims, &d_strides, &d_dims, &total_elements, &numBlocks);
  
  if(dtype == 0)
    hipLaunchKernelGGL(gather_kernel_f32, dim3(numBlocks), dim3(blockSize), 0, 0, (float*)input, (float*)output, (int*)index, d_dims, d_strides, gather_dim, num_dims, total_elements);
  if(dtype == 1)
    hipLaunchKernelGGL(gather_kernel_f16, dim3(numBlocks), dim3(blockSize), 0, 0, (half*)input, (half*)output, (int*)index, d_dims, d_strides, gather_dim, num_dims, total_elements);

  hipFree(d_strides);
  hipFree(d_dims);
}

} // hip_custom

