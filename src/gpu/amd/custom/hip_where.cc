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

// Kernel for where op. condition, input, other should be broadcastable, and the broadcasted shape is the same as output's shape.
__global__ void where_kernel_f32(char* condition, float* input, float* other, float* output, size_t* dim_c, size_t* dims_input, 
  size_t* dims_otherther, size_t* dims_b, size_t* stride_c, size_t* strides_input, size_t* strides_other, size_t* strides_b, 
  int num_dims, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int coordinate_b[MAX_DIM];  // broadcasted coordinate, also output coordinate
  int coordinate[MAX_DIM];
  
  __shared__ size_t s_dims_b[MAX_DIM]; // broadcasted shape

  __shared__ size_t s_stride_c[MAX_DIM];
  __shared__ size_t s_strides_input[MAX_DIM];
  __shared__ size_t s_strides_other[MAX_DIM];
  __shared__ size_t s_strides_b[MAX_DIM]; // broadcasted strides

  if(idx < num_dims)
  {
    s_dim_c[threadIdx.x] = dim_c[threadIdx.x];
    s_dims_input[threadIdx.x] = dims_input[threadIdx.x];
    s_dims_other[threadIdx.x] = dims_other[threadIdx.x];
    s_strides_c[threadIdx.x] = strides_c[threadIdx.x];
    s_strides_input[threadIdx.x] = strides_input[threadIdx.x];
    s_strides_other[threadIdx.x] = strides_other[threadIdx.x];
    s_strides_b[threadIdx.x] = strides_b[threadIdx.x];
  }

  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t idx_offset = idx + Scope * i;
    if(idx_offset >= total_elements)
      continue;

    // broadcast coordinate
    size_t margin = idx_offset;
    for(int j = 0; j < num_dims; j++)
    {
      coordinate_b[j] = margin / s_strides_b[j];
      margin = margin % s_strides_b[j];
    }

    // condition coordinate
    for(int j = 0; j < num_dims; j++)
    {
      // or s_dim_c[j] == 1, means we need broadcast
      if(coordinate_b[j] > s_dim_c[j]-1)
        coordinate[j] = 0;
      else
        coordinate[j] = coordinate_b[j];
    }
    // condition offset
    int c_offset = 0;
    for(int j=0; j<num_dims; j++){
      c_offset += coordinate[j]*s_stride_c[j];
    }
    // read condition
    char cond = condition[c_offset];

    // input coordinate
    for(int j = 0; j < num_dims; j++)
    {
      if(coordinate_b[j] > s_dims_input[j]-1)
        coordinate[j] = 0;
      else
        coordinate[j] = coordinate_b[j];
    }
    // input offset
    int i_offset = 0;
    for(int j=0; j<num_dims; j++){
      i_offset += coordinate[j]*s_stride_i[j];
    }
    // read input (positive value)
    float p_value = input[i_offset];

    // other coordinate
    for(int j = 0; j < num_dims; j++)
    {
      if(coordinate_b[j] > s_dims_otherther[j]-1)
        coordinate[j] = 0;
      else
        coordinate[j] = coordinate_b[j];
    }
    // other offset
    int other_offset = 0;
    for(int j=0; j<num_dims; j++){
      other_offset += coordinate[j]*s_strides_otherther[j];
    }
    // read other (nagetive value)
    float n_value = other[other_offset];

    float output_value = cond != 0 ? p_value : n_value;
    // write
    output[idx_offset] = output_value;
  }
}

// Kernel for where op. condition, input, other should be broadcastable, and the broadcasted shape is the same as output's shape.
__global__ void where_kernel_f16(char* condition, half* input, half* other, half* output, size_t* dim_c, size_t* dims_input, 
  size_t* dims_other, size_t* dims_b, size_t* stride_c, size_t* strides_input, size_t* strides_other, size_t* strides_b, 
  int num_dims, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int coordinate_b[MAX_DIM];  // broadcasted coordinate, also output coordinate
  int coordinate[MAX_DIM];
  
  __shared__ size_t s_dims_b[MAX_DIM]; // broadcasted shape

  __shared__ size_t s_dim_c[MAX_DIM];
  __shared__ size_t s_dims_input[MAX_DIM];
  __shared__ size_t s_dims_other[MAX_DIM];
  __shared__ size_t s_stride_c[MAX_DIM];
  __shared__ size_t s_strides_input[MAX_DIM];
  __shared__ size_t s_strides_other[MAX_DIM];
  __shared__ size_t s_strides_b[MAX_DIM]; // broadcasted strides

  if(idx < num_dims)
  {
    s_dim_c[threadIdx.x] = dim_c[threadIdx.x];
    s_dims_input[threadIdx.x] = dims_input[threadIdx.x];
    s_dims_other[threadIdx.x] = dims_other[threadIdx.x];
    s_strides_c[threadIdx.x] = strides_c[threadIdx.x];
    s_strides_input[threadIdx.x] = strides_input[threadIdx.x];
    s_strides_other[threadIdx.x] = strides_other[threadIdx.x];
    s_strides_b[threadIdx.x] = strides_b[threadIdx.x];
  }

  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t idx_offset = idx + Scope * i;
    if(idx_offset >= total_elements)
      continue;

    // broadcast coordinate
    size_t margin = idx_offset;
    for(int j = 0; j < num_dims; j++)
    {
      coordinate_b[j] = margin / s_strides_b[j];
      margin = margin % s_strides_b[j];
    }

    // condition coordinate
    for(int j = 0; j < num_dims; j++)
    {
      // or s_dim_c[j] == 1, means we need broadcast
      if(coordinate_b[j] > s_dim_c[j]-1)
        coordinate[j] = 0;
      else
        coordinate[j] = coordinate_b[j];
    }
    // condition offset
    int c_offset = 0;
    for(int j=0; j<num_dims; j++){
      c_offset += coordinate[j]*s_stride_c[j];
    }
    // read condition
    char cond = condition[c_offset];

    // input coordinate
    for(int j = 0; j < num_dims; j++)
    {
      if(coordinate_b[j] > s_dims_input[j]-1)
        coordinate[j] = 0;
      else
        coordinate[j] = coordinate_b[j];
    }
    // input offset
    int i_offset = 0;
    for(int j=0; j<num_dims; j++){
      i_offset += coordinate[j]*s_stride_i[j];
    }
    // read input (positive value)
    half p_value = input[i_offset];

    // other coordinate
    for(int j = 0; j < num_dims; j++)
    {
      if(coordinate_b[j] > s_dims_other[j]-1)
        coordinate[j] = 0;
      else
        coordinate[j] = coordinate_b[j];
    }
    // other offset
    int other_offset = 0;
    for(int j=0; j<num_dims; j++){
      other_offset += coordinate[j]*s_strides_other[j];
    }
    // read other (nagetive value)
    half n_value = other[other_offset];

    half output_value = cond != 0 ? p_value : n_value;
    // write
    output[idx_offset] = output_value;
  }
}

inline void prepare_utils(int num_dims, const size_t *dims_c, const size_t *dims_i, const size_t *dims_other, const size_t *dims_o,
  size_t **d_dims_c, size_t **d_dims_i, size_t **d_dims_other, size_t **d_dims_o, 
  size_t **d_strides_c, size_t **d_strides_i, size_t **d_strides_other, size_t **d_strides_o,
  size_t* total_elements, int* numBlocks)
{
  hipMalloc(d_dims_c, num_dims * sizeof(size_t));
  hipMalloc(d_dims_i, num_dims * sizeof(size_t));
  hipMalloc(d_dims_other, num_dims * sizeof(size_t));
  hipMalloc(d_dims_o, num_dims * sizeof(size_t));

  hipMemcpy(*d_dims_c, dims_c, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(*d_dims_i, dims_i, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(*d_dims_other, dims_other, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(*d_dims_o, dims_o, num_dims*sizeof(size_t), hipMemcpyHostToDevice);

  hipMalloc(d_strides_c, num_dims * sizeof(size_t));
  hipMalloc(d_strides_i, num_dims * sizeof(size_t));
  hipMalloc(d_strides_other, num_dims * sizeof(size_t));
  hipMalloc(d_strides_o, num_dims * sizeof(size_t));

  size_t* strides = (size_t*)malloc(num_dims * sizeof(size_t));
  // c
  strides[num_dims-1] = 1;
  for (int i = num_dims-2; i >= 0; i--)
    strides[i] = strides[i+1] * dims_c[i+1];
  hipMemcpy(*d_strides_c, strides, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  // i
  strides[num_dims-1] = 1;
  for (int i = num_dims-2; i >= 0; i--)
    strides[i] = strides[i+1] * dims_i[i+1];
  hipMemcpy(*d_strides_i, strides, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  // other
  strides[num_dims-1] = 1;
  for (int i = num_dims-2; i >= 0; i--)
    strides[i] = strides[i+1] * dims_other[i+1];
  hipMemcpy(*d_strides_other, strides, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  // o
  strides[num_dims-1] = 1;
  for (int i = num_dims-2; i >= 0; i--)
    strides[i] = strides[i+1] * dims_o[i+1];
  hipMemcpy(*d_strides_o, strides, num_dims*sizeof(size_t), hipMemcpyHostToDevice);

  *total_elements = strides[0] * dims_o[0];

  // ceiling div
  *numBlocks = *total_elements / blockSize;
  if(*total_elements % blockSize != 0)
    *numBlocks += 1;
  
  free(strides);
}

namespace hip_custom {

void where(void *condition, void *input, void *other, void *output, 
  const size_t *dims_c, const size_t *dims_i, const size_t *dims_other, const size_t *dims_o, 
  int num_dims, int dtype)
{
  size_t *d_dims_c;
  size_t *d_dims_i;
  size_t *d_dims_other;
  size_t *d_dims_o;

  size_t *d_strides_c;
  size_t *d_strides_i;
  size_t *d_strides_other;
  size_t *d_strides_o;

  size_t total_elements;
  int numBlocks;

  prepare_utils(num_dims, dims_c, dims_i, dims_other, dims_o, 
    &d_dims_c, &d_dims_i, &d_dims_other, &d_dims_o,
    &d_strides_c, &d_strides_i, &d_strides_other, &d_strides_o, 
    &total_elements, &numBlocks);
  
  if(dtype == 0)
    hipLaunchKernelGGL(where_kernel_f32, dim3(numBlocks), dim3(blockSize), 0, 0, 
      (char*)condition, (float*)input, (float*)other, (float*)output, 
      d_dims_c, d_dims_i, d_dims_other, d_dims_o, 
      d_strides_c, d_strides_i, d_strides_other, d_strides_o, 
      num_dims, total_elements);
  if(dtype == 1)
    hipLaunchKernelGGL(where_kernel_f16, dim3(numBlocks), dim3(blockSize), 0, 0, 
      (char*)condition, (half*)input, (half*)other, (half*)output, 
      d_dims_c, d_dims_i, d_dims_other, d_dims_o, 
      d_strides_c, d_strides_i, d_strides_other, d_strides_o, 
      num_dims, total_elements);

  hipFree(d_dims_c);
  hipFree(d_dims_i);
  hipFree(d_dims_other);
  hipFree(d_dims_o);
  hipFree(d_strides_c);
  hipFree(d_strides_i);
  hipFree(d_strides_other);
  hipFree(d_strides_o);
}

} // hip_custom

