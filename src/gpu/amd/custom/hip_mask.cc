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

// Kernel to mask the input, can be done in place. the mask tensor shall be broadcast to the input tensor.
__global__ void mask_kernel_f32(float* input, float* output, char* mask, size_t* stride_io, size_t* dims_mask, int num_dims, size_t total_elements, float masked_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int mask_coordinate[MAX_DIM];
  
  __shared__ size_t s_stride_io[MAX_DIM];
  __shared__ size_t s_dims_mask[MAX_DIM];
  __shared__ size_t s_stride_mask[MAX_DIM];

  if(idx < num_dims)
  {
    s_stride_io[threadIdx.x] = stride_io[threadIdx.x];
    s_dims_mask[threadIdx.x] = dims_mask[threadIdx.x];
  }

  if(idx == 0)
  {
    s_stride_mask[num_dims-1] = 1;
    for(int i=num_dims-2; i >= 0; i--)
      s_stride_mask[i] = s_stride_mask[i+1] * s_dims_mask[i+1];
  }

  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t io_offset = idx + Scope * i;
    if(io_offset >= total_elements)
      continue;
    
    // read
    float value = input[io_offset];
    
    // colculate mask_coordinate of element.
    size_t margin = io_offset;
    for(int j = 0; j < num_dims; j++)
    {
      mask_coordinate[j] = margin / s_stride_io[j];
      if(s_dims_mask[j] == 1)
        mask_coordinate[j] = 1;
      margin = margin % s_stride_io[j];
    }

    // broadcast mask offset
    size_t mask_offset = 0;
    for(int j = 0; j < num_dims; j++)
    {
      mask_offset += s_stride_mask[j] * mask_coordinate[j];
    }

    char mask_value = mask[mask_offset];
    // Fills elements of input tensor with value where mask is True
    if(mask_value != 0)
      output[io_offset] = masked_value;
    // if not inplace and mask is False, copy input to output.
    if(input != output && mask_value == 0)
      output[io_offset] = input[io_offset];
  }
}

// Kernel to mask the input, can be done in place. the mask tensor shall be broadcast to the input tensor.
__global__ void mask_kernel_f16(half* input, half* output, char* mask, size_t* stride_io, size_t* dims_mask, int num_dims, size_t total_elements, half masked_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int mask_coordinate[MAX_DIM];
  
  __shared__ size_t s_stride_io[MAX_DIM];
  __shared__ size_t s_dims_mask[MAX_DIM];
  __shared__ size_t s_stride_mask[MAX_DIM];

  if(idx < num_dims)
  {
    s_stride_io[threadIdx.x] = stride_io[threadIdx.x];
    s_dims_mask[threadIdx.x] = dims_mask[threadIdx.x];
  }

  if(idx == 0)
  {
    s_stride_mask[num_dims-1] = 1;
    for(int i=num_dims-2; i >= 0; i--)
      s_stride_mask[i] = s_stride_mask[i+1] * s_dims_mask[i+1];
  }

  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t io_offset = idx + Scope * i;
    if(io_offset >= total_elements)
      continue;
    
    // read
    half value = input[io_offset];
    
    // colculate mask_coordinate of element.
    size_t margin = io_offset;
    for(int j = 0; j < num_dims; j++)
    {
      mask_coordinate[j] = margin / s_stride_io[j];
      if(s_dims_mask[j] == 1)
        mask_coordinate[j] = 1;
      margin = margin % s_stride_io[j];
    }

    // broadcast mask offset
    size_t mask_offset = 0;
    for(int j = 0; j < num_dims; j++)
    {
      mask_offset += s_stride_mask[j] * mask_coordinate[j];
    }

    char mask_value = mask[mask_offset];
    // Fills elements of input tensor with value where mask is True
    if(mask_value != 0)
      output[io_offset] = __half2float(masked_value);
    // if not inplace and mask is False, copy input to output.
    if(input != output && mask_value == 0)
      output[io_offset] = input[io_offset];
  }
}


inline void prepare_utils(int num_dims, const size_t *dims, const size_t *dims_mask, size_t **d_io_strides, size_t **d_dims_mask, size_t* total_elements, int* numBlocks){
  size_t* io_strides = (size_t*)malloc(num_dims * sizeof(size_t));
  io_strides[num_dims-1] = 1;

  for (int i = num_dims-2; i >= 0; i--)
    io_strides[i] = io_strides[i+1] * dims[i+1];
  
  hipMalloc(d_io_strides, num_dims * sizeof(size_t));
  hipMalloc(d_dims_mask, num_dims * sizeof(size_t));
  hipMemcpy(*d_io_strides, io_strides, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(*d_dims_mask, dims_mask, num_dims*sizeof(size_t), hipMemcpyHostToDevice);

  *total_elements = io_strides[0] * dims[0];

  // ceiling div
  *numBlocks = *total_elements / blockSize;
  if(*total_elements % blockSize != 0)
    *numBlocks += 1;
  
  free(io_strides);
}

namespace hip_custom {

// float tensor mask
void mask(void *input, void *output, void *mask, const size_t *dims, const size_t *dims_mask, int num_dims, float masked_value, int fp_length)
{
  size_t *d_io_strides, *d_dims_mask;
  size_t total_elements;
  int numBlocks;

  prepare_utils(num_dims, dims, dims_mask, &d_io_strides, &d_dims_mask, &total_elements, &numBlocks);
  
  if(fp_length == 32)
    hipLaunchKernelGGL(mask_kernel_f32, dim3(numBlocks), dim3(blockSize), 0, 0, (float*)input, (float*)output, (char*)mask, d_io_strides, d_dims_mask, num_dims, total_elements, masked_value);
  if(fp_length == 16)
    hipLaunchKernelGGL(mask_kernel_f16, dim3(numBlocks), dim3(blockSize), 0, 0, (half*)input, (half*)output, (char*)mask, d_io_strides, d_dims_mask, num_dims, total_elements, masked_value);

  hipFree(d_io_strides);
  hipFree(d_dims_mask);
}

}
