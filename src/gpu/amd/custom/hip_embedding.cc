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
#include <hip/hip_fp16.h>
#include "hip_customs.h"

#define MAX_DIM 8
#define blockSize 256

// Kernel for embedding op, A simple lookup table that stores embeddings of a fixed dictionary and size.
__global__ void embedding_kernel_f32(int64_t* input, float* output, float* dictionary, size_t* dims_dict, size_t* strides_i, size_t* strides_o, 
  int num_dims_o, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int coordinate[MAX_DIM];
  
  __shared__ size_t s_strides_i[MAX_DIM];
  __shared__ size_t s_strides_o[MAX_DIM];
  __shared__ size_t s_dims_dict[MAX_DIM];

  if(idx < num_dims_o)
    s_strides_o[threadIdx.x] = strides_o[threadIdx.x];
  if(idx < num_dims_o - 1)
    s_strides_i[threadIdx.x] = strides_i[threadIdx.x];
  if(idx < 2)
    s_dims_dict[threadIdx.x] = dims_dict[threadIdx.x];

  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t out_offset = idx + Scope * i;
    if(out_offset >= total_elements)
      continue;

    // colculate coordinate of element.
    size_t margin = out_offset;
    for(int j = 0; j < num_dims_o; j++)
    {
      coordinate[j] = margin / s_strides_o[j];
      margin = margin % s_strides_o[j];
    }

    size_t in_offset = 0;
    for(int j = 0; j < num_dims_o-1; j++)
      in_offset += coordinate[j] * s_strides_i[j];

    // read dictionary idx
    int64_t dict_idx = input[in_offset];

    size_t dict_offset = dict_idx*s_dims_dict[1] + coordinate[num_dims_o-1];

    // read dictionary value
    float dict_value = dictionary[dict_offset];
    output[out_offset] = dict_value;
  }
}

// Kernel for embedding op, A simple lookup table that stores embeddings of a fixed dictionary and size.
__global__ void embedding_kernel_f16(int64_t* input, half* output, half* dictionary, size_t* dims_dict, size_t* strides_i, size_t* strides_o, 
  int num_dims_o, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int coordinate[MAX_DIM];
  
  __shared__ size_t s_strides_i[MAX_DIM];
  __shared__ size_t s_strides_o[MAX_DIM];
  __shared__ size_t s_dims_dict[MAX_DIM];

  if(idx < num_dims_o)
    s_strides_o[threadIdx.x] = strides_o[threadIdx.x];
  if(idx < num_dims_o - 1)
    s_strides_i[threadIdx.x] = strides_i[threadIdx.x];
  if(idx < 2)
    s_dims_dict[threadIdx.x] = dims_dict[threadIdx.x];

  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t out_offset = idx + Scope * i;
    if(out_offset >= total_elements)
      continue;

    // colculate coordinate of element.
    size_t margin = out_offset;
    for(int j = 0; j < num_dims_o; j++)
    {
      coordinate[j] = margin / s_strides_o[j];
      margin = margin % s_strides_o[j];
    }

    size_t in_offset = 0;
    for(int j = 0; j < num_dims_o-1; j++)
      in_offset += coordinate[j] * s_strides_i[j];

    // read dictionary idx
    int64_t dict_idx = input[in_offset];

    size_t dict_offset = dict_idx*s_dims_dict[1] + coordinate[num_dims_o-1];

    // read dictionary value
    half dict_value = dictionary[dict_offset];
    output[out_offset] = dict_value;
  }
}


inline void prepare_utils(int num_dims_o, const size_t *dims_i, const size_t *dims_dict, size_t **d_strides_i, 
  size_t ** d_strides_o,  size_t ** d_dims_dict, size_t* total_elements, int* numBlocks){
  size_t* strides_i = (size_t*)malloc((num_dims_o-1) * sizeof(size_t));
  size_t* strides_o = (size_t*)malloc(num_dims_o * sizeof(size_t));

  strides_i[num_dims_o-2] = 1;
  strides_o[num_dims_o-1] = 1;
  strides_o[num_dims_o-2] = dims_dict[1];

  for (int i = num_dims_o-3; i >= 0; i--){
    strides_i[i] = strides_i[i+1] * dims_i[i+1];
    strides_o[i] = strides_o[i+1] * dims_i[i+1];    
  }

  printf("dims_i: %ld, %ld\n", dims_i[0], dims_i[1]);
  printf("strides_i: %ld, %ld\n", strides_i[0], strides_i[1]);
  printf("strides_o: %ld, %ld, %ld\n", strides_o[0], strides_o[1], strides_o[2]);
  printf("dims_dict: %ld, %ld\n", dims_dict[0], dims_dict[1]);

  hipMalloc(d_strides_i, (num_dims_o-1) * sizeof(size_t));
  hipMalloc(d_strides_o, num_dims_o * sizeof(size_t));
  hipMalloc(d_dims_dict, 2 * sizeof(size_t));

  hipMemcpy(*d_strides_i, strides_i, (num_dims_o-1)*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(*d_strides_o, strides_o, num_dims_o*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(*d_dims_dict, dims_dict, 2*sizeof(size_t), hipMemcpyHostToDevice);

  *total_elements = strides_o[0] * dims_i[0];

  // ceiling div
  *numBlocks = *total_elements / blockSize;
  if(*total_elements % blockSize != 0)
    *numBlocks += 1;
  
  free(strides_i);
  free(strides_o);
}

namespace hip_custom {

// float tensor mask
void embedding(void *input, void *output, void *dictionary, const size_t *dims_i, const size_t *dims_dict, int num_dims_i, int dtype)
{
  size_t *d_strides_i;
  size_t *d_strides_o;
  size_t *d_dims_dict;
  size_t total_elements;
  int num_dims_o = num_dims_i + 1;
  int numBlocks;

  prepare_utils(num_dims_o, dims_i, dims_dict, &d_strides_i, &d_strides_o, &d_dims_dict, &total_elements, &numBlocks);
  
  if(dtype == 0)
    hipLaunchKernelGGL(embedding_kernel_f32, dim3(numBlocks), dim3(blockSize), 0, 0, 
      (int64_t*)input, (float*)output, (float*)dictionary, d_dims_dict, d_strides_i, 
      d_strides_o, num_dims_o, total_elements);
  if(dtype == 1)
    hipLaunchKernelGGL(embedding_kernel_f16, dim3(numBlocks), dim3(blockSize), 0, 0, 
      (int64_t*)input, (half*)output, (half*)dictionary, d_dims_dict, d_strides_i, 
      d_strides_o, num_dims_o, total_elements);

  hipFree(d_strides_i);
  hipFree(d_strides_o);
  hipFree(d_dims_dict);
}

}
