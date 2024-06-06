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

#define MAX_DIM 8
// Kernel to transpose the specified dimensions of an N-dimensional array
__global__ void transpose_kernel_f32(float *input, float *output, size_t* in_strides, size_t* out_strides, int num_dims, int dim1, int dim2, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;
  int coordinate[MAX_DIM];
  
  __shared__ size_t istrides[MAX_DIM];
  __shared__ size_t ostrides[MAX_DIM];

  if(threadIdx.x < num_dims)
  {
    istrides[threadIdx.x] = in_strides[threadIdx.x];
    ostrides[threadIdx.x] = out_strides[threadIdx.x];
  }
  
  __syncthreads();

  for(int i = 0; i < IterCount; i++)
  {
    size_t input_offset = idx + Scope * i;
    if(input_offset >= total_elements)
      continue;

    // read
    float value = input[input_offset];

    // colculate coordinate of element.
    size_t margin = idx + Scope * i;
    for(int j = 0; j < num_dims; j++)
    {
      coordinate[j] = margin / istrides[j];
      margin = margin % istrides[j];
    }

    // swap to transposed coordinate.
    int temp = coordinate[dim1];
    coordinate[dim1] = coordinate[dim2];
    coordinate[dim2] = temp;

    // colculate output_offset.
    size_t output_offset = 0;
    for(int j = 0; j < num_dims; j++)
    {
      output_offset += coordinate[j]*ostrides[j];
    }
      
    output[output_offset] = value;
  }
}

namespace hip_custom {

void transpose(int dtype, void *input, void *output, const size_t *dims, int num_dims, int dim1, int dim2)
{
  size_t* outdims = (size_t*)malloc(num_dims * sizeof(size_t));
  memcpy(outdims, dims, num_dims*sizeof(size_t));

  size_t temp = outdims[dim1];
  outdims[dim1] = outdims[dim2];
  outdims[dim2] = temp;

  size_t* in_strides = (size_t*)malloc(num_dims * sizeof(size_t));
  size_t* out_strides = (size_t*)malloc(num_dims * sizeof(size_t));
  in_strides[num_dims-1] = 1;
  out_strides[num_dims-1] = 1;

  for (int i = num_dims-2; i >= 0; i--)
  {
    in_strides[i] = in_strides[i+1] * dims[i+1];
    out_strides[i] = out_strides[i+1] * outdims[i+1];
  }
  
  size_t *d_in_strides, *d_out_strides;
  hipMalloc(&d_in_strides, num_dims * sizeof(size_t));
  hipMalloc(&d_out_strides, num_dims * sizeof(size_t));
  hipMemcpy(d_in_strides, in_strides, num_dims*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(d_out_strides, out_strides, num_dims*sizeof(size_t), hipMemcpyHostToDevice);

  size_t total_elements = in_strides[0] * dims[0];

  int blockSize = 256;
  int numBlocks = total_elements / blockSize;
  if(total_elements % blockSize != 0)
    numBlocks += 1;
  
  if(dtype == 0)
    hipLaunchKernelGGL(transpose_kernel_f32, dim3(numBlocks), dim3(blockSize), 0, 0, (float*)input, (float*)output, d_in_strides, d_out_strides, num_dims, dim1, dim2, total_elements);
  // TODO:
  
  free(outdims);
  free(in_strides);
  free(out_strides);

  hipFree(d_in_strides);
  hipFree(d_out_strides);

}

}
