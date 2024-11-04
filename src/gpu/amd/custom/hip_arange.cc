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
#include<math.h>
#include <type_traits>

#define MAX_DIM 8
#define blockSize 256

// Kernel to initialize a vector with values from the interval [start, end) taken with 
// common difference step beginning from start.
template <typename T1, typename T2>
__global__ void arange_kernel(T1* ioput, T2 start, T2 end, T2 step, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;

  for(int i = 0; i < IterCount; i++)
  {
    size_t io_offset = idx + Scope * i;
    if(io_offset >= total_elements)
      continue;
    
    // write
    T2 value = start + io_offset * step;
    if(value < end)
      ioput[io_offset] = value;
  }
}

// rocm5.4 and lower version don't support type cast from double/float to hip_bfloat16, so an individual function is needed.
__global__ void arange_kernel_bf16(hip_bfloat16* ioput, double start, double end, double step, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Scope = blockDim.x * gridDim.x;
  int IterCount = total_elements / Scope + 1;

  for(int i = 0; i < IterCount; i++)
  {
    size_t io_offset = idx + Scope * i;
    if(io_offset >= total_elements)
      continue;
    
    // write
    hip_bfloat16 value(start + io_offset * step);
    if(value < end)
      ioput[io_offset] = value;
  }
}

namespace hip_custom {

template <typename T1, typename T2>
void arange(T1 *ioput, T2 start, T2 end, T2 step){
  size_t total_elements = ceil((end - start)/ step);
  int numBlocks = total_elements / blockSize;
  if(total_elements % blockSize != 0)
    numBlocks += 1;

  hipLaunchKernelGGL(arange_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, ioput, start, 
      end, step, total_elements);
}

template void arange<float, double>(float *ioput, double start, double end, double step);
template void arange<__half, double>(__half *ioput, double start, double end, double step);
template void arange<double, double>(double *ioput, double start, double end, double step);
template void arange<int32_t, int64_t>(int32_t *ioput, int64_t start, int64_t end, int64_t step);
template void arange<int64_t, int64_t>(int64_t *ioput, int64_t start, int64_t end, int64_t step);

void arange_bf16(hip_bfloat16 *ioput, double start, double end, double step){
  size_t total_elements = ceil((end - start) / step);
  int numBlocks = total_elements / blockSize;
  if(total_elements % blockSize != 0)
    numBlocks += 1;

  hipLaunchKernelGGL(arange_kernel_bf16, dim3(numBlocks), dim3(blockSize), 0, 0, ioput, start, 
      end, step, total_elements);
}

}
