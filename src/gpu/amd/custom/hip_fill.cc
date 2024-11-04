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
#define blockSize 256

// Kernel to mask the input, can be done in place. the mask tensor shall be broadcastable to the input tensor.
template <typename T1>
__global__ void fill_kernel(T1* ioput, size_t total_elements, T1 value)
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
    ioput[io_offset] = value;
  }
}

namespace hip_custom {

// fill tensor with a scalar
template <typename T1>
void fill(T1 *ioput, const size_t length, T1 value){
  size_t total_elements = length;
  int numBlocks = total_elements / blockSize;
  if(total_elements % blockSize != 0)
    numBlocks += 1;

  hipLaunchKernelGGL(fill_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, ioput, total_elements, value);
}

template void fill<int32_t>(int32_t* ioput, const size_t length, int32_t value);
template void fill<int64_t>(int64_t* ioput, const size_t length, int64_t value);
template void fill<float>(float* ioput, const size_t length, float value);
template void fill<double>(double* ioput, const size_t length, double value);
template void fill<__half>(__half* ioput, const size_t length, __half value);
template void fill<hip_bfloat16>(hip_bfloat16* ioput, const size_t length, hip_bfloat16 value);

// The clang used to compile onednn doesn't support float cast to bf6.
void fill_bf16(hip_bfloat16* ioput, const size_t length, double value){
  hip_bfloat16 bf16_v((float)value);
  fill(ioput, length, bf16_v);
}

}
