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
#include <hiprand/hiprand_kernel.h>
#include "hip_customs.h"

#define MAX_DIM 8
#define blockSize 256
// Kernel for multinamial op, treate weight as 2 dimsion matrix by default.
__global__ void multinamial_kernel_f32(float* weights, int64_t* output, int64_t rows, int64_t lines, int64_t n_sample, bool replacement, int64_t seed)
{
  // shared memory weights, one block deal with one row of weights.
  __shared__ float s_weights_row[1024];
  __shared__ float norm_w_row[1024];
  __shared__ int8_t sampled_tags[1024]; // for multinamial without replacement
  // register buffer
  float weights_local[16];
  // init sampled_tags, 0 means not sampled yet.
  for(int i=0; i<4; i++)
    sampled_tags[threadIdx.x+i*256] = 0;

  hiprandStateXORWOW_t randstate;
  hiprand_init(seed, 0, 0, &randstate);

  if(lines <= 1024){
    if(!replacement){
      int sample_round = 0;
      while (sample_round < n_sample)
      {
        // load weights to shared memory, and register buffer
        int s_offset = threadIdx.x;
        int g_offset = blockIdx.x*lines + threadIdx.x;
        int l_offset = 0;
        while (s_offset < lines)
        {
          // if this element is sampled, then load 0;
          if(sampled_tags[s_offset])
            s_weights_row[s_offset] = 0;
          else
            s_weights_row[s_offset] = weights[g_offset];
            
          weights_local[l_offset] = s_weights_row[s_offset];
          s_offset += 256;
          g_offset += 256;
          l_offset ++;
        }

        // sum s_weights_row, all values will be reduced to s_weights_row[0].
        s_offset = threadIdx.x;
        int mid = lines;
        while (s_offset < mid)
        {
          if(mid & 1){
            mid += 1; // make mid always a even number before divide by 2.
            s_weights_row[mid-1] = 0;
          }
          mid = mid >> 1;

          s_weights_row[s_offset] += s_weights_row[s_offset + mid];

          if(mid == 1)
            break;
        }
        // TODO: overflow? 
        float row_sum = s_weights_row[0];
        // sum end

        // reload and normalize weights to shared memory from register buffer.
        s_offset = threadIdx.x;
        l_offset = 0;
        while (s_offset < lines)
        {
          if(row_sum != 0)
            norm_w_row[s_offset] = (float)(weights_local[l_offset]) / row_sum;
          else
            norm_w_row[s_offset] = 1; // if all left value is 0, then all 0's sample possibility is 1.
          // printf("norm_w_row[%d]: %f\n", s_offset, norm_w_row[s_offset]);
          s_offset += 256;
          l_offset ++; 
        }

        __syncthreads();
        // sampling with replacement.
        if(threadIdx.x == 0){
          // sample---
          float uni_f = hiprand_uniform(&randstate);
          // cumulative distribution function of N(0,1)
          float possible = 0.5 * (1 + erf(uni_f / sqrt(2)));

          float p_sum = 0;
          int i = 0;
          for(; i<lines; i++){
            p_sum += norm_w_row[i];
            if(p_sum >= possible && sampled_tags[i] == 0)
              break;
          }
          // printf("uni_f:%f, possible:%f, i: %d\n", uni_f, possible, i);
          output[blockIdx.x*n_sample + sample_round] = i;
          sampled_tags[i] = 1;
        }
        sample_round ++;

        __syncthreads();
      }
    }
  }
  else{
    // TODO:
    return;
  }
}


// Kernel for multinamial op, treate weight as 2 dimsion matrix by default.
__global__ void multinamial_kernel_i64(int64_t* weights, int64_t* output, int64_t rows, int64_t lines, int64_t n_sample, bool replacement, int64_t seed)
{
  // shared memory weights, one block deal with one row of weights.
  __shared__ int64_t s_weights_row[1024];
  __shared__ float norm_w_row[1024];
  __shared__ int8_t sampled_tags[1024]; // for multinamial without replacement
  // register buffer
  int64_t weights_local[16];
  // init sampled_tags, 0 means not sampled yet.
  for(int i=0; i<4; i++)
    sampled_tags[threadIdx.x+i*256] = 0;

  hiprandStateXORWOW_t randstate;
  hiprand_init(seed, 0, 0, &randstate);

  if(lines <= 1024){
    if(!replacement){
      int sample_round = 0;
      while (sample_round < n_sample)
      {
        // load weights to shared memory, and register buffer
        int s_offset = threadIdx.x;
        int g_offset = blockIdx.x*lines + threadIdx.x;
        int l_offset = 0;
        while (s_offset < lines)
        {
          // if this element is sampled, then load 0;
          if(sampled_tags[s_offset])
            s_weights_row[s_offset] = 0;
          else
            s_weights_row[s_offset] = weights[g_offset];
            
          weights_local[l_offset] = s_weights_row[s_offset];
          s_offset += 256;
          g_offset += 256;
          l_offset ++;
        }

        // sum s_weights_row, all values will be reduced to s_weights_row[0].
        s_offset = threadIdx.x;
        int mid = lines;
        while (s_offset < mid)
        {
          if(mid & 1){
            mid += 1; // make mid always a even number before divide by 2.
            s_weights_row[mid-1] = 0;
          }
          mid = mid >> 1;

          s_weights_row[s_offset] += s_weights_row[s_offset + mid];

          if(mid == 1)
            break;
        }
        // TODO: overflow? 
        int64_t row_sum = s_weights_row[0];
        // sum end

        // reload and normalize weights to shared memory from register buffer.
        s_offset = threadIdx.x;
        l_offset = 0;
        while (s_offset < lines)
        {
          if(row_sum != 0)
            norm_w_row[s_offset] = (float)(weights_local[l_offset]) / row_sum;
          else
            norm_w_row[s_offset] = 1; // if all left value is 0, then all 0's sample possibility is 1.
          // printf("norm_w_row[%d]: %f\n", s_offset, norm_w_row[s_offset]);
          s_offset += 256;
          l_offset ++; 
        }

        __syncthreads();
        // sampling with replacement.
        if(threadIdx.x == 0){
          // sample---
          float uni_f = hiprand_uniform(&randstate);
          // cumulative distribution function of N(0,1)
          float possible = 0.5 * (1 + erf(uni_f / sqrt(2)));

          float p_sum = 0;
          int i = 0;
          for(; i<lines; i++){
            p_sum += norm_w_row[i];
            if(p_sum >= possible && sampled_tags[i] == 0)
              break;
          }
          // printf("uni_f:%f, possible:%f, i: %d\n", uni_f, possible, i);
          output[blockIdx.x*n_sample + sample_round] = i;
          sampled_tags[i] = 1;
        }
        sample_round ++;

        __syncthreads();
      }
    }
  }
  else{
    // TODO:
    return;
  }
}

inline void prepare_utils(int num_dims_weight, const size_t *dims_weight, int64_t *rows, 
  int64_t *lines, int* numBlocks){
  size_t total_elements = 1;
  for(int i=0; i<num_dims_weight; i++)
    total_elements *= dims_weight[i];
  
  *numBlocks = total_elements / blockSize;
  if(total_elements % blockSize != 0)
    *numBlocks += 1;
  
  *lines = dims_weight[num_dims_weight-1];
  *rows = total_elements / *lines;
}

namespace hip_custom {

// float tensor mask
void multinormial(void *weight, void *output, const size_t *dims_weight, int num_dims_weight, int64_t n_sample, bool replacement, int64_t seed, int weight_dtype)
{
  // size_t total_elements;
  int64_t rows;
  int64_t lines;
  int numBlocks;

  prepare_utils(num_dims_weight, dims_weight, &rows, &lines, &numBlocks);
  if(weight_dtype == 0)
    hipLaunchKernelGGL(multinamial_kernel_f32, dim3(numBlocks), dim3(blockSize), 0, 0, 
      (float*)weight, (int64_t*)output, rows, lines, n_sample, replacement, seed);
  if(weight_dtype == 4)
    hipLaunchKernelGGL(multinamial_kernel_i64, dim3(numBlocks), dim3(blockSize), 0, 0, 
      (int64_t*)weight, (int64_t*)output, rows, lines, n_sample, replacement, seed);
}

}