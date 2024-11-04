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
#include <hip/hip_bfloat16.h>

namespace hip_custom {

void multinormial(void *weight, void *output, const size_t *dims_weight, int num_dims_weight, int64_t n_sample, 
  bool replacement, int64_t seed, int weight_dtype);

void embedding(void *input, void *output, void *dictionary, const size_t *dims_i, const size_t *dims_dict, int num_dims_i, int dtype);

void transpose(int dtype, void *input, void *output, const size_t *dims, int num_dims, int dim1, int dim2);

void mask(void *input, void *output, void *mask, const size_t *dims, const size_t *dims_mask, int num_dims, float masked_value, int fp_length);

void gather(void *input, void *output, void *index, const size_t *dims, int num_dims, int gather_dim, int dtype);

void where(void *condition, void *input, void *other, void *output, 
  const size_t *dims_c, const size_t *dims_i, const size_t *dims_other, const size_t *dims_o, 
  int num_dims, int dtype);

template <typename T1>
void fill(T1 *ioput, const size_t length, T1 value);

void fill_bf16(hip_bfloat16* ioput, const size_t length, double value);

template <typename T1, typename T2>
void arange(T1 *ioput, T2 start, T2 end, T2 step);

void arange_bf16(hip_bfloat16 *ioput, double start, double end, double step);

void print_device_array(void* dev_a, size_t length, int dtype);

// dtype: 0:f32, 1:f16, 2:i32, 3:f64, 4:i64
}

