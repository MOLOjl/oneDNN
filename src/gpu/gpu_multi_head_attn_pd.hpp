/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef GPU_GPU_MULTI_HEAD_ATTN_PD_HPP
#define GPU_GPU_MULTI_HEAD_ATTN_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/multi_head_attn_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

// forward and backward use one pd.
struct gpu_multi_head_attn_pd_t : public multi_head_attn_pd_t {
    using multi_head_attn_pd_t::multi_head_attn_pd_t;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
