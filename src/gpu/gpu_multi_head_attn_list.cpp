/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "gpu/gpu_impl_list.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/cudnn_multi_head_attn.hpp"
#endif

// #if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
// #include "gpu/amd/miopen_multi_head_attn.hpp"
// #endif

// #ifdef GENERIC_SYCL_KERNELS_ENABLED
// #include "gpu/generic/sycl/ref_multi_head_attn.hpp"
// #endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_MULTI_HEAD_ATTN_P({
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_multi_head_attn_fwd_t)
        // GPU_INSTANCE_AMD(amd::miopen_multi_head_attn_t)
        // GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_multi_head_attn_t)
        nullptr,
});
// clang-format on
} // namespace

const impl_list_item_t *get_multi_head_attn_impl_list(const multi_head_attn_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
