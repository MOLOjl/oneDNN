/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/amd/hip_multinormial.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/stream.hpp"
#include "xpu/sycl/buffer_memory_storage.hpp"
#include "xpu/sycl/memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t hip_multinormial_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->src_md()).has_zero_dim())
        return status::success;
    amd::stream_t *hip_stream
            = utils::downcast<amd::stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_weights = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<amd::engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);

            void *w = arg_weights.get_native_pointer(ih);
            void *y = arg_dst.get_native_pointer(ih);
            
            pd()->multinormial_impl_->execute(w, y);
        });
    });
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
