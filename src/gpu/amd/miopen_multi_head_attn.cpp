/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "gpu/amd/miopen_multi_head_attn.hpp"
#include "gpu/amd/stream.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include "xpu/sycl/memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t miopen_multi_head_attn_fwd_t::execute(
        const exec_ctx_t &ctx) const {
    amd::stream_t *hip_stream
            = utils::downcast<amd::stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {

        auto arg_queries = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 0);
        auto arg_residuals = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 1);
        auto arg_keys = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 2);
        auto arg_values = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 3);

        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

        auto arg_qweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 4);
        auto arg_qbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 5);
        auto arg_kweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 6);
        auto arg_kbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 7);
        auto arg_vweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 8);
        auto arg_vbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 9);
        auto arg_oweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 10);
        auto arg_obias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 11);
        
        auto arg_workspace = CTX_SCRATCH_SYCL_MEMORY(memory_tracking::names::key_attn_workspace);
        auto arg_reservespace = CTX_SCRATCH_SYCL_MEMORY(memory_tracking::names::key_attn_reservespace);
        auto arg_states1 = CTX_SCRATCH_SYCL_MEMORY(memory_tracking::names::key_attn_dropout_states);
        auto arg_states2 = CTX_SCRATCH_SYCL_MEMORY(memory_tracking::names::key_attn_post_dropout_states);
        
        compat::host_task(cgh, [=, this](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<amd::engine_t *>(
                    hip_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            std::vector<void *> args;
            args.push_back(arg_queries.get_native_pointer(ih));
            args.push_back(arg_residuals.get_native_pointer(ih));
            args.push_back(arg_keys.get_native_pointer(ih));
            args.push_back(arg_values.get_native_pointer(ih));
            args.push_back(arg_dst.get_native_pointer(ih));
            args.push_back(arg_qweight.get_native_pointer(ih));
            args.push_back(arg_qbias.get_native_pointer(ih));
            args.push_back(arg_kweight.get_native_pointer(ih));
            args.push_back(arg_kbias.get_native_pointer(ih));
            args.push_back(arg_vweight.get_native_pointer(ih));
            args.push_back(arg_vbias.get_native_pointer(ih));
            args.push_back(arg_oweight.get_native_pointer(ih));
            args.push_back(arg_obias.get_native_pointer(ih));
            args.push_back(arg_workspace.get_native_pointer(ih));
            args.push_back(arg_reservespace.get_native_pointer(ih));
            args.push_back(arg_states1.get_native_pointer(ih));
            args.push_back(arg_states2.get_native_pointer(ih));

            pd()->set_workspace(arg_workspace.get_native_pointer(ih));
            pd()->set_reservespace(arg_reservespace.get_native_pointer(ih));

            pd()->multi_head_attn_fwd_impl_->execute(handle, args);
        });
    });
}

status_t miopen_multi_head_attn_bwd_data_t::execute(
        const exec_ctx_t &ctx) const {
    amd::stream_t *hip_stream
            = utils::downcast<amd::stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_dout = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 0);

        auto arg_dqueries = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 0);
        auto arg_dkeys = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 1);
        auto arg_dvalues = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 2);
        
        auto arg_qweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 1);
        auto arg_qbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 2);
        auto arg_kweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 3);
        auto arg_kbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 4);
        auto arg_vweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 5);
        auto arg_vbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 6);
        auto arg_oweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 7);
        auto arg_obias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 8);

        compat::host_task(cgh, [=, this](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<amd::engine_t *>(
                    hip_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            std::vector<void *> args;
            args.push_back(arg_dout.get_native_pointer(ih));

            args.push_back(arg_dqueries.get_native_pointer(ih));
            args.push_back(arg_dkeys.get_native_pointer(ih));
            args.push_back(arg_dvalues.get_native_pointer(ih));

            args.push_back(arg_qweight.get_native_pointer(ih));
            args.push_back(arg_qbias.get_native_pointer(ih));
            args.push_back(arg_kweight.get_native_pointer(ih));
            args.push_back(arg_kbias.get_native_pointer(ih));
            args.push_back(arg_vweight.get_native_pointer(ih));
            args.push_back(arg_vbias.get_native_pointer(ih));
            args.push_back(arg_oweight.get_native_pointer(ih));
            args.push_back(arg_obias.get_native_pointer(ih));

            pd()->multi_head_attn_bwd_data_impl_->execute(handle, args);
        });
    });
}

status_t miopen_multi_head_attn_bwd_weights_t::execute(
        const exec_ctx_t &ctx) const {
    amd::stream_t *hip_stream
            = utils::downcast<amd::stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_queries = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 0);
        auto arg_keys = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 1);
        auto arg_values = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 2);
        auto arg_dout = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 3);
        
        auto arg_qweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 4);
        auto arg_qbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 5);
        auto arg_kweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 6);
        auto arg_kbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 7);
        auto arg_vweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 8);
        auto arg_vbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 9);
        auto arg_oweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 10);
        auto arg_obias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 11);

        auto arg_dqweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 0);
        auto arg_dqbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 1);
        auto arg_dkweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 2);
        auto arg_dkbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 3);
        auto arg_dvweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 4);
        auto arg_dvbias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 5);
        auto arg_doweight = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 6);
        auto arg_dobias = CTX_IN_SYCL_MEMORY(DNNL_ARG_MULTIPLE_DST + 7);

        auto arg_reduce_workspace = CTX_SCRATCH_SYCL_MEMORY(memory_tracking::names::key_attn_reduce);

        compat::host_task(cgh, [=, this](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<amd::engine_t *>(
                    hip_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            std::vector<void *> args;
            args.push_back(arg_queries.get_native_pointer(ih));
            args.push_back(arg_keys.get_native_pointer(ih));
            args.push_back(arg_values.get_native_pointer(ih));
            args.push_back(arg_dout.get_native_pointer(ih));

            args.push_back(arg_qweight.get_native_pointer(ih));
            args.push_back(arg_qbias.get_native_pointer(ih));
            args.push_back(arg_kweight.get_native_pointer(ih));
            args.push_back(arg_kbias.get_native_pointer(ih));
            args.push_back(arg_vweight.get_native_pointer(ih));
            args.push_back(arg_vbias.get_native_pointer(ih));
            args.push_back(arg_oweight.get_native_pointer(ih));
            args.push_back(arg_obias.get_native_pointer(ih));

            args.push_back(arg_dqweight.get_native_pointer(ih));
            args.push_back(arg_dqbias.get_native_pointer(ih));
            args.push_back(arg_dkweight.get_native_pointer(ih));
            args.push_back(arg_dkbias.get_native_pointer(ih));
            args.push_back(arg_dvweight.get_native_pointer(ih));
            args.push_back(arg_dvbias.get_native_pointer(ih));
            args.push_back(arg_doweight.get_native_pointer(ih));
            args.push_back(arg_dobias.get_native_pointer(ih));

            args.push_back(arg_reduce_workspace.get_native_pointer(ih));

            pd()->multi_head_attn_bwd_weights_impl_->execute(handle, args);
        });
    });
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
