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

#ifndef GPU_NVIDIA_SYCL_CUDA_MULTI_HEAD_ATTN_HPP
#define GPU_NVIDIA_SYCL_CUDA_MULTI_HEAD_ATTN_HPP

#include "common/multi_head_attn_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/nvidia/cudnn_multi_head_attn_impl.hpp"
#include "gpu/nvidia/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_multi_head_attn_fwd_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public multi_head_attn_pd_t {
        using multi_head_attn_pd_t::multi_head_attn_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_multi_head_attn_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace alg_kind;

            auto sycl_dev
                    = utils::downcast<nvidia::engine_t *>(engine)->device();
            
            multi_head_attn_fwd_impl_.reset(new cudnn_multi_head_attn_fwd_impl_t());
            return multi_head_attn_fwd_impl_->init(engine, this);
        }
        std::shared_ptr<cudnn_multi_head_attn_fwd_impl_t> multi_head_attn_fwd_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_multi_head_attn_bwd_data_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public multi_head_attn_pd_t {
        using multi_head_attn_pd_t::multi_head_attn_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_multi_head_attn_bwd_data_t);

        status_t init(impl::engine_t * engine) {
            using namespace alg_kind;

            multi_head_attn_bwd_data_impl_.reset(new cudnn_multi_head_attn_bwd_data_impl_t());
            return multi_head_attn_bwd_data_impl_->init(engine, this);
        }
        std::shared_ptr<cudnn_multi_head_attn_bwd_data_impl_t> multi_head_attn_bwd_data_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_multi_head_attn_bwd_weights_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public multi_head_attn_pd_t {
        using multi_head_attn_pd_t::multi_head_attn_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_multi_head_attn_bwd_weights_t);

        status_t init(impl::engine_t * engine) {
            using namespace alg_kind;

            multi_head_attn_bwd_weights_impl_.reset(new cudnn_multi_head_attn_bwd_weights_impl_t());
            return multi_head_attn_bwd_weights_impl_->init(engine, this);
        }
        std::shared_ptr<cudnn_multi_head_attn_bwd_weights_impl_t> multi_head_attn_bwd_weights_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
