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

#ifndef GPU_AMD_HIP_MULTINORMIAL_HPP
#define GPU_AMD_HIP_MULTINORMIAL_HPP

#include "common/multinormial_pd.hpp"
#include "gpu/amd/hip_multinormial_impl.hpp"

#include "gpu/amd/sycl_hip_utils.hpp"
#include "common/c_types_map.hpp"
#include "gpu/amd/engine.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_multinormial_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public multinormial_pd_t {
        using multinormial_pd_t::multinormial_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", hip_multinormial_t);

        status_t init(impl::engine_t *) {
            using namespace data_type;

            // TODO: do some check.
            // bool ok = true;

            if (check_for_zero_dims()) 
                return status::invalid_arguments;
            if (!check_data_types()) 
                return status::invalid_arguments;
            
            multinormial_impl_.reset(new hip_multinormial_impl_t());
            return multinormial_impl_->init(this);
        }

        bool check_for_zero_dims() const {
            bool dst_zero = has_zero_dims(dst_md()->dims, dst_md()->ndims);
            bool weights_zero = has_zero_dims(weights_md()->dims, weights_md()->ndims);
            return dst_zero || weights_zero;
        }

        bool check_no_blocking() const {
            // Blocking is not supported by MIOPENOpTensor, return false if any
            // blocks are present
            return dst_md()->format_desc.blocking.inner_nblks == 0;
        }

        bool check_data_types() const {
            // using namespace data_type;
            // weight type is int64 by default
            // data_type_t weights_type = weights_md()->data_type;
            return true;
        }

        std::shared_ptr<hip_multinormial_impl_t> multinormial_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
