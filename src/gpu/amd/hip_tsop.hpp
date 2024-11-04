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

#ifndef GPU_AMD_HIP_TSOP_HPP
#define GPU_AMD_HIP_TSOP_HPP

#include "common/tsop_pd.hpp"
#include "gpu/amd/hip_tsop_impl.hpp"

#include "gpu/amd/sycl_hip_utils.hpp"
#include "common/c_types_map.hpp"
#include "gpu/amd/engine.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_tsop_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public tsop_pd_t {
        using tsop_pd_t::tsop_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", hip_tsop_t);

        status_t init(impl::engine_t *) {
            using namespace data_type;

            // TODO: do some check.
            // bool ok = true;

            if (check_for_zero_dims()) return status::success;
            if (check_no_blocking()) 
                return status::invalid_arguments;
            if(!check_data_types())
                return status::invalid_arguments;

            tsop_impl_.reset(new hip_tsop_impl_t());
            return tsop_impl_->init(this);
        }

        bool check_for_zero_dims() const {
            bool src_zero = has_zero_dims(src_md()->dims, src_md()->ndims);
            bool dst_zero = has_zero_dims(dst_md()->dims, dst_md()->ndims);
            return src_zero || dst_zero;
        }

        bool check_no_blocking() const {
            // Blocking is not supported, return false if any blocks are present
            bool src_blk = src_md()->format_desc.blocking.inner_nblks;
            bool dst_blk = dst_md()->format_desc.blocking.inner_nblks;
            return src_blk || dst_blk;
        }

        bool check_data_types() const {
            using namespace data_type;
            data_type_t src_type = src_md()->data_type;
            data_type_t dst_type = dst_md()->data_type;

            bool type_ok = (src_type == dst_type);
            return type_ok;
        }
        
        std::shared_ptr<hip_tsop_impl_t> tsop_impl_;
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
