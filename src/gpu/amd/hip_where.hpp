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

#ifndef GPU_AMD_HIP_WHERE_HPP
#define GPU_AMD_HIP_WHERE_HPP

#include "common/where_pd.hpp"
#include "gpu/amd/hip_where_impl.hpp"

#include "gpu/amd/sycl_hip_utils.hpp"
#include "common/c_types_map.hpp"
#include "gpu/amd/engine.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_where_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public where_pd_t {
        using where_pd_t::where_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", hip_where_t);

        status_t init(impl::engine_t *) {
            using namespace data_type;

            // TODO: do some check.
            // bool ok = true;

            if (check_for_zero_dims()) return status::success;
            if (check_no_blocking()) 
                return status::invalid_arguments;
            if(!check_data_types())
                return status::invalid_arguments;

            where_impl_.reset(new hip_where_impl_t());
            return where_impl_->init(this);
        }

        bool check_for_zero_dims() const {
            bool cond_zero = has_zero_dims(src_md(0)->dims, src_md(0)->ndims);
            bool src1_zero = has_zero_dims(src_md(1)->dims, src_md(1)->ndims);
            bool src2_zero = has_zero_dims(src_md(2)->dims, src_md(2)->ndims);
            bool dst_zero = has_zero_dims(dst_md()->dims, dst_md()->ndims);
            return cond_zero || src1_zero || src2_zero || dst_zero;
        }

        bool check_no_blocking() const {
            // Blocking is not supported, return false if any blocks are present
            bool cond_blk = src_md(0)->format_desc.blocking.inner_nblks;
            bool src1_blk = src_md(1)->format_desc.blocking.inner_nblks;
            bool src2_blk = src_md(2)->format_desc.blocking.inner_nblks;
            bool dst_blk = dst_md()->format_desc.blocking.inner_nblks;

            return cond_blk || src1_blk || src2_blk || dst_blk;
        }

        bool check_data_types() const {
            using namespace data_type;
            data_type_t cond_type = src_md(0)->data_type;
            data_type_t src1_type = src_md(1)->data_type;
            data_type_t src2_type = src_md(2)->data_type;
            data_type_t dst_type = dst_md()->data_type;

            bool type_ok = (src1_type == src2_type);
            type_ok = type_ok && (src1_type == dst_type);
            type_ok = type_ok && (cond_type == dnnl::impl::data_type_t::dnnl_s8);
            return type_ok;
        }
        
        std::shared_ptr<hip_where_impl_t> where_impl_;
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
