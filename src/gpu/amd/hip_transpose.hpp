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

#ifndef GPU_AMD_HIP_TRANSPOSE_HPP
#define GPU_AMD_HIP_TRANSPOSE_HPP

#include "common/transpose_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/amd/hip_transpose_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include <miopen/miopen.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_transpose_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public transpose_pd_t {
        using transpose_pd_t::transpose_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", hip_transpose_t);

        status_t init(engine_t *) {
            using namespace data_type;

            // TODO: do some check.
            // bool ok = true;

            if (check_for_zero_dims()) return status::success;

            transpose_impl_.reset(new hip_transpose_impl_t());

            return transpose_impl_->init(this);
        }

        bool check_for_zero_dims() const {
            return has_zero_dims(src_md()->dims, src_md()->ndims)
                    || has_zero_dims(dst_md()->dims, dst_md()->ndims);
        }

        bool check_no_blocking() const {
            // Blocking is not supported by MIOPENOpTensor, return false if any
            // blocks are present
            return src_md(0)->format_desc.blocking.inner_nblks
                    + src_md(1)->format_desc.blocking.inner_nblks
                    + dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        bool check_data_types() const {
            using namespace data_type;
            data_type_t input_type = src_md()->data_type;
            data_type_t output_type = dst_md()->data_type;
            bool type_same = (input_type == output_type);
            return type_same;
        }
        
        std::shared_ptr<hip_transpose_impl_t> transpose_impl_;
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
