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

#ifndef GPU_AMD_HIP_MASK_IMPL_HPP
#define GPU_AMD_HIP_MASK_IMPL_HPP
#include "gpu/amd/sycl_hip_utils.hpp"
#include "gpu/amd/custom/hip_customs.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_mask_impl_t {
    status_t init(const mask_pd_t *pd) {
        this->src_dtype = pd->src_md()->data_type;
        this->value_f = pd->value_f();
        this->value_i = pd->value_i();
        this->num_dims = pd->src_md()->ndims;

        for(int i=0; i<this->num_dims; i++){
            this->dims_io[i] = pd->src_md()->dims[i];
            this->dims_mask[i] = pd->mask_md()->dims[i];
        }
        this->src_dtype = pd->src_md()->data_type;
        return status::success;
    }

    void execute(void *x, void *y, void* mask) const {
        if(src_dtype == dnnl::impl::data_type_t::dnnl_f32){
            printf("???\n");
            // hip_custom::mask(x, y, mask, dims_io, dims_mask, num_dims, (float)value_f, 32);
        }
        else if(src_dtype == dnnl::impl::data_type_t::dnnl_f16){
            // hip_custom::mask(x, y, mask, dims_io, dims_mask, num_dims, (float)value_f, 16);
        }
      	return;
    }

    dnnl::impl::data_type_t src_dtype;
    int num_dims;
    size_t dims_io[8];
    size_t dims_mask[8];

    double value_f;
    int64_t value_i;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
