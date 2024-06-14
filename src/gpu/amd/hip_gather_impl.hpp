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

struct hip_gather_impl_t {
    status_t init(const gather_pd_t *pd) {
        printf("---\n");
        this->src_dtype = pd->src_md()->data_type;
        this->gather_dim = pd->gather_dim();
        this->num_dims = pd->src_md()->ndims;

        for(int i=0; i<this->num_dims; i++){
            this->dims[i] = pd->src_md()->dims[i];
            if(this->dims[i] != pd->idx_md()->dims[i])
                return status::invalid_arguments;
        }

        this->src_dtype = pd->src_md()->data_type;
        return status::success;
    }

    void execute(void *x, void *y, void* index) const {
        if(src_dtype == dnnl::impl::data_type_t::dnnl_f32){
            // printf("x:%p", x);
            // hip_custom::print_device_array(x, dims[0]*dims[3]*dims[1]*dims[2], 0);
            // hip_custom::print_device_array(y, dims[0]*dims[3]*dims[1]*dims[2], 0);
            // hip_custom::print_device_array(index, dims[0]*dims[3]*dims[1]*dims[2], 2);

            // hip_custom::gather(x, y, index, dims, num_dims, gather_dim, 0);
        }
        else if(src_dtype == dnnl::impl::data_type_t::dnnl_f16){
            hip_custom::gather(x, y, index, dims, num_dims, gather_dim, 1);
        }
      	return;
    }

    dnnl::impl::data_type_t src_dtype;
    int num_dims;
    size_t dims[8];
    int gather_dim = 0;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
