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

#ifndef GPU_AMD_HIP_MULTINORMIAL_IMPL_HPP
#define GPU_AMD_HIP_MULTINORMIAL_IMPL_HPP
#include "gpu/amd/sycl_hip_utils.hpp"
#include "gpu/amd/custom/hip_customs.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_multinormial_impl_t {
    status_t init(const multinormial_pd_t *pd) {
        this->num_dims_w = pd->weights_md()->ndims;
        this->n_sample = (int64_t)(pd->n_sample());
        this->replacement = pd->replacement();
        this->seed = (int64_t)(pd->seed());

        this->w_dtype = pd->weights_md()->data_type;
        this->o_dtype = pd->dst_md()->data_type;

        for(int i=0; i<this->num_dims_w; i++)
            this->dims_w[i] = pd->weights_md()->dims[i];
        
        return status::success;
    }

    void execute(void *w, void *y) const {
        printf("hiya multinormial\n");
        // if(w_dtype == dnnl::impl::data_type_t::dnnl_f32){
        //     hip_custom::multinormial(w, y, dims_w, num_dims_w, n_sample, replacement, seed, 0);
        // }
        // else if(w_dtype == dnnl::impl::data_type_t::dnnl_f16){
        //     hip_custom::multinormial(w, y, dims_w, num_dims_w, n_sample, replacement, seed, 1);
        // }
      	return;
    }

    dnnl::impl::data_type_t w_dtype;
    dnnl::impl::data_type_t o_dtype;
    int num_dims_w;
    size_t dims_w[8];
    int64_t n_sample;
    bool replacement = false;
    int64_t seed;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
