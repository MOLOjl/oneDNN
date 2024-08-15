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

#ifndef GPU_AMD_HIP_EMBEDDING_IMPL_HPP
#define GPU_AMD_HIP_EMBEDDING_IMPL_HPP
#include "gpu/amd/sycl_hip_utils.hpp"
#include "gpu/amd/custom/hip_customs.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_embedding_impl_t {
    status_t init(const embedding_pd_t *pd) {
        this->dict_dtype = pd->dict_md()->data_type;
        this->num_dims_i = pd->src_md()->ndims;
        // input dims
        for(int i=0; i<this->num_dims_i; i++)
            this->dims_i[i] = pd->src_md()->dims[i];
        if(pd->dict_md()->ndims != 2)
            return status::invalid_arguments;
        // dictionary dims
        for(int i=0; i<2; i++)
            this->dims_dict[i] = pd->dict_md()->dims[i];

        // src tensor's data type will be treat as int64_t
        return status::success;
    }

    void execute(void *x, void *y, void* dict) const {
        printf("hiya embedding\n");
        // if(dict_dtype == dnnl::impl::data_type_t::dnnl_f32){
        //     hip_custom::embedding(x, y, dict, dims_i, dims_dict, num_dims_i, 0);
        // }
        // else if(dict_dtype == dnnl::impl::data_type_t::dnnl_f16){
        //     hip_custom::embedding(x, y, dict, dims_i, dims_dict, num_dims_i, 1);
        // }
      	return;
    }

    dnnl::impl::data_type_t dict_dtype;
    int num_dims_i;
    size_t dims_i[8];
    size_t dims_dict[8];
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
