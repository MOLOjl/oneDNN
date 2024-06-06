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

#ifndef GPU_AMD_HIP_TRANSPOSE_IMPL_HPP
#define GPU_AMD_HIP_TRANSPOSE_IMPL_HPP
#include "gpu/amd/sycl_hip_utils.hpp"
#include "gpu/amd/custom/hip_customs.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_transpose_impl_t {
    status_t init(const transpose_pd_t *pd) {
		this->dim1 = (int)(pd->dim1());
		this->dim2 = (int)(pd->dim2());
		this->num_dims = pd->src_md()->ndims;

		for(int i=0; i<this->num_dims; i++){
			this->dims[i] = pd->src_md()->dims[i];
		}
		this->src_dtype = pd->src_md()->data_type;
        return status::success;
    }
    void execute(void *x, void *y) const {
		if(src_dtype == dnnl::impl::data_type_t::dnnl_f32){
			hip_custom::transpose(0, x, y, dims, num_dims, dim1, dim2);
		}
		// else if(src_dtype == dnnl::impl::data_type_t::dnnl_s8){
		// 	hip_custom::transpose<int8_t>(x, y, dims, num_dims, dim1, dim2);
		// }
		// else if(src_dtype == dnnl::impl::data_type_t::dnnl_s32){
		// 	hip_custom::transpose<int32_t>(x, y, dims, num_dims, dim1, dim2);
		// 	hip_custom::print_hello();
		// }
      	return;
    }

	dnnl::impl::data_type_t src_dtype;
	int dim1;
	int dim2;
	int num_dims;
	size_t dims[8];
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
