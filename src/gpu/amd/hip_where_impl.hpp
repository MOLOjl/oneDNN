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

#ifndef GPU_AMD_HIP_WHERE_IMPL_HPP
#define GPU_AMD_HIP_WHERE_IMPL_HPP
#include "gpu/amd/sycl_hip_utils.hpp"
#include "gpu/amd/custom/hip_customs.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_where_impl_t {
	status_t init(const where_pd_t *pd) {
		this->num_dims = pd->src_md(0)->ndims;

		for(int i=0; i<this->num_dims; i++){
			this->dims_cond[i] = pd->src_md(0)->dims[i];
			this->dims_x1[i] = pd->src_md(1)->dims[i];
			this->dims_x2[i] = pd->src_md(2)->dims[i];
			this->dims_y[i] = pd->dst_md()->dims[i];
		}

		this->dtype_io = pd->src_md(0)->data_type;
		return status::success;
	}

	void execute(void *cond, void *x1, void *x2, void *y) const {
		printf("hiya\n");
		// if(dtype_io == dnnl::impl::data_type_t::dnnl_f32){
		// 	hip_custom::where(cond, x1, x2, y, dims_cond, dims_x1, dims_x2, dims_y, num_dims, 0);
		// }
		// else if(dtype_io == dnnl::impl::data_type_t::dnnl_f16){
		// 	hip_custom::where(cond, x1, x2, y, dims_cond, dims_x1, dims_x2, dims_y, num_dims, 1);
		// }
		// else if(dtype_io == dnnl::impl::data_type_t::dnnl_s32){
		// 	hip_custom::where(cond, x1, x2, y, dims_cond, dims_x1, dims_x2, dims_y, num_dims, 2);
		// }
		return;
	}

	dnnl::impl::data_type_t dtype_io;
	int num_dims;
	size_t dims_cond[8];
	size_t dims_x1[8];
	size_t dims_x2[8];
	size_t dims_y[8];
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
