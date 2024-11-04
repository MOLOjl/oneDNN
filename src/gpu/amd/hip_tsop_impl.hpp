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

#ifndef GPU_AMD_HIP_TSOP_IMPL_HPP
#define GPU_AMD_HIP_TSOP_IMPL_HPP
#include "gpu/amd/sycl_hip_utils.hpp"
#include "gpu/amd/custom/hip_customs.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct hip_tsop_impl_t {
	status_t init(const tsop_pd_t *pd) {
		auto num_dims = pd->src_md()->ndims;
		for(int i=0; i<num_dims; i++){
			src_length *= pd->src_md()->dims[i];
		}
		alg = pd->desc()->alg_kind;
		this->dtype_io = pd->src_md()->data_type;
		auto value_3 = pd->desc()->v3;

		int num_v = 1;
		if(alg == alg_kind_t::dnnl_tsop_arange)
			num_v = 3;
		
		for(int i=0; i<num_v; i++) {
			vd[i] = std::get<0>(value_3)[i];
			vi[i] = std::get<1>(value_3)[i];
			vb[i] = std::get<2>(value_3)[i];
		}

		return status::success;
	}

	void execute(void *src, void *dst) const {
		if(alg == alg_kind_t::dnnl_tsop_fill)
		{
			assert(src == dst);
			switch (dtype_io)
			{
			case dnnl::impl::data_type_t::dnnl_f32:
				hip_custom::fill((float*)src, src_length, (float)(vd[0]));
				break;
			case dnnl::impl::data_type_t::dnnl_f16:
				hip_custom::fill((__half*)src, src_length, (__half)(vd[0]));
				break;
			case dnnl::impl::data_type_t::dnnl_bf16:
				hip_custom::fill_bf16((hip_bfloat16*)src, src_length, vd[0]);
				break;
			case dnnl::impl::data_type_t::dnnl_s32:
				hip_custom::fill((int32_t*)src, src_length, (int32_t)(vi[0]));
				break;
			case dnnl::impl::data_type_t::dnnl_f64:
				hip_custom::fill((double*)src, src_length, vd[0]);
				break;
			case dnnl::impl::data_type_t::dnnl_s64:
				hip_custom::fill((int64_t*)src, src_length, vi[0]);
				break;
			default:
				break;
			}
		}
		
		if(alg == alg_kind_t::dnnl_tsop_arange) 
		{
			assert(src == dst);
			switch (dtype_io)
			{
			case dnnl::impl::data_type_t::dnnl_f32:
				hip_custom::arange((float*)src, vd[0], vd[1], vd[2]);
				break;
			case dnnl::impl::data_type_t::dnnl_f16:
				hip_custom::arange((__half*)src, vd[0], vd[1], vd[2]);
				break;
			case dnnl::impl::data_type_t::dnnl_bf16:
				hip_custom::arange_bf16((hip_bfloat16*)src, vd[0], vd[1], vd[2]);
				break;
			case dnnl::impl::data_type_t::dnnl_s32:
				hip_custom::arange((int32_t*)src, vi[0], vi[1], vi[2]);
				break;
			case dnnl::impl::data_type_t::dnnl_f64:
				hip_custom::arange((double*)src, vd[0], vd[1], vd[2]);
				break;
			case dnnl::impl::data_type_t::dnnl_s64:
				hip_custom::arange((int64_t*)src, vi[0], vi[1], vi[2]);
				break;
			default:
				break;
			}
		}
		return;
	}

	dnnl::impl::data_type_t dtype_io;
	size_t src_length = 1;
	alg_kind_t alg;
	double vd[8];
	int64_t vi[8];
	bool vb[8];
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
