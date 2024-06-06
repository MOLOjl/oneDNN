/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#ifndef COMMON_TRANSPOSE_PD_HPP
#define COMMON_TRANSPOSE_PD_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_TRANSPOSE(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, transpose, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

namespace dnnl {
namespace impl {

status_t transpose_desc_init(transpose_desc_t *transpose_desc,
        const memory_desc_t *src_md, const memory_desc_t *dst_md, 
        dnnl_dim_t dim1, dnnl_dim_t dim2);

struct transpose_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::transpose;

    typedef transpose_pd_t base_class;
    typedef transpose_pd_t hint_class;

    const transpose_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_FROM) return arg_usage_t::input;

        if (arg == DNNL_ARG_TO) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_FROM: return src_md(0);
            case DNNL_ARG_TO: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override { return 1; }

	dim_t dim1() const { return dim1_; }
	dim_t dim2() const { return dim2_; }

protected:
    transpose_desc_t desc_;
    memory_desc_t src_md_;
    memory_desc_t dst_md_;
    dnnl_dim_t dim1_;
    dnnl_dim_t dim2_;

	transpose_pd_t(const transpose_desc_t *adesc, const primitive_attr_t *attr,
		const transpose_pd_t *hint_fwd_pd)
		: primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc)
        , dim1_(desc_.dim1)
        , dim2_(desc_.dim2) {}
};

} // namespace impl
} // namespace dnnl

#endif