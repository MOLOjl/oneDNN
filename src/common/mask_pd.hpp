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

#ifndef COMMON_MASK_PD_HPP
#define COMMON_MASK_PD_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_MASK(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, mask, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

namespace dnnl {
namespace impl {

status_t mask_desc_init(mask_desc_t *mask_desc,
        const memory_desc_t *src_md, const memory_desc_t *dst_md, 
        const memory_desc_t *mask_md, 
        double value_f, double value_i);

struct mask_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::mask;

    typedef mask_pd_t base_class;
    typedef mask_pd_t hint_class;

    const mask_desc_t *desc() const { return &desc_; }
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

    const memory_desc_t *mask_md(
            int index = 0, bool user_input = false) const {
        if (index == 0) return user_input ? &desc()->mask_desc : &mask_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override { return 1; }

	double value_f() const { return value_f_; }
	int64_t value_i() const { return value_i_; }

protected:
    mask_desc_t desc_;
    memory_desc_t src_md_;
    memory_desc_t dst_md_;
    memory_desc_t mask_md_;
	double value_f_;
    int64_t value_i_;

	mask_pd_t(const mask_desc_t *adesc, const primitive_attr_t *attr,
		const mask_pd_t *hint_fwd_pd)
		: primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc)
        , mask_md_(desc_.mask_desc)
        , value_f_(desc_.value_f)
        , value_i_(desc_.value_i) {}
};

} // namespace impl
} // namespace dnnl

#endif