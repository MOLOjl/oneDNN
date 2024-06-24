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

#ifndef COMMON_WHERE_PD_HPP
#define COMMON_WHERE_PD_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_where(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, where, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

namespace dnnl {
namespace impl {

status_t where_desc_init(where_desc_t *where_desc,
        const memory_desc_t *cond_md, const memory_desc_t *src1_md, 
        const memory_desc_t *src2_md, const memory_desc_t *dst_md);

struct where_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::where;

    typedef where_pd_t base_class;
    typedef where_pd_t hint_class;

    const where_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC_0 || arg == DNNL_ARG_SRC_1 || arg == DNNL_ARG_SRC_2)
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC_0: return src_md(0);
            case DNNL_ARG_SRC_1: return src_md(1);
            case DNNL_ARG_SRC_2: return src_md(2);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->cond_desc : &cond_md_;
        if (index == 1) return user_input ? &desc()->src1_desc : &src1_md_;
        if (index == 2) return user_input ? &desc()->src2_desc : &src2_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 3; }
    int n_outputs() const override { return 1; }

protected:
    where_desc_t desc_;
    memory_desc_t cond_md_;
    memory_desc_t src1_md_;
    memory_desc_t src2_md_;
    memory_desc_t dst_md_;

	where_pd_t(const where_desc_t *adesc, const primitive_attr_t *attr,
		const where_pd_t *hint_fwd_pd)
		: primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , cond_md_(desc_.cond_desc)
        , src1_md_(desc_.src1_desc)
        , src2_md_(desc_.src2_desc)
        , dst_md_(desc_.dst_desc) {}
};

} // namespace impl
} // namespace dnnl

#endif