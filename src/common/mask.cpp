/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#include <assert.h>
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {

#define VCHECK_TRANSPOSE(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, mask, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

status_t mask_desc_init(mask_desc_t *mask_desc,
        const memory_desc_t *src_md, const memory_desc_t *dst_md, 
        const memory_desc_t *mask_md, double value_f, double value_i) {
    VCHECK_TRANSPOSE(!any_null(src_md, dst_md, mask_md), VERBOSE_NULL_ARG);
    
    auto op_d = mask_desc_t();
    op_d.primitive_kind = primitive_kind::mask;
    op_d.src_desc = *src_md;
    op_d.dst_desc = *dst_md;
    op_d.mask_desc = *mask_md;
    op_d.value_f = value_f;
    op_d.value_i = value_i;

    *mask_desc = op_d;
    return status::success;
}

} // namespace impl
} // namespace dnnl

status_t dnnl_mask_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, dnnl_engine_t engine, 
        const memory_desc_t *src_md, const memory_desc_t *dst_md, 
        const memory_desc_t *mask_md, 
        double value) {
    auto mask_desc = mask_desc_t();
    CHECK(mask_desc_init(
            &mask_desc, src_md, dst_md, mask_md, value, 0));
	
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&mask_desc, nullptr, nullptr);
}

status_t dnnl_mask_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, dnnl_engine_t engine, 
        const memory_desc_t *src_md, const memory_desc_t *dst_md, 
        const memory_desc_t *mask_md, 
        int64_t value) {
    auto mask_desc = mask_desc_t();
    CHECK(mask_desc_init(
            &mask_desc, src_md, dst_md, mask_md, 0, value));
	
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&mask_desc, nullptr, nullptr);
}