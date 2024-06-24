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

#define VCHECK_WHERE(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, dnnl_where, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

status_t where_desc_init(where_desc_t *where_desc,
        const memory_desc_t *cond_md, const memory_desc_t *src1_md, 
        const memory_desc_t *src2_md, const memory_desc_t *dst_md) {
    VCHECK_WHERE(!any_null(cond_md, src1_md, src2_md, dst_md), VERBOSE_NULL_ARG);
    
    auto op_d = where_desc_t();
    op_d.primitive_kind = primitive_kind::where;
    op_d.cond_desc = *cond_md;
    op_d.src1_desc = *src1_md;
    op_d.src2_desc = *src2_md;
    op_d.dst_desc = *dst_md;

    *where_desc = op_d;
    return status::success;
}

} // namespace impl
} // namespace dnnl

status_t dnnl_where_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, dnnl_engine_t engine, 
        const memory_desc_t *cond_md, const memory_desc_t *src1_md, 
        const memory_desc_t *src2_md, const memory_desc_t *dst_md) {
    auto where_desc = where_desc_t();
    CHECK(where_desc_init(&where_desc, cond_md, src1_md, src2_md, dst_md));
	
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&where_desc, nullptr, nullptr);
}