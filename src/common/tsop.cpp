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

#define VCHECK_TSOP(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, dnnl_tsop, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

status_t dnnl_tsop_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, dnnl_engine_t engine, 
        alg_kind_t alg_kind, const memory_desc_t *src_md, const memory_desc_t *dst_md,
        double* vd, int64_t* vi, bool* vb) {
    auto op_d = tsop_desc_t();

    op_d.primitive_kind = primitive_kind::tsop;
    op_d.alg_kind = alg_kind;
    op_d.src_desc = *src_md;
    op_d.dst_desc = *dst_md;
    op_d.v3 = std::make_tuple(vd, vi, vb);

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&op_d, nullptr, nullptr);
}