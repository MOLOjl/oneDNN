/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_EMBEDDING(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, embedding, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

status_t dnnl_embedding_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *src_md, const memory_desc_t *dict_md, 
        const memory_desc_t *dst_md) {
    VCHECK_EMBEDDING(!any_null(src_md, dict_md, dst_md), VERBOSE_NULL_ARG);

    auto op_d = embedding_desc_t();
    op_d.primitive_kind = primitive_kind::embedding;
    op_d.src_desc = *src_md;
    op_d.dict_desc = *dict_md;
    op_d.dst_desc = *dst_md;

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&op_d, nullptr, nullptr);
}
