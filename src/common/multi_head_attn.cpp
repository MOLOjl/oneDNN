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
#include "oneapi/dnnl/dnnl.h"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

status_t dnnl_multi_head_attn_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t alg_kind, int num_heads, double softmax_scaler, 
        const memory_desc_t *devSeqLengthsQO_desc, const memory_desc_t *devSeqLengthsKV_desc, 
        const memory_desc_t *queries_desc, int* q_axes, int* seqlength_Q, 
        const memory_desc_t *residuals_desc, const memory_desc_t *keys_desc, int* k_axes, 
        int* seqlength_K, const memory_desc_t *values_desc, int* v_axes, int* seqlength_V, 
        const memory_desc_t *out_desc, int* o_axes, int* seqlength_O, 
        const memory_desc_t *qweight_desc, const memory_desc_t *qbias_desc, 
        const memory_desc_t *kweight_desc, const memory_desc_t *kbias_desc, 
        const memory_desc_t *vweight_desc, const memory_desc_t *vbias_desc, 
        const memory_desc_t *oweight_desc, const memory_desc_t *obias_desc, 
        int* p_currIdx, int* loWinIdx, int* hiWinIdx, float dropout, float postdropout, 
        unsigned long long seed, unsigned long long postseed,
        const primitive_attr_t *attr) {
    
    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;

    auto attn_desc = multi_head_attn_desc_t();
    attn_desc.primitive_kind = primitive_kind::multi_head_attn;
    attn_desc.prop_kind = prop_kind;
    attn_desc.alg_kind = alg_kind;

    attn_desc.num_heads = num_heads;
    attn_desc.softmax_scaler = softmax_scaler;
    attn_desc.devSeqLengthsQO_desc = *devSeqLengthsQO_desc;
    attn_desc.devSeqLengthsKV_desc = *devSeqLengthsKV_desc;
    attn_desc.queries_desc = *queries_desc;
    attn_desc.q_axes = q_axes;
    attn_desc.seqlength_Q = seqlength_Q;

    if(residuals_desc != nullptr)
        attn_desc.residuals_desc = *residuals_desc;

    attn_desc.keys_desc = *keys_desc;
    attn_desc.k_axes = k_axes;
    attn_desc.seqlength_K = seqlength_K;
    attn_desc.values_desc = *values_desc;
    attn_desc.v_axes = v_axes;
    attn_desc.seqlength_V = seqlength_V;
    attn_desc.out_desc = *out_desc;
    attn_desc.o_axes = o_axes;
    attn_desc.seqlength_O = seqlength_O;

    if(qweight_desc != nullptr)
        attn_desc.qweight_desc = *qweight_desc;
    if(qbias_desc != nullptr)
        attn_desc.qbias_desc = *qbias_desc;
    if(kweight_desc != nullptr)
        attn_desc.kweight_desc = *kweight_desc;
    if(kbias_desc != nullptr)
        attn_desc.kbias_desc = *kbias_desc;
    if(vweight_desc != nullptr)
        attn_desc.vweight_desc = *vweight_desc;
    if(vbias_desc != nullptr)
        attn_desc.vbias_desc = *vbias_desc;
    if(oweight_desc != nullptr)
        attn_desc.oweight_desc = *oweight_desc;
    if(obias_desc != nullptr)
        attn_desc.obias_desc = *obias_desc;

    attn_desc.p_currIdx = p_currIdx;
    attn_desc.loWinIdx = loWinIdx;
    attn_desc.hiWinIdx = hiWinIdx;

    attn_desc.dropout = dropout;
    attn_desc.postdropout = postdropout;
    attn_desc.dropout = seed;
    attn_desc.postdropout = postseed;

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&attn_desc, nullptr, attr);
}

status_t dnnl_multi_head_attn_forward_primitive_desc_create_rocm(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, int num_heads, double softmax_scaler, 
        const memory_desc_t *queries_desc, int* q_axes, 
        const memory_desc_t *residuals_desc, const memory_desc_t *keys_desc, 
        int* k_axes, const memory_desc_t *values_desc, int* v_axes, 
        const memory_desc_t *out_desc, int* o_axes, 
        const memory_desc_t *qweight_desc, const memory_desc_t *qbias_desc, 
        const memory_desc_t *kweight_desc, const memory_desc_t *kbias_desc, 
        const memory_desc_t *vweight_desc, const memory_desc_t *vbias_desc, 
        const memory_desc_t *oweight_desc, const memory_desc_t *obias_desc, 
        float dropout, float postdropout, unsigned long long seed, 
        unsigned long long postseed, const primitive_attr_t *attr) {
    
    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;

    auto attn_desc = multi_head_attn_desc_t();
    attn_desc.primitive_kind = primitive_kind::multi_head_attn;
    attn_desc.prop_kind = prop_kind;

    attn_desc.num_heads = num_heads;
    attn_desc.softmax_scaler = softmax_scaler;
    attn_desc.queries_desc = *queries_desc;
    attn_desc.q_axes = q_axes;

    if(residuals_desc != nullptr)
        attn_desc.residuals_desc = *residuals_desc;

    attn_desc.keys_desc = *keys_desc;
    attn_desc.k_axes = k_axes;
    attn_desc.values_desc = *values_desc;
    attn_desc.v_axes = v_axes;
    attn_desc.out_desc = *out_desc;
    attn_desc.o_axes = o_axes;

    if(qweight_desc != nullptr)
        attn_desc.qweight_desc = *qweight_desc;
    if(qbias_desc != nullptr)
        attn_desc.qbias_desc = *qbias_desc;
    if(kweight_desc != nullptr)
        attn_desc.kweight_desc = *kweight_desc;
    if(kbias_desc != nullptr)
        attn_desc.kbias_desc = *kbias_desc;
    if(vweight_desc != nullptr)
        attn_desc.vweight_desc = *vweight_desc;
    if(vbias_desc != nullptr)
        attn_desc.vbias_desc = *vbias_desc;
    if(oweight_desc != nullptr)
        attn_desc.oweight_desc = *oweight_desc;
    if(obias_desc != nullptr)
        attn_desc.obias_desc = *obias_desc;

    attn_desc.dropout = dropout;
    attn_desc.postdropout = postdropout;
    attn_desc.dropout = seed;
    attn_desc.postdropout = postseed;

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&attn_desc, nullptr, attr);
}

status_t dnnl_multi_head_attn_backward_data_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;

    auto attn_desc = multi_head_attn_desc_t();
    attn_desc.primitive_kind = primitive_kind::multi_head_attn;
    attn_desc.prop_kind = prop_kind::backward_data;

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&attn_desc, hint_fwd_pd, attr);

}

status_t dnnl_multi_head_attn_backward_weights_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t WgradMode_alg, 
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;

    auto attn_desc = multi_head_attn_desc_t();
    attn_desc.primitive_kind = primitive_kind::multi_head_attn;
    attn_desc.prop_kind = prop_kind::backward_weights;
    attn_desc.wgrad_alg_kind = WgradMode_alg;

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&attn_desc, hint_fwd_pd, attr);

}
