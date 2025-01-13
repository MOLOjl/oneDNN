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

#ifndef COMMON_MULTI_HEAD_ATTN_PD_HPP
#define COMMON_MULTI_HEAD_ATTN_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

// #define VDISPATCH_ATTN(cond, msg, ...) \
//     VCONDCHECK(primitive, create, dispatch, multi_head_attn, (cond), \
//             status::unimplemented, "%s," msg, this->info(engine), \
//             ##__VA_ARGS__)

// #define VDISPATCH_ATTN_SC(f, msg, ...) \
//     VCHECK(primitive, create, dispatch, multi_head_attn, (f), "%s," msg, \
//             this->info(engine), ##__VA_ARGS__)

// #define VDISPATCH_ATTN_IC(cond, msg, ...) \
//     VCONDCHECK(primitive, create, dispatch, multi_head_attn, (cond), \
//             status::unimplemented, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {

struct multi_head_attn_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::multi_head_attn;

    typedef multi_head_attn_pd_t base_class;
    typedef multi_head_attn_pd_t hint_class;

    const multi_head_attn_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }
    
    bool querymap_all_to_one() const {
        return desc()->alg_kind == alg_kind::attn_querymap_all2one;
    }

    bool enable_proj_bias() const {
        if(desc()->qbias_desc.ndims != 0)
            return true;
        if(desc()->kbias_desc.ndims != 0)
            return true;
        if(desc()->vbias_desc.ndims != 0)
            return true;
        if(desc()->obias_desc.ndims != 0)
            return true;
        return false;
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool is_bwd_d() const {
        return desc_.prop_kind == prop_kind::backward_data;
    }

    bool is_bwd_w() const {
        return desc_.prop_kind == prop_kind::backward_weights;
    }

    const int* currIdx() const {
        return desc()->p_currIdx;
    }

    const int* loWinIdxArray() const {
        return desc()->loWinIdx;
    }

    const int* hiWinIdxArray() const {
        return desc()->hiWinIdx;
    }

    const memory_desc_t* weight_md(int idx) const {
        switch (idx)
        {
        case 0:
            return &desc()->qweight_desc;
            break;
        case 1:
            return &desc()->qbias_desc;
            break;
        case 2:
            return &desc()->kweight_desc;
            break;
        case 3:
            return &desc()->kbias_desc;
            break;
        case 4:
            return &desc()->vweight_desc;
            break;
        case 5:
            return &desc()->vbias_desc;
            break;
        case 6:
            return &desc()->oweight_desc;
            break;
        case 7:
            return &desc()->obias_desc;
            break;
        default:
            return nullptr;
            break;
        }
    }
    
    const memory_desc_t* query_md() const {
        return &desc()->queries_desc;
    }

    const int* query_axes() const {
        return desc()->q_axes;
    }

    const memory_desc_t* key_md() const {
        return &desc()->queries_desc;
    }

    const int* key_axes() const {
        return desc()->k_axes;
    }

    const memory_desc_t* value_md() const {
        return &desc()->keys_desc;
    }

    const int* value_axes() const {
        return desc()->v_axes;
    }

    const memory_desc_t* output_md() const {
        return &desc()->out_desc;
    }

    const int* output_axes() const {
        return desc()->o_axes;
    }

    int num_heads() const {
        return desc()->num_heads;
    }

    double softmax_scaler() const {
        return desc()->softmax_scaler;
    }

    int* seqlength_Q() const {
        return desc()->seqlength_Q;
    }

    int* seqlength_K() const {
        return desc()->seqlength_K;
    }

    int* seqlength_V() const {
        return desc()->seqlength_V;
    }

    int* seqlength_O() const {
        return desc()->seqlength_O;
    }

    float attn_dropout() const {
        return desc()->dropout;
    }

    float post_attn_dropout() const {
        return desc()->postdropout;
    }

    float dropout_seed() const {
        return desc()->seed;
    }

    float post_dropout_seed() const {
        return desc()->postseed;
    }

    bool addGrad() const {
        return desc()->wgrad_alg_kind == alg_kind::attn_wgrad_add;
    }

    void set_attnDesc(void* ad) const {
        desc_.attnDesc = ad;
    }

    void* get_attnDesc() const {
        return desc()->attnDesc;
    }

    void set_SeqDataDesc(void* seqdatadesc, int idx) const {
        desc_.SeqDataDescs[idx] = seqdatadesc;
    }

    void* get_SeqDataDesc(int idx) const {
        return desc()->SeqDataDescs[idx];
    }

    void set_weightbias_tdesc(void* wb_tdesc, int idx) const {
        desc_.weightbias_tdesc[idx] = wb_tdesc;
    }

    void* get_weightbias_tdesc(int idx) const {
        return desc()->weightbias_tdesc[idx];
    }

    void set_weightbias_size(size_t wb_size, int idx) const {
        desc_.weightbias_size[idx] = wb_size;
    }

    size_t get_weightbias_size(int idx) const {
        return desc()->weightbias_size[idx];
    }

    void set_reserveSpaceSizeInBytes(size_t rs_size) const {
        desc_.reserveSpaceSizeInBytes = rs_size;
    }

    size_t get_reserveSpaceSizeInBytes() const {
        return desc()->reserveSpaceSizeInBytes;
    }

    void set_workSpaceSizeInBytes(size_t ws_size) const {
        desc_.workSpaceSizeInBytes = ws_size;
    }

    size_t get_workSpaceSizeInBytes() const {
        return desc()->workSpaceSizeInBytes;
    }

    void set_weightSizeInBytes(size_t ws_size) const {
        desc_.weightSizeInBytes = ws_size;
    }

    size_t get_weightSizeInBytes() const {
        return desc()->weightSizeInBytes;
    }

    void set_weightspace(void* ws_p) const {
        desc_.weightspace = ws_p;
    }

    void* get_weightspace() const {
        return desc()->weightspace;
    }

    void set_workspace(void* ws_p) const {
        desc_.workspace = ws_p;
    }

    void* get_workspace() const {
        return desc()->workspace;
    }
    
    void set_reservespace(void* rs_p) const {
        desc_.reservespace = rs_p;
    }

    void* get_reservespace() const {
        return desc()->reservespace;
    }

    void set_offsets(size_t* offsets_) const {
        for(int i=0; i<31; i++)
            desc_.offsets[i] = offsets_[i];
    }

    const size_t* get_offsets() const {
        return desc()->offsets;
    }

    void set_dropDesc(int idx, void* DropoutDesc) const {
        if(idx == 0)
            desc_.attnDropoutDesc = DropoutDesc;
        if(idx == 1)
            desc_.postDropoutDesc = DropoutDesc;
    }

    void* get_dropDesc(int idx) const {
        if(idx == 0)
            return desc()->attnDropoutDesc;
        if(idx == 1)
            return desc()->postDropoutDesc;
        return nullptr;
    }

protected:
    mutable multi_head_attn_desc_t desc_;

    multi_head_attn_pd_t(const multi_head_attn_desc_t *adesc,
            const primitive_attr_t *attr,
            const multi_head_attn_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind) {
            if(hint_fwd_pd) {
                desc_ = *(hint_fwd_pd->desc());
                desc_.prop_kind = adesc->prop_kind;
                desc_.wgrad_alg_kind = adesc->wgrad_alg_kind;
            }
            else
                desc_ = *adesc;
        }
};

} // namespace impl
} // namespace dnnl

#endif
