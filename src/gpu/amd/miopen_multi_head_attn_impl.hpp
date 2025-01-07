/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_AMD_MIOPEN_MULTI_HEAD_ATTN_IMPL_HPP
#define GPU_AMD_MIOPEN_MULTI_HEAD_ATTN_IMPL_HPP

#include <miopen/miopen.h>
#include "rocblas/rocblas.h"

#include "common/c_types_map.hpp"
#include "common/multi_head_attn_pd.hpp"
#include "common/utils.hpp"
#include "gpu/amd/engine.hpp"
#include "gpu/amd/stream.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

#include "gpu/amd/custom/hip_customs.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_multi_head_attn_impl_base_t {
protected:
	enum io { q = 0, k, v, o, NUM_IO };

	miopenDropoutDescriptor_t attnDropoutDesc;
	miopenDropoutDescriptor_t postDropoutDesc;
    miopenTensorDescriptor_t midtensor_desc; // For softmax, dropout, mask and scale.
    miopenTensorDescriptor_t qo_desc;   // q and o projection tensor desc 
    miopenTensorDescriptor_t kv_desc;   // k and v projection tensor desc 
    miopenTensorDescriptor_t bias_desc;  // qkvo projection bias tensor desc 

    int ndims_qkvo;
    int batch_size = 1; // q, k, v, o share the same bacth size
    int embed_dim, kdim, vdim;
    int seq_length_L, seq_length_S;
    int num_heads;
    double smScaler;
    // default input layout should be [0,1,2,3], ordered accroding to cudnn's axes.
    // Assume tranB in matmul(projection) is always true.

    rocblas_datatype data_types[NUM_IO];
    size_t dtype_bytesize[NUM_IO];
    rocblas_datatype weight_type;
    rocblas_datatype compute_type;
    miopenDataType_t midtensor_dtype;
    miopenDataType_t qo_dtype;
    miopenDataType_t kv_dtype;
    bool type_initialized;
    bool weight_enabled[8];

	size_t reserveSpaceSizeInBytes;
	size_t workSpaceSizeInBytes;
	size_t Dropout_stateSize;

    void* workspace;
    void* reservespace;

    bool is_fwd;
    bool is_bwd_data;
    bool is_bwd_weight;

    float alpha_f32 = 1.0;
    float beta_f32 = 0;
    int16_t alpha_f16 = 0x3C00;
    int16_t beta_f16 = 0;

    size_t q_proj_offset;
    size_t q_proj_t_offset;
    size_t k_proj_offset;
    size_t k_proj_t_offset;
    size_t v_proj_offset;
    size_t v_proj_t_offset;
    size_t mid_s_offset;
    size_t s_buffer_offset;
    size_t o_proj_offset;
    size_t o_proj_t_offset;
    size_t dAQ_offset;
    size_t dAK_offset;
    size_t dAS_offset;
    size_t dASb_offset;
    size_t dAV_offset;
    size_t dAO_offset;
    
    // backward
    size_t dout_offset;
    size_t doproj_offset;
    size_t doproj_t_offset;
    size_t ds_offset;
    size_t dvproj_offset;
    size_t dvproj_t_offset;
    size_t dqproj_offset;
    size_t dqproj_t_offset;
    size_t dkproj_offset;
    size_t dkproj_t_offset;
    size_t ddAQ_offset;
    size_t ddAK_offset;
    size_t ddAS_offset;
    size_t ddAV_offset;
    size_t ddAO_offset;
public:
    virtual ~miopen_multi_head_attn_impl_base_t() {
        if(attnDropoutDesc != nullptr){
		    MIOPEN_EXECUTE_FUNC_V(miopenDestroyDropoutDescriptor, attnDropoutDesc);
            attnDropoutDesc = nullptr;
        }
        if(postDropoutDesc != nullptr){
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyDropoutDescriptor, postDropoutDesc);
            attnDropoutDesc = nullptr;
        }
    }

    bool with_reserveSpace() const { return reserveSpaceSizeInBytes > 0; }

	// MHA always need scratchpad(workSpace).
	bool with_scratchpad() const { return true; } 

    const void *get_gemm_alpha() const {
        switch (compute_type) {
            case rocblas_datatype::rocblas_datatype_f16_r:
                return reinterpret_cast<const void *>(&alpha_f16);
            case rocblas_datatype::rocblas_datatype_f32_r:
                return reinterpret_cast<const void *>(&alpha_f32);
            default: assert(!"unknown acc type"); return nullptr;
        }
    }

    const void *get_gemm_beta() const {
        switch (compute_type) {
            case rocblas_datatype::rocblas_datatype_i32_r:
                return reinterpret_cast<const void *>(&beta_f16);
            case rocblas_datatype::rocblas_datatype_f32_r:
                return reinterpret_cast<const void *>(&beta_f32);
            default: assert(!"unknown acc type"); return nullptr;
        }
    }

    status_t get_dtype_bytesize(rocblas_datatype &blas_dt, size_t &bytesize){
        switch (compute_type) {
            case rocblas_datatype::rocblas_datatype_f16_r:
                bytesize = 2;
                return status::success;
            case rocblas_datatype::rocblas_datatype_f32_r:
                bytesize = 4;
                return status::success;
            default: assert(!"unsupport data type"); return status::invalid_arguments;
        }
    }

    status_t get_rocblas_data_type(
            dnnl_data_type_t data_type, rocblas_datatype &blas_dt) {
        switch (data_type) {
            case dnnl_data_type_t::dnnl_f32:
                blas_dt = rocblas_datatype_f32_r;
                return status::success;
            case dnnl_data_type_t::dnnl_f16:
                blas_dt = rocblas_datatype_f16_r;
                return status::success;
            case dnnl_data_type_t::dnnl_s8:
                blas_dt = rocblas_datatype_i8_r;
                return status::success;
            case dnnl_data_type_t::dnnl_s32:
                blas_dt = rocblas_datatype_i32_r;
                return status::success;
            case dnnl_data_type_t::dnnl_bf16:
                blas_dt = rocblas_datatype_bf16_r;
                return status::success;
            default: return status::unimplemented;
        }
        return status::unimplemented;
    }


    status_t check_qkvo_layout(io idx, memory_desc_t* md, int* axes) {
        bool ok = true;
        // TODO: support transpositon for io tensor
        if(axes) {
            // seq_len must be at the first dimension.
            ok = ok && (axes[0] == 0); 
            ok = ok && (axes[1] == 1);
            ok = ok && (axes[2] == 2);
            if(!ok) return status::invalid_arguments;
        }
        else
            return status::invalid_arguments;

        // // calculate size of QKVO.
        // size_QKVO[idx] = 1;
        // for(int j=0; j<ndims_qkvo; j++){
        //     size_QKVO *= md->dims[j];
        // }
        ndims_qkvo = (int)(md->ndims);
        if(idx == io::q) {
            embed_dim = md->dims[ndims_qkvo-1];
            seq_length_L = md->dims[0];
            for(int i=1; i<ndims_qkvo-1; i++)
                batch_size *= q_dims[i];
        }
        else {
            if(ndims_qkvo != (int)(md->ndims))
                return status::invalid_arguments;
            if(idx == io::k){
                kdim = md->dims[ndims_qkvo-1];
                seq_length_S = md->dims[0];
            }

            if(idx == io::v) {
                vdim = md->dims[ndims_qkvo-1];
                if(seq_length_S != md->dims[0])
                    return status::invalid_arguments;
            }

            if(idx == io::o){
                if(embed_dim != md->dims[ndims_qkvo-1])
                    return status::invalid_arguments;
                if(seq_length_L = md->dims[0])
                    return status::invalid_arguments;
            }

            int64_t product = 1;
            for(int i=1; i<ndims_qkvo-1; i++)
                product *= q_dims[i];
            if(product != batch_size)
                return status::invalid_arguments;
        }
        return status::success;
    }

	status_t configure_parameters(const multi_head_attn_pd_t *pd) {
        // check and set batchsize, embed_dim, kdim. vdim, seq_len_L, seq_len_S.
		CHECK(check_qkvo_layout(io::q, pd->query_md(), pd->query_axes()));
		CHECK(check_qkvo_layout(io::k, pd->key_md(), pd->key_axes()));
		CHECK(check_qkvo_layout(io::v, pd->value_md(), pd->value_axes()));
		CHECK(check_qkvo_layout(io::o, pd->output_md(), pd->output_axes()));

		// convert datatype
        CHECK(convert_data_type(pd->query_md()->data_type, &midtensor_dtype));
        CHECK(convert_data_type(pd->key_md()->data_type, &kv_dtype));
        CHECK(convert_data_type(pd->query_md()->data_type, &qo_dtype));
        CHECK(get_rocblas_data_type(pd->query_md()->data_type, &data_types[0]));
        CHECK(get_rocblas_data_type(pd->key_md()->data_type, &data_types[1]));
        CHECK(get_rocblas_data_type(pd->value_md()->data_type, &data_types[2]));
        CHECK(get_rocblas_data_type(pd->output_md()->data_type, &data_types[3]));

        // check if qkvo dtype all same.
        rocblas_datatype qdt = data_types[0];
		for(int i=1; i<NUM_IO; i++){
			if(data_types[i] != qdt)
				return status::invalid_arguments;
		}
        compute_type = qdt;

		num_heads = pd->num_heads();
		smScaler = pd->softmax_scaler();

        attndropout = pd->attn_dropout();
        postattndropout = pd->post_attn_dropout();
        dropoutseed = pd->dropout_seed();
        postdropoutseed = pd->post_dropout_seed();

        return status::success;
    }

    // Check the shape of weights and biases.
    status_t check_proj_weight(const multi_head_attn_pd_t* pd){
        for(int i=0; i<8; i++){
            auto wb_md = *(pd->weight_md(i));
            // It means this weight/bias is disabled.
            if(wb_md.ndims == 0) {
                weight_enabled[i] = false;
                continue;                
            }
            weight_enabled[i] = true;

            // check weight or bias data type.
            rocblas_datatype wb_dt;
            CHECK(get_rocblas_data_type(wb_md.data_type, &wb_dt));
            if(!type_initialized){
                weight_type = wb_dt;
                type_initialized = true;
            }
            else if(weight_type != wb_dt)
                return status::invalid_arguments;
            
            // create and set weight or bias tensor descriptor.
            // weight[nHeads*projected size, original size], bias[nHeads*projected size, 1]
            int dim_wb[2] = {1, 1};
            for(int j=0; j<wb_md.ndims; j++){
                if(j == wb_md.ndims - 1)
                    dim_wb[1] = wb_md.dims[j];
                else
                    dim_wb[0] = dim_wb[0]*wb_md.dims[j];
            }
            
            if(dim_wb[0] != embed_dim || dim_wb[0]%num_heads != 0)
                return status::invalid_arguments;

            if(i == 0 && dim_wb[1] != embed_dim)
                return status::invalid_arguments;
            if(i == 2 && dim_wb[1] != kdim)
                return status::invalid_arguments;
            if(i == 4 && dim_wb[1] != vdim)
                return status::invalid_arguments;
            if(i == 6 && dim_wb[1] != embed_dim)
                return status::invalid_arguments;

            auto data_size = types::data_type_size(wb_md.data_type);
        }
        return status::success;
    }
    
    virtual status_t init(impl::engine_t *engine, multi_head_attn_pd_t *pd) {
        is_fwd = pd->is_fwd();
        is_bwd_data = pd->is_bwd_d();
        is_bwd_weight = pd->is_bwd_w();

        return status::success;
    }

    virtual void execute(rocblas_handle rocblas_handle, miopenHandle_t miopen_handle, 
            const std::vector<void *> &args) const = 0;
};

struct miopen_multi_head_attn_fwd_impl_t : public miopen_multi_head_attn_impl_base_t {
protected:
    unsigned attnMode;
	// Compute precision.
    // miopenDataType_t computePrec = CUDNN_DATA_FLOAT;
    memory_desc_t dnnl_descs[NUM_IO];

    int size_QKVO[NUM_IO];
    // If projsize is set to 0, it means the coresponding project is disabled.
    int projsize_QKVO[NUM_IO];

    int nHeads;
	double smScaler;

    float attndropout;
    float postattndropout;
    unsigned long long dropoutseed;
    unsigned long long postdropoutseed;
public:
    virtual ~cudnn_multi_head_attn_fwd_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, midtensor_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, qo_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, kv_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, bias_desc);
    }

    status_t init(impl::engine_t *engine, multi_head_attn_pd_t *pd) override {
        CHECK(miopen_multi_head_attn_impl_base_t::init(engine, pd));

        CHECK(configure_parameters(pd));
        CHECK(check_proj_weight(pd));
        CHECK(create_miopen_descs(engine, pd));
        CHECK(init_scratchpad(engine, pd));
        return status::success;
    }

    // mid tensor desc and droupout desc
    status_t create_miopen_descs(impl::engine_t *engine, const multi_head_attn_pd_t *pd) {
        auto &sycl_engine = *utils::downcast<amd::engine_t *>(engine);
        impl::stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<amd::stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        // MIOPEN_SOFTMAX_MODE_INSTANCE 
        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &midtensor_desc);
        // apply softmax to 'seq_length_S' dimension, 
        // and since miopen use column-first, do reshape and index transpose.
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, midtensor_desc, midtensor_dtype,
                1, seq_length_S, batch_size*num_heads*seq_length_L, 1);
        // MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, midtensor_desc, midtensor_dtype,
        //         batch_size*num_heads*seq_length_L, seq_length_S, 1, 1);

		// dropouts
		MIOPEN_EXECUTE_FUNC_V(miopenCreateDropoutDescriptor, &attnDropoutDesc);
		MIOPEN_EXECUTE_FUNC_V(miopenCreateDropoutDescriptor, &postDropoutDesc);
        pd->set_dropDesc(0, attnDropoutDesc);
        pd->set_dropDesc(1, postDropoutDesc);

        size_t size1, size2;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetReserveSpaceSize, midtensor_desc, &size1);
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetReserveSpaceSize, qo_desc, &size2);
        reserveSpaceSizeInBytes = size1 + size2;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetStatesSize, handle, &Dropout_stateSize);

        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &qo_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, qo_desc, qo_dtype,
                1, seq_length_L, batch_size, embed_dim);
        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &kv_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, kv_desc, kv_dtype,
                1, seq_length_S, batch_size, embed_dim);
        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &bias_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, bias_desc, qo_dtype,
                1, 1, 1, embed_dim);

        return status::success;
    }

    void get_workspace(multi_head_attn_pd_t *pd){
        workSpaceSizeInBytes = 0;
        for(int i=0; i<NUM_IO; i++)
            CHECK(get_dtype_bytesize(data_types[i], dtype_bytesize[i]));

        std::vector<size_t> offsets;

        size_t QO_proj_bytesize = dtype_bytesize[io::q] * seq_length_L * batch_size * embed_dim;
        size_t KV_proj_bytesize = dtype_bytesize[io::k] * seq_length_S * batch_size * embed_dim;

        // output buffers of QKV project and their transposes
        q_proj_offset = 0;
        size_t q_proj = weight_enabled[0] ? QO_proj_bytesize : 0;

        q_proj_t_offset = q_proj_offset + q_proj;
        size_t q_proj_tran = QO_proj_bytesize;
        
        k_proj_offset = q_proj_t_offset + q_proj_tran;
        size_t k_proj = weight_enabled[2] ? KV_proj_bytesize : 0;
        
        k_proj_t_offset = k_proj_offset + k_proj;
        size_t k_proj_tran = KV_proj_bytesize;

        v_proj_offset = k_proj_t_offset + k_proj_tran;
        size_t v_proj = weight_enabled[4] ? KV_proj_bytesize : 0;

        v_proj_t_offset = v_proj_offset + v_proj;
        size_t v_proj_tran = KV_proj_bytesize;

        // middle matrix s = q x k^T
        mid_s_offset = v_proj_t_offset + v_proj_tran;
        size_t mid_s = dtype_bytesize[io::q] * (batch_size*num_heads) * seq_length_S * seq_length_L;
        
        s_buffer_offset = mid_s_offset + mid_s;
        if(pd->desc()->prop_kind == forward_inference)
            s_buffer_offset = mid_s_offset;        
        size_t s_buffer = mid_s;

        // device array store the pointers of batched matrices q' and k'
        dAQ_offset = s_buffer_offset + s_buffer;
        size_t dev_AQ = sizeof(void*)*batch_size*num_heads;

        dAK_offset = dAQ_offset + dev_AQ;
        size_t dev_AK = dev_AQ;

        dAS_offset = dAK_offset + dev_AK;
        size_t dev_AS = dev_AQ;
        
        dASb_offset = dAS_offset + dev_AS;
        size_t dev_ASb = dev_AQ;

        dAV_offset = dASb_offset + dev_ASb;
        size_t dev_AV = dev_AQ;

        dAO_offset = dAV_offset + dev_AV;
        size_t dev_AO = dev_AQ;

        o_proj_t_offset = dAO_offset + dev_AO;
        size_t o_proj_tran = QO_proj_bytesize;

        o_proj_offset = o_proj_t_offset + o_proj_tran;
        size_t o_proj = weight_enabled[6] ? QO_proj_bytesize : 0;
        
        offsets.push_back(q_proj_offset);
        offsets.push_back(q_proj_t_offset);
        offsets.push_back(k_proj_offset);
        offsets.push_back(k_proj_t_offset);
        offsets.push_back(v_proj_offset);
        offsets.push_back(v_proj_t_offset);
        offsets.push_back(mid_s_offset);
        offsets.push_back(s_buffer_offset);
        offsets.push_back(o_proj_offset);
        offsets.push_back(o_proj_t_offset);
        offsets.push_back(dAQ_offset);
        offsets.push_back(dAK_offset);
        offsets.push_back(dAS_offset);
        offsets.push_back(dASb_offset);
        offsets.push_back(dAV_offset);
        offsets.push_back(dAO_offset);

        if(pd->desc()->prop_kind == forward_training) {
            // backward data
            dout_offset = o_proj_offset + o_proj;   // for dropout
            size_t dout_buffer = QO_proj_bytesize;

            doproj_offset = dout_offset + dout_buffer;
            size_t doproj_buffer = o_proj;

            doproj_t_offset = doproj_offset + doproj_buffer;    // for transpose
            size_t doproj_t_buffer = QO_proj_bytesize;

            ds_offset = doproj_t_offset + doproj_t_buffer;
            size_t ds_buffer = mid_s;

            dvproj_t_offset = ds_offset + ds_buffer;
            size_t dvproj_t_buffer = KV_proj_bytesize;

            dvproj_offset = dvproj_t_offset + dvproj_t_buffer;
            size_t dvproj_buffer = v_proj;

            dqproj_t_offset = dvproj_offset + dvproj_buffer;
            size_t dqproj_t_buffer = QO_proj_bytesize;

            dqproj_offset = dqproj_t_offset + dqproj_t_buffer;
            size_t dqproj_buffer = q_proj;

            dkproj_t_offset = dqproj_offset + dqproj_buffer;
            size_t dkproj_t_buffer = KV_proj_bytesize;

            dkproj_offset = dkproj_t_offset + dkproj_t_buffer;
            size_t dkproj_buffer = k_proj;

            ddAQ_offset = dkproj_offset + dkproj_buffer;
            size_t dev_dAQ = sizeof(void*)*batch_size*num_heads;

            ddAK_offset = ddAQ_offset + dev_dAQ;
            size_t dev_dAK = dev_dAQ;

            ddAS_offset = ddAK_offset + dev_dAK;
            size_t dev_dAS = dev_dAQ;

            ddAV_offset = ddAS_offset + dev_dAS;
            size_t dev_dAV = dev_dAQ;

            ddAO_offset = ddAV_offset + dev_dAV;
            size_t dev_dAO = dev_dAQ;

            workSpaceSizeInBytes += ddAO_offset + dev_dAO;

            offsets.push_back(dout_offset);
            offsets.push_back(doproj_offset);
            offsets.push_back(doproj_t_offset);
            offsets.push_back(ds_offset);
            offsets.push_back(dvproj_offset);
            offsets.push_back(dvproj_t_offset);
            offsets.push_back(dqproj_offset);
            offsets.push_back(dqproj_t_offset);
            offsets.push_back(dkproj_offset);
            offsets.push_back(dkproj_t_offset);
            offsets.push_back(ddAQ_offset);
            offsets.push_back(ddAK_offset);
            offsets.push_back(ddAS_offset);
            offsets.push_back(ddAV_offset);
            offsets.push_back(ddAO_offset);
        }
        else {
            workSpaceSizeInBytes = o_proj_offset + o_proj;
        }
    }

    // Scratchpad will still be used in backward, which is different from other primitives.
    // Will set reservedSpace and state of droupouts, which will be used in backward.
    status_t init_scratchpad(impl::engine_t *engine, multi_head_attn_pd_t *pd) {        
        auto &sycl_engine = *utils::downcast<amd::engine_t *>(engine);
        impl::stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<amd::stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();
        
        if(reserveSpaceSizeInBytes > 0 && is_fwd)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_attn_reservespace, reserveSpaceSizeInBytes,
                    size_t(1));
        
		// dropouts
        if (Dropout_stateSize > 0){
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_attn_dropout_states,
                    Dropout_stateSize, size_t(1));
			pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_attn_post_dropout_states,
                    Dropout_stateSize, size_t(1));
		}

        get_workspace(pd);
        if(workSpaceSizeInBytes > 0 && is_fwd)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_attn_workspace, workSpaceSizeInBytes,
                    size_t(1));

        pd->set_workSpaceSizeInBytes(workSpaceSizeInBytes);
        pd->set_reserveSpaceSizeInBytes(reserveSpaceSizeInBytes);

        return status::success;
    }

    void set_batch_matrices(void* workspace, void*& d_AQ, void*& d_AK, void*& d_AS, void*& d_ASb, 
            void*& d_AV, void*& d_AO) {

        void **h_AQ, **h_AK, **h_AS, **h_ASb, **h_AV, **h_AO;
        int batch_count = batch_size*num_heads;
        h_AQ = (void**)malloc(sizeof(void*) * batch_count);
        h_AK = (void**)malloc(sizeof(void*) * batch_count);
        h_AS = (void**)malloc(sizeof(void*) * batch_count);
        h_ASb = (void**)malloc(sizeof(void*) * batch_count);
        h_AV = (void**)malloc(sizeof(void*) * batch_count);
        h_AO = (void**)malloc(sizeof(void*) * batch_count);
        for(int i=0; i<batch_count; i++) {
            h_AQ[i] = (char*)workspace + q_proj_t_offset + sizeof(void*)*i;
            h_AK[i] = (char*)workspace + k_proj_t_offset + sizeof(void*)*i;
            h_AS[i] = (char*)workspace + mid_s_offset + sizeof(void*)*i;
            // when prop_kind is forward_inference, s_buffer_offset eqs to mid_s_offset.
            h_ASb[i] = (char*)workspace + s_buffer_offset + sizeof(void*)*i;
            h_AV[i] = (char*)workspace + v_proj_t_offset + sizeof(void*)*i;
            h_AO[i] = (char*)workspace + o_proj_t_offset + sizeof(void*)*i;
        }

        d_AQ = (char*)workspace + dAQ_offset;
        d_AK = (char*)workspace + dAK_offset;
        d_AS = (char*)workspace + dAS_offset;
        d_ASb = (char*)workspace + dASb_offset;
        d_AV = (char*)workspace + dAV_offset;
        d_AO = (char*)workspace + dAO_offset;

        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_AQ, (HIPdeviceptr)h_AQ, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_AK, (HIPdeviceptr)h_AK, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_AS, (HIPdeviceptr)h_AS, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_ASb, (HIPdeviceptr)h_ASb, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_AV, (HIPdeviceptr)h_AV, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_AO, (HIPdeviceptr)h_AO, 
                batch_count*sizeof(void*), hipMemcpyDefault);

        free(h_AQ);
        free(h_AK);
        free(h_AS);
        free(h_ASb);
        free(h_AV);
        free(h_AO);
    }

    void execute(rocblas_handle rocblas_handle, miopenHandle_t miopen_handle, 
            const std::vector<void *> &args) const override {
        auto queries = args[0], residuals = args[1], keys = args[2], 
                values = args[3], out = args[4];
        
        float f32_alpha = 1.0;
        float f32_beta = 0;

        void* weightbias[8];
        for(int i=5; i<5+8; i++)
            weightbias[i-5] = args[i];
        // buffers
        workspace = args[13];
        reservespace = args[14];

        void* attn_dropout_states = args[15];
        void* attn_post_dropout_states = args[16];

        // dropout
        MIOPEN_EXECUTE_FUNC_V(miopenSetDropoutDescriptor, attnDropoutDesc, miopen_handle,
                attndropout, attn_dropout_states, Dropout_stateSize, dropoutseed, false, 
                false, MIOPEN_RNG_PSEUDO_XORWOW);
        MIOPEN_EXECUTE_FUNC_V(miopenSetDropoutDescriptor, postDropoutDesc, miopen_handle,
                postattndropout, attn_post_dropout_states, Dropout_stateSize, 
                postdropoutseed, dropoutseed, false, false, MIOPEN_RNG_PSEUDO_XORWOW);

        const void *alpha = get_gemm_alpha();
        const void *beta = get_gemm_beta();

        // project matmul for qkvo, no need to do batched matmul. 
        // and since rocblas using column-first default, do fake transpose.
        // project matmul for q, (seq_length_L*batch_size, embed_dim) x (embed_dim*h/h, embed_dim)^T
        void* q_proj = weight_enabled[0] ? (char*)workspace + q_proj_offset : queries;
        void* q_proj_tran = (char*)workspace + q_proj_t_offset;
        if(weight_enabled[0])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_transpose, 
                    rocblas_operation::rocblas_operation_none, 
                    embed_dim, seq_length_L*batch_size, embed_dim, alpha, 
                    weightbias[0], weight_type, embed_dim, 
                    queries, data_types[io::q], embed_dim, beta, 
                    q_proj, data_types[io::q], embed_dim, 
                    q_proj, data_types[io::q], embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

        if(weight_enabled[1])
            MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, miopen_handle, miopenTensorOpAdd, 
                    &f32_alpha, qo_desc, q_proj, &f32_alpha, bias_desc, weightbias[1], 
                    &f32_beta, qo_desc, q_proj);	


        // (reshape and) transpose q_proj
        size_t dims_q_proj[3] = {seq_length_L, batch_size*num_heads, embed_dim/num_heads};
        hip_custom::transpose(dtype_bytesize[io::q], q_proj, q_proj_tran, dims_q_proj, 3, 0, 1);

        // project matmul for k, (seq_length_S*batch_size, kdim) x (embed_dim*h/h, kdim)^T
        void* k_proj = weight_enabled[2] ? (char*)workspace + k_proj_offset : keys;
        void* k_proj_tran = (char*)workspace + k_proj_t_offset;
        if(weight_enabled[2])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_transpose, 
                    rocblas_operation::rocblas_operation_none, 
                    embed_dim, seq_length_S*batch_size, kdim, alpha, 
                    weightbias[2], weight_type, kdim, 
                    keys, data_types[io::k], kdim, beta, 
                    k_proj, data_types[io::k], embed_dim, 
                    k_proj, data_types[io::k], embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

        if(weight_enabled[3])
            MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, miopen_handle, miopenTensorOpAdd, 
                    &f32_alpha, kv_desc, k_proj, &f32_alpha, bias_desc, weightbias[3], 
                    &f32_beta, kv_desc, k_proj);

        // (reshape and) transpose k_proj, another transpoe will be done during the matmul of QK^T.
        size_t dims_k_proj[3] = {seq_length_S, batch_size*num_heads, embed_dim/num_heads};
        hip_custom::transpose(dtype_bytesize[io::k], k_proj, k_proj_tran, dims_k_proj, 3, 0, 1);

        // project matmul for v, (seq_length_S*batch_size, vdim) x (embed_dim*h/h, vdim)^T
        void* v_proj = weight_enabled[4] ? (char*)workspace + v_proj_offset : values;
        void* v_proj_tran = (char*)workspace + v_proj_t_offset;
        if(weight_enabled[4])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_transpose, 
                    rocblas_operation::rocblas_operation_none, 
                    embed_dim, seq_length_S*batch_size, vdim, alpha, 
                    weightbias[4], weight_type, vdim, 
                    values, data_types[io::v], vdim, beta, 
                    v_proj, data_types[io::v], embed_dim, 
                    v_proj, data_types[io::v], embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

        if(weight_enabled[5])
            MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, miopen_handle, miopenTensorOpAdd, 
                    &f32_alpha, kv_desc, v_proj, &f32_alpha, bias_desc, weightbias[5], 
                    &f32_beta, kv_desc, v_proj);

        // (reshape and) transpose v_proj
        size_t dims_v_proj[3] = {seq_length_S, batch_size*num_heads, embed_dim/num_heads};
        hip_custom::transpose(dtype_bytesize[io::v], v_proj, v_proj_tran, dims_v_proj, 3, 0, 1);

        int head_dim = embed_dim / num_heads;
        // Q(batch*h, seqlen_L, embed_dim/h) x K(batch*h, seqlen_S, embed_dim/h)^T
        // alloc and set matrices array
        void *d_AQ, *d_AK, *d_AS, *d_ASb, *d_AV, *d_AO;
        set_batch_matrices(workspace, d_AQ, d_AK, d_AS, d_ASb, d_AV, d_AO);
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_transpose, 
                rocblas_operation::rocblas_operation_none, 
                seq_length_S, seq_length_L, head_dim, alpha,
                d_AK, data_types[io::k], head_dim, 
                d_AQ, data_types[io::q], head_dim, beta,
                d_AS, data_types[io::k], seq_length_S, 
                d_AS, data_types[io::k], seq_length_S, batch_size*num_heads,
                compute_type, rocblas_gemm_algo_standard, 0);
        
        // softmax
        void* mid_tensor_s = (char*)workspace + mid_s_offset;
        MIOPEN_EXECUTE_FUNC_V(miopenSoftmaxForward, miopen_handle, alpha, midtensor_desc, 
            mid_tensor_s, beta, midtensor_desc, mid_tensor_s);

        // since backward need mid_tensor_s, it can't be overwrote.
        // when prop_kind is forward_inference, s_buffer_offset eqs to mid_s_offset.
        void* s_buffer = (char*)workspace + s_buffer_offset;

        // TODO: scale

        // attn dropout
        size_t attn_reserve_size;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetReserveSpaceSize, midtensor_desc, &attn_reserve_size);
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutForward, miopen_handle, attnDropoutDesc, 
                nullptr, midtensor_desc, mid_tensor_s, midtensor_desc, s_buffer, 
                reservespace, attn_reserve_size);
        
        // S(batch*h, seqlen_L, seqlen_S) x V(batch*h, seq_length_S, embed_dim/h)
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_none, 
                rocblas_operation::rocblas_operation_none, 
                head_dim, seq_length_L, seq_length_S, alpha,
                d_AV, data_types[io::v], head_dim, 
                d_ASb, data_types[io::q], seq_length_S, beta,
                d_AO, data_types[io::o], head_dim, 
                d_AO, data_types[io::o], head_dim, batch_size*num_heads,
                compute_type, rocblas_gemm_algo_standard, 0);
        
        void* o_proj_tran = (char*)workspace + o_proj_t_offset;
        void* o_proj = weight_enabled[6] ? (char*)workspace + o_proj_offset : out;
        // transpose o_proj_tran (and reshape to) -> {seq_length_L, batch_size, embed_dim/h*h}
        size_t dims_o_proj_tran[3] = {batch_size*num_heads, seq_length_L, embed_dim/num_heads};
        hip_custom::transpose(dtype_bytesize[io::v], o_proj_tran, o_proj, dims_o_proj_tran, 3, 0, 1);

        // reproject matmul for o_proj to get o, (seq_length_L*batch_size, embed_dim) x (embed_dim*h/h, embed_dim)^T
        if(weight_enabled[6])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_transpose, 
                    rocblas_operation::rocblas_operation_none, 
                    embed_dim, seq_length_S*batch_size, embed_dim, alpha, 
                    weightbias[6], weight_type, embed_dim, 
                    o_proj, data_types[io::o], embed_dim, beta, 
                    out, data_types[io::o], embed_dim, 
                    out, data_types[io::o], embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);
        
        if(weight_enabled[7])
            MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, miopen_handle, miopenTensorOpAdd, 
                    &f32_alpha, qo_desc, out, &f32_alpha, bias_desc, weightbias[7], 
                    &f32_beta, qo_desc, out);

        // post dropout
        size_t post_reserve_size;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetReserveSpaceSize, qo_desc, &post_reserve_size);
        void* post_reservespace = (char*)reservespace + attn_reserve_size;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutForward, miopen_handle, postDropoutDesc, nullptr, 
                qo_desc, out, qo_desc, out, post_reservespace, post_reserve_size);

        if(residuals)
            MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, miopen_handle, miopenTensorOpAdd, 
                    &f32_alpha, qo_desc, out, &f32_alpha, qo_desc, residuals, 
                    &f32_beta, qo_desc, out);
    }
};

struct miopen_multi_head_attn_bwd_data_impl_t : public miopen_multi_head_attn_impl_base_t {
public:
    virtual ~miopen_multi_head_attn_bwd_data_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, midtensor_desc);                                                                                                                          
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, qo_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, kv_desc);
    }

    status_t init(impl::engine_t *engine, multi_head_attn_pd_t *pd) override {
        CHECK(cudnn_multi_head_attn_impl_base_t::init(engine, pd));
        // configure parameters again
        CHECK(configure_parameters(pd));
        CHECK(check_proj_weight(pd));
        // get some parameters from forward primitive desc
        CHECK(get_fwd_parameters(pd));
        CHECK(create_miopen_descs(pd));
        return status::success;
    }

    // mid tensor desc and droupout desc
    status_t create_miopen_descs(const multi_head_attn_pd_t *pd) {
        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &midtensor_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, midtensor_desc, midtensor_dtype,
                1, seq_length_S, batch_size*num_heads*seq_length_L, 1);

        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &qo_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, qo_desc, qo_dtype,
                1, seq_length_L, batch_size, embed_dim);
        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &kv_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, kv_desc, kv_dtype,
                1, seq_length_S, batch_size, embed_dim);

        return status::success;
    }

    status_t get_fwd_parameters(multi_head_attn_pd_t *pd) {
        attnDropoutDesc = (miopenDropoutDescriptor_t)(pd->get_dropDesc(0));
        postDropoutDesc = (miopenDropoutDescriptor_t)(pd->get_dropDesc(1));

        reserveSpaceSizeInBytes = pd->get_reserveSpaceSizeInBytes();
        workSpaceSizeInBytes = pd->get_workSpaceSizeInBytes();
        workspace = pd->get_workspace();
        reservespace = pd->get_reservespace();

        size_t* offsets = pd->get_offsets();
        // forward
        q_proj_offset = offsets[0];
        q_proj_t_offset = offsets[1];
        k_proj_offset = offsets[2];
        k_proj_t_offset = offsets[3];
        v_proj_offset = offsets[4];
        v_proj_t_offset = offsets[5];
        mid_s_offset = offsets[6];
        s_buffer_offset = offsets[7];
        o_proj_offset = offsets[8];
        o_proj_t_offset = offsets[9];
        dAQ_offset = offsets[10];
        dAK_offset = offsets[11];
        dAS_offset = offsets[12];
        dASb_offset = offsets[13];
        dAV_offset = offsets[14];
        dAO_offset = offsets[15];
        // backward
        dout_offset = offsets[16];
        doproj_offset = offsets[17];
        doproj_t_offset = offsets[18];
        ds_offset = offsets[19];
        dvproj_offset = offsets[20];
        dvproj_t_offset = offsets[21];
        dqproj_offset = offsets[22];
        dqproj_t_offset = offsets[23];
        dkproj_offset = offsets[24];
        dkproj_t_offset = offsets[25];
        ddAQ_offset = offsets[26];
        ddAK_offset = offsets[27];
        ddAS_offset = offsets[28];
        ddAV_offset = offsets[29];
        ddAO_offset = offsets[30];

        return status::success;
    }

    void set_batch_matrices_bw(void* workspace, void*& d_dAQ, void*& d_dAK, 
            void*& d_dAS, void*& d_dAV, void*& d_dAO) {
        
        void **h_dAQ, **h_dAK, **h_dAS, **h_dAV, **h_dAO;
        int batch_count = batch_size*num_heads;
        h_dAQ = (void**)malloc(sizeof(void*) * batch_count);
        h_dAK = (void**)malloc(sizeof(void*) * batch_count);
        h_dAS = (void**)malloc(sizeof(void*) * batch_count);
        h_dAV = (void**)malloc(sizeof(void*) * batch_count);
        h_dAO = (void**)malloc(sizeof(void*) * batch_count);
        for(int i=0; i<batch_count; i++) {
            h_dAQ[i] = (char*)workspace + dqproj_t_offset + sizeof(void*)*i;
            h_dAK[i] = (char*)workspace + dkproj_t_offset + sizeof(void*)*i;
            h_dAS[i] = (char*)workspace + ds_offset + sizeof(void*)*i;
            h_dAV[i] = (char*)workspace + dvproj_t_offset + sizeof(void*)*i;
            h_dAO[i] = (char*)workspace + doproj_t_offset + sizeof(void*)*i;
        }

        d_dAQ = (char*)workspace + ddAQ_offset;
        d_dAK = (char*)workspace + ddAK_offset;
        d_dAS = (char*)workspace + ddAS_offset;
        d_dAV = (char*)workspace + ddAV_offset;
        d_dAO = (char*)workspace + ddAO_offset;

        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAQ, (HIPdeviceptr)h_dAQ, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAK, (HIPdeviceptr)h_dAK, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAS, (HIPdeviceptr)h_dAS, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAV, (HIPdeviceptr)h_dAV, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAO, (HIPdeviceptr)h_dAO, 
                batch_count*sizeof(void*), hipMemcpyDefault);

        free(h_dAQ);
        free(h_dAK);
        free(h_dAS);
        free(h_dAV);
        free(h_dAO);
    }

    void get_batch_matrices(void*& d_AQ, void*& d_AK, void*& d_AS, void*& d_AV, void*& d_AO) {
        d_AQ = (char*)workspace + dAQ_offset;
        d_AK = (char*)workspace + dAK_offset;
        d_AS = (char*)workspace + dAS_offset;
        d_AV = (char*)workspace + dAV_offset;
        d_AO = (char*)workspace + dAO_offset;
    }

    // We assume the corresponding forward is already finished before the backward call.
    void execute(rocblas_handle rocblas_handle, miopenHandle_t miopen_handle, 
            const std::vector<void *> &args) const override {
        auto dout = args[0], dqueries = args[1], dkeys = args[2], dvalues = args[3];
        
        void* weightbias[8];
        for(int i=4; i<4+8; i++)
            weightbias[i-4] = args[i];

        const void *alpha = get_gemm_alpha();
        const void *beta = get_gemm_beta();

        // backward post dropout
        void* dout_buffer =  dout_offset + (char*)workspace;
        void* post_reservespace = (char*)reservespace + attn_reserve_size;
        size_t post_reserve_size;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetReserveSpaceSize, qo_desc, &post_reserve_size);
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutBackward, miopen_handle, postDropoutDesc, 
                nullptr, qo_desc, dout, qo_desc, dout_buffer, post_reservespace, 
                post_reserve_size);
        
        void* do_proj = weight_enabled[6] ? doproj_offset + (char*)workspace : dout_buffer;
        void* do_proj_tran = (char*)workspace + doproj_t_offset;

        // backward o weight, (seq_length_L, batch_size, embed_dim) x (embed_dim, embed_dim)
        if(weight_enabled[6])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_none, 
                    embed_dim, seq_length_L*batch_size, embed_dim, alpha, 
                    weightbias[6], weight_type, embed_dim, 
                    dout_buffer, data_types[io::o], embed_dim, beta, 
                    do_proj, data_types[io::o], embed_dim, 
                    do_proj, data_types[io::o], embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

        // transpose do_proj (and reshape to) -> {batch_size*h, seq_length_L, embed_dim/h}
        size_t dims_do_proj_tran[3] = {seq_length_L, batch_size*num_heads, embed_dim/num_heads};
        hip_custom::transpose(dtype_bytesize[io::v], do_proj, do_proj_tran, dims_o_proj_tran, 3, 0, 1);

        int head_dim = embed_dim / num_heads;

        void *d_AQ, *d_AK, *d_AS, *d_AV, *d_AO;
        get_batch_matrices(d_AQ, d_AK, d_AS, d_AV, d_AO);

        void *d_dAQ, *d_dAK, *d_dAS, *d_dAV, *d_dAO;
        set_batch_matrices_bw(workspace, d_dAQ, d_dAK, d_dAS, d_dAV, d_dAO);    // set for backward

        // dO(batch*h, seqlen_L, embed_dim/h) x V(batch*h, seq_length_S, embed_dim/h)^T
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_transpose, 
                rocblas_operation::rocblas_operation_none, 
                seq_length_S, seq_length_L, head_dim, alpha,
                d_AV, data_types[io::v], head_dim, 
                d_dAO, data_types[io::o], head_dim, beta,
                d_dAS, data_types[io::o], seq_length_S, 
                d_dAS, data_types[io::o], seq_length_S, batch_size*num_heads,
                compute_type, rocblas_gemm_algo_standard, 0);

        // S(batch*h, seqlen_L, seq_length_S)^T x dO(batch*h, seqlen_L, embed_dim/h)
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_none, 
                rocblas_operation::rocblas_operation_transpose, 
                head_dim, seq_length_S, seq_length_L, alpha,
                d_dAO, data_types[io::o], head_dim, 
                d_AS, data_types[io::o], seq_length_S, beta,
                d_dAV, data_types[io::v], head_dim, 
                d_dAV, data_types[io::v], head_dim, batch_size*num_heads,
                compute_type, rocblas_gemm_algo_standard, 0);

        void* ds_buffer = ds_offset + (char*)workspace;
        void* attn_reservespace = (char*)reservespace;
        size_t attn_reserve_size;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetReserveSpaceSize, midtensor_desc, &attn_reserve_size);
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutBackward, miopen_handle, attnDropoutDesc, 
                nullptr, midtensor_desc, ds_buffer, midtensor_desc, ds_buffer, 
                attn_reservespace, attn_reserve_size);

        void* mid_tensor_s = (char*)workspace + mid_s_offset;
        MIOPEN_EXECUTE_FUNC_V(miopenSoftmaxBackward, miopen_handle, alpha, midtensor_desc, 
            mid_tensor_s, beta, midtensor_desc, ds_buffer, midtensor_desc, ds_buffer);
        
        void* dv_proj = weight_enabled[4] ? dvproj_offset + (char*)workspace : dvalues;
        void* dv_proj_tran = (char*)workspace + doproj_t_offset;
        // transpose dv_proj (and reshape to) -> {seq_length_S, batch_size, embed_dim/h*h}
        size_t dims_dv_proj_tran[3] = {batch_size*num_heads, seq_length_S, head_dim};
        hip_custom::transpose(dtype_bytesize[io::v], dv_proj_tran, dv_proj, dims_dv_proj_tran, 3, 0, 1);

        // backward v weight, (seq_length_S, batch_size, embed_dim) x (embed_dim, vdim)
        if(weight_enabled[4])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_none, 
                    vdim, seq_length_S*batch_size, embed_dim, alpha, 
                    weightbias[4], weight_type, vdim, 
                    dv_proj, data_types[io::v], embed_dim, beta, 
                    dvalues, data_types[io::v], vdim, 
                    dvalues, data_types[io::v], vdim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

        // dS(batch*h, seqlen_L, seqlen_S)^T x Q(batch*h, seqlen_L, embed_dim/h)
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_none, 
                rocblas_operation::rocblas_operation_transpose, 
                head_dim, seq_length_S, seq_length_L, alpha,
                d_AQ, data_types[io::q], head_dim, 
                d_dAS, data_types[io::o], seq_length_S, beta,
                d_dAK, data_types[io::k], head_dim, 
                d_dAK, data_types[io::k], head_dim, batch_size*num_heads, 
                compute_type, rocblas_gemm_algo_standard, 0);

        void* dk_proj = weight_enabled[2] ? dkproj_offset + (char*)workspace : dkeys;
        void* dk_proj_tran = (char*)workspace + dkproj_t_offset;
        // transpose dk_proj (and reshape to) -> {seq_length_S, batch_size, embed_dim/h*h}
        size_t dims_dk_proj_tran[3] = {batch_size*num_heads, seq_length_S, head_dim};
        hip_custom::transpose(dtype_bytesize[io::k], dk_proj_tran, dk_proj, dims_dk_proj_tran, 3, 0, 1);

        // backward k weight, (seq_length_S, batch_size, embed_dim) x (embed_dim, kdim)
        if(weight_enabled[2])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_none, 
                    kdim, seq_length_S*batch_size, embed_dim, alpha, 
                    weightbias[2], weight_type, kdim, 
                    dk_proj, data_types[io::k], embed_dim, beta, 
                    dkeys, data_types[io::k], kdim, 
                    dkeys, data_types[io::k], kdim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

        // dS(batch*h, seqlen_L, seqlen_S) x K(batch*h, seqlen_S, embed_dim/h)
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_none, 
                rocblas_operation::rocblas_operation_none, 
                head_dim, seq_length_L, seq_length_S, alpha, 
                d_AK, data_types[io::k], head_dim, 
                d_dAS, data_types[io::o], seq_length_S, beta, 
                d_dAQ, data_types[io::q], head_dim, 
                d_dAQ, data_types[io::q], head_dim, batch_size*num_heads, 
                compute_type, rocblas_gemm_algo_standard, 0);
        
        void* dq_proj = weight_enabled[0] ? dqproj_offset + (char*)workspace : dqueries;
        void* dq_proj_tran = (char*)workspace + dqproj_t_offset;
        // transpose dq_proj (and reshape to) -> {seq_length_L, batch_size, embed_dim/h*h}
        size_t dims_dq_proj_tran[3] = {batch_size*num_heads, seq_length_L, head_dim};
        hip_custom::transpose(dtype_bytesize[io::q], dq_proj_tran, dq_proj, dims_dq_proj_tran, 3, 0, 1);

        // backward q weight, (seq_length_L, batch_size, embed_dim) x (embed_dim, embed_dim)
        if(weight_enabled[0])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_none, 
                    embed_dim, seq_length_S*batch_size, embed_dim, alpha, 
                    weightbias[0], weight_type, embed_dim, 
                    dq_proj, data_types[io::q], embed_dim, beta, 
                    dqueries, data_types[io::q], embed_dim, 
                    dqueries, data_types[io::q], embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);
    }
};

struct miopen_multi_head_attn_bwd_weights_impl_t : public miopen_multi_head_attn_impl_base_t {
protected:
    miopenReduceTensorDescriptor_t reduceDesc;
    size_t reduce_workspace_size = 0;
public:
    virtual ~miopen_multi_head_attn_bwd_weights_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, midtensor_desc);                                                                                                                          
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, qo_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, kv_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryTensorDescriptor, bias_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestoryReduceTensorDescriptor, reduceDesc);
    }
    
    status_t init(impl::engine_t *engine, multi_head_attn_pd_t *pd) override {
        CHECK(cudnn_multi_head_attn_impl_base_t::init(engine, pd));
        // configure parameters again
        CHECK(configure_parameters(pd));
        CHECK(check_proj_weight(pd));
        // get some parameters from forward primitive desc
        CHECK(get_fwd_parameters(pd));
        CHECK(create_miopen_descs(pd));
        return status::success;
    }

    status_t get_fwd_parameters(multi_head_attn_pd_t *pd) {
        attnDropoutDesc = (miopenDropoutDescriptor_t)(pd->get_dropDesc(0));
        postDropoutDesc = (miopenDropoutDescriptor_t)(pd->get_dropDesc(1));

        reserveSpaceSizeInBytes = pd->get_reserveSpaceSizeInBytes();
        workSpaceSizeInBytes = pd->get_workSpaceSizeInBytes();
        workspace = pd->get_workspace();
        reservespace = pd->get_reservespace();

        size_t* offsets = pd->get_offsets();
        // forward
        q_proj_offset = offsets[0];
        q_proj_t_offset = offsets[1];
        k_proj_offset = offsets[2];
        k_proj_t_offset = offsets[3];
        v_proj_offset = offsets[4];
        v_proj_t_offset = offsets[5];
        mid_s_offset = offsets[6];
        s_buffer_offset = offsets[7];
        o_proj_offset = offsets[8];
        o_proj_t_offset = offsets[9];
        dAQ_offset = offsets[10];
        dAK_offset = offsets[11];
        dAS_offset = offsets[12];
        dASb_offset = offsets[13];
        dAV_offset = offsets[14];
        dAO_offset = offsets[15];
        // backward
        dout_offset = offsets[16];
        doproj_offset = offsets[17];
        doproj_t_offset = offsets[18];
        ds_offset = offsets[19];
        dvproj_offset = offsets[20];
        dvproj_t_offset = offsets[21];
        dqproj_offset = offsets[22];
        dqproj_t_offset = offsets[23];
        dkproj_offset = offsets[24];
        dkproj_t_offset = offsets[25];
        ddAQ_offset = offsets[26];
        ddAK_offset = offsets[27];
        ddAS_offset = offsets[28];
        ddAV_offset = offsets[29];
        ddAO_offset = offsets[30];

        return status::success;
    }

    // mid tensor desc and droupout desc
    status_t create_miopen_descs(impl::engine_t *engine, const multi_head_attn_pd_t *pd) {
        auto &sycl_engine = *utils::downcast<amd::engine_t *>(engine);
        impl::stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<amd::stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();
        
        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &midtensor_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, midtensor_desc, midtensor_dtype,
                1, seq_length_S, batch_size*num_heads*seq_length_L, 1);

        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &qo_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, qo_desc, qo_dtype,
                1, seq_length_L, batch_size, embed_dim);
        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &kv_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, kv_desc, kv_dtype,
                1, seq_length_S, batch_size, embed_dim);
        MIOPEN_EXECUTE_FUNC_V(miopenCreateTensorDescriptor, &bias_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenSet4dTensorDescriptor, bias_desc, qo_dtype,
                1, 1, 1, embed_dim);

        MIOPEN_EXECUTE_FUNC_V(miopenCreateReduceTensorDescriptor, &reduceDesc);
        MIOPEN_EXECUTE_FUNC_V(miopenSetReduceTensorDescriptor, reduceDesc, MIOPEN_REDUCE_TENSOR_ADD, 
                qo_dtype, MIOPEN_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_TENSOR_NO_INDICES,
                MIOPEN_32BIT_INDICES);

        size_t size_temp = 0;
        // bias[seq_len_L or seq_len_S, batch, embed_dim, 1] -> bias[1, 1, embed_dim]
        MIOPEN_EXECUTE_FUNC_V(miopenGetReductionWorkspaceSize, handle, reduceDesc, qo_desc
                bias_desc, &reduce_workspace_size);
        MIOPEN_EXECUTE_FUNC_V(miopenGetReductionWorkspaceSize, handle, reduceDesc, kv_desc
                bias_desc, &size_temp);

        if(size_temp > reduce_workspace_size)
            reduce_workspace_size = size_temp;

        if(reduce_workspace_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_attn_reduce, reduce_workspace_size,
                    size_t(1));
        
        return status::success;
    }

    void set_batch_matrices_bw(void* workspace, void*& d_dAQ, void*& d_dAK, 
            void*& d_dAS, void*& d_dAV, void*& d_dAO) {
        
        void **h_dAQ, **h_dAK, **h_dAS, **h_dAV, **h_dAO;
        int batch_count = batch_size*num_heads;
        h_dAQ = (void**)malloc(sizeof(void*) * batch_count);
        h_dAK = (void**)malloc(sizeof(void*) * batch_count);
        h_dAS = (void**)malloc(sizeof(void*) * batch_count);
        h_dAV = (void**)malloc(sizeof(void*) * batch_count);
        h_dAO = (void**)malloc(sizeof(void*) * batch_count);
        for(int i=0; i<batch_count; i++) {
            h_dAQ[i] = (char*)workspace + dqproj_t_offset + sizeof(void*)*i;
            h_dAK[i] = (char*)workspace + dkproj_t_offset + sizeof(void*)*i;
            h_dAS[i] = (char*)workspace + ds_offset + sizeof(void*)*i;
            h_dAV[i] = (char*)workspace + dvproj_t_offset + sizeof(void*)*i;
            h_dAO[i] = (char*)workspace + doproj_t_offset + sizeof(void*)*i;
        }

        d_dAQ = (char*)workspace + ddAQ_offset;
        d_dAK = (char*)workspace + ddAK_offset;
        d_dAS = (char*)workspace + ddAS_offset;
        d_dAV = (char*)workspace + ddAV_offset;
        d_dAO = (char*)workspace + ddAO_offset;

        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAQ, (HIPdeviceptr)h_dAQ, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAK, (HIPdeviceptr)h_dAK, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAS, (HIPdeviceptr)h_dAS, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAV, (HIPdeviceptr)h_dAV, 
                batch_count*sizeof(void*), hipMemcpyDefault);
        HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&d_dAO, (HIPdeviceptr)h_dAO, 
                batch_count*sizeof(void*), hipMemcpyDefault);

        free(h_dAQ);
        free(h_dAK);
        free(h_dAS);
        free(h_dAV);
        free(h_dAO);
    }

    // We assume the corresponding forward is already finished before the backward call.
    void execute(rocblas_handle rocblas_handle, miopenHandle_t miopen_handle, 
            const std::vector<void *> &args) const override {
        auto queries = args[0], keys = args[1], values = args[2], 
                dout = args[3];
        
        void* weightbias[8];
        for(int i=4; i<4+8; i++)
            weightbias[i-4] = args[i];

        void* dweightbias[8];
        for(int i=12; i<12+8; i++)
            dweightbias[i-12] = args[i];

        void* reduce_workspace = args[20];

        const void *alpha = get_gemm_alpha();
        const void *beta = get_gemm_beta();

        // backward post dropout
        void* dout_buffer =  dout_offset + (char*)workspace;
        void* post_reservespace = (char*)reservespace + attn_reserve_size;
        size_t post_reserve_size;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetReserveSpaceSize, qo_desc, &post_reserve_size);
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutBackward, miopen_handle, postDropoutDesc, 
                nullptr, qo_desc, dout, qo_desc, dout_buffer, post_reservespace, 
                post_reserve_size);
        
        // bias have no influence on gradient. reduce to get dbias.
        if(weight_enabled[7])
            MIOPEN_EXECUTE_FUNC_V(miopenReduceTensor, miopen_handle, reduceDesc, nullptr, 0, 
                    reduce_workspace, reduce_workspace_size, alpha, qo_desc, dout_buffer, beta, 
                    bias_desc, dweightbias[7]);

        void* do_proj = weight_enabled[6] ? doproj_offset + (char*)workspace : dout_buffer;
        void* do_proj_tran = (char*)workspace + doproj_t_offset;

        if(weight_enabled[6]) {
            // backward o weight, (seq_length_L, batch_size, embed_dim) x (embed_dim, embed_dim)
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_none, 
                    embed_dim, seq_length_L*batch_size, embed_dim, alpha, 
                    weightbias[6], weight_type, embed_dim, 
                    dout_buffer, data_types[io::o], embed_dim, beta, 
                    do_proj, data_types[io::o], embed_dim, 
                    do_proj, data_types[io::o], embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

            void* o_proj = (char*)workspace + o_proj_offset;
            // backward o, do(seq_length_L, batch_size, embed_dim)^T x o(seq_length_L, batch_size, embed_dim)
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_transpose, 
                    embed_dim, embed_dim, seq_length_L*batch_size, alpha, 
                    o_proj, data_types[io::o], embed_dim, 
                    dout_buffer, data_types[io::o], embed_dim, beta, 
                    dweightbias[6], weight_type, embed_dim, 
                    dweightbias[6], weight_type, embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);
        }

        // transpose do_proj (and reshape to) -> {batch_size*h, seq_length_L, embed_dim/h}
        size_t dims_do_proj_tran[3] = {seq_length_L, batch_size*num_heads, embed_dim/num_heads};
        hip_custom::transpose(dtype_bytesize[io::v], do_proj, do_proj_tran, dims_o_proj_tran, 3, 0, 1);

        int head_dim = embed_dim / num_heads;

        void *d_AQ, *d_AK, *d_AS, *d_AV, *d_AO;
        get_batch_matrices(d_AQ, d_AK, d_AS, d_AV, d_AO);

        void *d_dAQ, *d_dAK, *d_dAS, *d_dAV, *d_dAO;
        set_batch_matrices_bw(workspace, d_dAQ, d_dAK, d_dAS, d_dAV, d_dAO);    // set for backward

        // dO(batch*h, seqlen_L, embed_dim/h) x V(batch*h, seq_length_S, embed_dim/h)^T
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_transpose, 
                rocblas_operation::rocblas_operation_none, 
                seq_length_S, seq_length_L, head_dim, alpha,
                d_AV, data_types[io::v], head_dim, 
                d_dAO, data_types[io::o], head_dim, beta,
                d_dAS, data_types[io::o], seq_length_S, 
                d_dAS, data_types[io::o], seq_length_S, batch_size*num_heads,
                compute_type, rocblas_gemm_algo_standard, 0);


        // S(batch*h, seqlen_L, seq_length_S)^T x dO(batch*h, seqlen_L, embed_dim/h)
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_none, 
                rocblas_operation::rocblas_operation_transpose, 
                head_dim, seq_length_S, seq_length_L, alpha,
                d_dAO, data_types[io::o], head_dim, 
                d_AS, data_types[io::o], seq_length_S, beta,
                d_dAV, data_types[io::v], head_dim, 
                d_dAV, data_types[io::v], head_dim, batch_size*num_heads,
                compute_type, rocblas_gemm_algo_standard, 0);

        void* ds_buffer = ds_offset + (char*)workspace;
        void* attn_reservespace = (char*)reservespace;
        size_t attn_reserve_size;
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutGetReserveSpaceSize, midtensor_desc, &attn_reserve_size);
        MIOPEN_EXECUTE_FUNC_V(miopenDropoutBackward, miopen_handle, attnDropoutDesc, 
                nullptr, midtensor_desc, ds_buffer, midtensor_desc, ds_buffer, 
                attn_reservespace, attn_reserve_size);

        void* mid_tensor_s = (char*)workspace + mid_s_offset;
        MIOPEN_EXECUTE_FUNC_V(miopenSoftmaxBackward, miopen_handle, alpha, midtensor_desc, 
            mid_tensor_s, beta, midtensor_desc, ds_buffer, midtensor_desc, ds_buffer);

        void* dv_proj = weight_enabled[4] ? dvproj_offset + (char*)workspace : dvalues;
        void* dv_proj_tran = (char*)workspace + doproj_t_offset;
        // transpose dv_proj (and reshape to) -> {seq_length_S, batch_size, embed_dim/h*h}
        size_t dims_dv_proj_tran[3] = {batch_size*num_heads, seq_length_S, head_dim};
        hip_custom::transpose(dtype_bytesize[io::v], dv_proj_tran, dv_proj, dims_dv_proj_tran, 3, 0, 1);

        if(weight_enabled[5])
            MIOPEN_EXECUTE_FUNC_V(miopenReduceTensor, miopen_handle, reduceDesc, nullptr, 0, 
                    reduce_workspace, reduce_workspace_size, alpha, kv_desc, dv_proj, beta, 
                    bias_desc, dweightbias[5]);

        // backward v, dv(seq_length_S, batch_size, embed_dim)^T x V(seq_length_S, batch_size, vdim)
        if(weight_enabled[4])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_transpose, 
                    vdim, embed_dim, seq_length_S*batch_size, alpha, 
                    values, data_types[io::v], vdim, 
                    dv_proj, data_types[io::v], embed_dim, beta, 
                    dweightbias[4], weight_type, vdim, 
                    dweightbias[4], weight_type, vdim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

        // dS(batch*h, seqlen_L, seqlen_S)^T x Q(batch*h, seqlen_L, embed_dim/h)
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_none, 
                rocblas_operation::rocblas_operation_transpose, 
                head_dim, seq_length_S, seq_length_L, alpha,
                d_AQ, data_types[io::q], head_dim, 
                d_dAS, data_types[io::o], seq_length_S, beta,
                d_dAK, data_types[io::k], head_dim, 
                d_dAK, data_types[io::k], head_dim, batch_size*num_heads, 
                compute_type, rocblas_gemm_algo_standard, 0);

        void* dk_proj = weight_enabled[2] ? dkproj_offset + (char*)workspace : dkeys;
        void* dk_proj_tran = (char*)workspace + dkproj_t_offset;
        // transpose dk_proj (and reshape to) -> {seq_length_S, batch_size, embed_dim/h*h}
        size_t dims_dk_proj_tran[3] = {batch_size*num_heads, seq_length_S, head_dim};
        hip_custom::transpose(dtype_bytesize[io::k], dk_proj_tran, dk_proj, dims_dk_proj_tran, 3, 0, 1);

        if(weight_enabled[3])
            MIOPEN_EXECUTE_FUNC_V(miopenReduceTensor, miopen_handle, reduceDesc, nullptr, 0, 
                    reduce_workspace, reduce_workspace_size, alpha, kv_desc, dk_proj, beta, 
                    bias_desc, dweightbias[3]);

        // backward k, dk(seq_length_S, batch_size, embed_dim)^T x K(seq_length_S, batch_size, kdim)
        if(weight_enabled[2])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_transpose, 
                    kdim, embed_dim, seq_length_S*batch_size, alpha, 
                    keys, data_types[io::k], kdim, 
                    dk_proj, data_types[io::k], embed_dim, beta, 
                    dweightbias[2], weight_type, kdim, 
                    dweightbias[2], weight_type, kdim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);

        // dS(batch*h, seqlen_L, seqlen_S) x K(batch*h, seqlen_S, embed_dim/h)
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_batched_ex, rocblas_handle, 
                rocblas_operation::rocblas_operation_none, 
                rocblas_operation::rocblas_operation_none, 
                head_dim, seq_length_L, seq_length_S, alpha, 
                d_AK, data_types[io::k], head_dim, 
                d_dAS, data_types[io::o], seq_length_S, beta, 
                d_dAQ, data_types[io::q], head_dim, 
                d_dAQ, data_types[io::q], head_dim, batch_size*num_heads, 
                compute_type, rocblas_gemm_algo_standard, 0);
        
        void* dq_proj = weight_enabled[0] ? dqproj_offset + (char*)workspace : dqueries;
        void* dq_proj_tran = (char*)workspace + dqproj_t_offset;
        // transpose dq_proj (and reshape to) -> {seq_length_L, batch_size, embed_dim/h*h}
        size_t dims_dq_proj_tran[3] = {batch_size*num_heads, seq_length_L, head_dim};
        hip_custom::transpose(dtype_bytesize[io::q], dq_proj_tran, dq_proj, dims_dq_proj_tran, 3, 0, 1);

        if(weight_enabled[1])
            MIOPEN_EXECUTE_FUNC_V(miopenReduceTensor, miopen_handle, reduceDesc, nullptr, 0, 
                    reduce_workspace, reduce_workspace_size, alpha, qo_desc, dq_proj, beta, 
                    bias_desc, dweightbias[1]);

        // backward q weight, dq(seq_length_L, batch_size, embed_dim)^T x Q(seq_length_L, batch_size, embed_dim)
        if(weight_enabled[0])
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, 
                    rocblas_operation::rocblas_operation_none, 
                    rocblas_operation::rocblas_operation_transpose, 
                    embed_dim, embed_dim, seq_length_S*batch_size, alpha, 
                    queries, weight_type, embed_dim, 
                    dq_proj, data_types[io::q], embed_dim, beta, 
                    dweightbias[0], weight_type, embed_dim, 
                    dweightbias[0], weight_type, embed_dim, 
                    compute_type, rocblas_gemm_algo_standard, -1, 0);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
