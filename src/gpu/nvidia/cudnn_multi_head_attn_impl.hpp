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

#ifndef GPU_NVIDIA_CUDNN_MULTI_HEAD_ATTN_IMPL_HPP
#define GPU_NVIDIA_CUDNN_MULTI_HEAD_ATTN_IMPL_HPP

#include "cudnn.h"

#include "common/c_types_map.hpp"
#include "common/multi_head_attn_pd.hpp"
#include "common/utils.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

// `CUDNN_FMA_MATH` is available starting from cuDNN v8.
// The behavior is consistent with `CUDNN_DEFAULT_MATH` for v7.
#if defined(CUDNN_MAJOR) && (CUDNN_MAJOR < 8)
#define CUDNN_FMA_MATH CUDNN_DEFAULT_MATH
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_multi_head_attn_impl_base_t {
protected:
    cudnnAttnDescriptor_t attnDesc;
	cudnnDropoutDescriptor_t attnDropoutDesc;
	cudnnDropoutDescriptor_t postDropoutDesc;

	unsigned attnMode;
	// Compute precision.
    cudnnDataType_t computePrec = CUDNN_DATA_FLOAT;
	// NVIDIA Tensor Core settings.
    cudnnMathType_t mathType = CUDNN_DEFAULT_MATH;
	
	enum io { q = 0, k, v, o, NUM_IO };

    cudnnSeqDataDescriptor_t SeqDataDescs[NUM_IO];
    cudnnTensorDescriptor_t weightbias_tdesc[8]; // 4 weight and 4 bias.
    bool proj_disabled[8];   // whether the projection is disabled.

	memory_desc_t dnnl_descs[NUM_IO];
	// memory_desc_t weight_desc;

    size_t wSize[8];
    mutable void* wAddr[8];

	cudnnDataType_t data_types[NUM_IO];
	cudnnDataType_t weight_type;

	size_t reserveSpaceSizeInBytes;
	size_t workSpaceSizeInBytes;
	size_t weightSizeInBytes;
	size_t Dropout_stateSize;

	int dimA[NUM_IO][CUDNN_SEQDATA_DIM_COUNT];  // CUDNN_SEQDATA_DIM_COUNT equals to 4.
    // indicate that the layout of qkvo tensor(seqdata). {0, 1, 2, 3} means plain layout.
	int axes[NUM_IO][CUDNN_SEQDATA_DIM_COUNT];

    // sla_size equals to dimA[CUDNN_SEQDATA_BATCH_DIM] * dimA[CUDNN_SEQDATA_BEAM_DIM]
    size_t seqlentharray_sizes[NUM_IO];
    std::vector<int> seqlenthArray[NUM_IO];

    int maxseqlength_qo;
    int maxseqlength_kv;
    int maxbatchsize;
    int maxbeamsize;

	int nHeads;
	double smScaler;

    int size_QKVO[NUM_IO];
    // If projsize is set to 0, it means the coresponding project is disabled.
    int projsize_QKVO[NUM_IO];

    size_t weightbias_totalsize = 0;

    // these 3 pointers shall not be freed during the lifecycle of this primitive.
    const int* p_currIdx;
    const int* loWinIdx;
    const int* hiWinIdx;

    float attndropout;
    float postattndropout;
    unsigned long long dropoutseed;
    unsigned long long postdropoutseed;
public:
    virtual ~cudnn_multi_head_attn_impl_base_t() {
		CUDNN_EXECUTE_FUNC_V(cudnnDestroyAttnDescriptor, attnDesc);
		CUDNN_EXECUTE_FUNC_V(cudnnDestroyDropoutDescriptor, attnDropoutDesc);
		CUDNN_EXECUTE_FUNC_V(cudnnDestroyDropoutDescriptor, postDropoutDesc);
        for (size_t i = 0; i < io::NUM_IO; i++) {
			CUDNN_EXECUTE_FUNC_V(cudnnDestroySeqDataDescriptor,  SeqDataDescs[i]);
        }
        for(size_t i = 0; i<8; i++) {
            if(!proj_disabled[i])
                CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, weightbias_tdesc[i]);
        }
    }

    bool with_reserveSpace() const { return reserveSpaceSizeInBytes > 0; }

	// MHA always need scratchpad(workSpace).
	bool with_scratchpad() const { return true; } 

    virtual status_t init(impl::engine_t *engine, multi_head_attn_pd_t *pd) {
        CHECK(check_proj_weight(pd));
        CHECK(configure_parameters(pd));
        CHECK(create_cudnn_descs(engine, pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    virtual status_t init_zero_dims(multi_head_attn_pd_t *pd) {
        return status::success;
    }

	status_t convert_dnnl_desc(io idx, const memory_desc_t* md, const int* axes_) {
		dnnl_descs[idx] = *md;
		if(dnnl_descs[idx].ndims != CUDNN_SEQDATA_DIM_COUNT) 
			return status::invalid_arguments;
		for(int i=0; i<CUDNN_SEQDATA_DIM_COUNT; i++){
			dimA[idx][i] = dnnl_descs[idx].dims[i];
			axes[idx][i] = axes_[i];
		}
		return status::success;
	}

    void get_seqlenth_utils(io idx, int* seqlenarray, int* maxseqlengh = nullptr){
        int arraylenth = seqlentharray_sizes[idx];
        seqlenthArray[idx].resize(arraylenth);
        if(maxseqlengh == nullptr){
            for(int i=0; i<arraylenth; i++)
                seqlenthArray[idx][i] = seqlenarray[i];
        }
        else {
            *maxseqlengh = 0;
            for(int i=0; i<arraylenth; i++){
                if(seqlenarray[i] > *maxseqlengh)
                    *maxseqlengh = seqlenarray[i];
                seqlenthArray[idx][i] = seqlenarray[i];
            }
        }
    }

    status_t check_proj_weight(const multi_head_attn_pd_t* pd){
        for(int i=0; i<8; i++){
            auto wb_md = *pd->weight_md(i);
            // check weight or bias data type.
            cudnnDataType_t wb_dt;
            CHECK(convert_data_type(&wb_md, &wb_dt));
            if(i == 0)
                weight_type = wb_dt;
            else
                if(weight_type != wb_dt)
                    return status::invalid_arguments;

            if(wb_md.ndims != 3)
                return status::invalid_arguments;
            
            // create and set weight or bias tensor descriptor.
            // weight[nHeads, projected size, original size], bias[nHeads, projected size, 1]
            int dimA_wb[3]; // all weight and bias md.ndims equals 3.
            int strideA_wb[3] = {1, 1, 1};  // not support blocking memory.
            for(int j=0; j<3; j++)
                dimA_wb[j] = wb_md.dims[j];

            // It means this weight/bias is disabled.
            if(dimA_wb[1] == 0) {
                proj_disabled[i] = true;
                continue;
            }

            proj_disabled[i] = false;
            strideA_wb[1] = dimA_wb[2];
            strideA_wb[0] = dimA_wb[1] * dimA_wb[2];
            
            CUDNN_EXECUTE_FUNC_V(cudnnCreateTensorDescriptor, weightbias_tdesc + i);
            CUDNN_EXECUTE_FUNC_V(cudnnSetTensorNdDescriptor, weightbias_tdesc[i], weight_type, 3, dimA_wb, strideA_wb);
            
            auto data_size = types::data_type_size(wb_md.data_type);
            wSize[i] = data_size*dimA_wb[0]*dimA_wb[1]*dimA_wb[2];
        }
        return status::success;
    }

	status_t configure_parameters(const multi_head_attn_pd_t *pd) {
		// attention mode
		unsigned mode0 = pd->querymap_all_to_one() ? CUDNN_ATTN_QUERYMAP_ALL_TO_ONE : CUDNN_ATTN_QUERYMAP_ONE_TO_ONE;
		unsigned mode1 = pd->enable_proj_bias() ? CUDNN_ATTN_ENABLE_PROJ_BIASES : CUDNN_ATTN_DISABLE_PROJ_BIASES;
		attnMode = mode0 | mode1;
		
        // configure dnnl_descs, dimA and axes.
		CHECK(convert_dnnl_desc(io::q, pd->query_md(), pd->query_axes()));
		CHECK(convert_dnnl_desc(io::k, pd->key_md(), pd->key_axes()));
		CHECK(convert_dnnl_desc(io::v, pd->value_md(), pd->value_axes()));
		CHECK(convert_dnnl_desc(io::o, pd->output_md(), pd->output_axes()));

		// datatype
        CHECK(convert_data_type(&dnnl_descs[0], &data_types[0]));
        cudnnDataType_t qdt = data_types[0];
		for(int i=1; i<NUM_IO; i++){
			CHECK(convert_data_type(&dnnl_descs[i], &data_types[i]));
			if(data_types[i] != qdt)
				return status::invalid_arguments;
		}
		// only support CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT and CUDNN_DATA_HALF.
		if(!utils::one_of(qdt, CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT, CUDNN_DATA_HALF))
			return status::invalid_arguments;
		if(qdt == CUDNN_DATA_DOUBLE)
			computePrec = CUDNN_DATA_DOUBLE;

		nHeads = pd->num_heads();
		smScaler = pd->softmax_scaler();
        
        // get qkvo shape utils. 
        maxbatchsize = 0;
        maxbeamsize = 0;
		// qkv size equals to dimA[CUDNN_SEQDATA_VECT_DIM]
        for(int i=0; i<NUM_IO; i++) {
            // qkv project size
            if(pd->weight_md(i) != nullptr)
                projsize_QKVO[i] = pd->weight_md(i)->dims[1];
            else
                projsize_QKVO[i] = 0;

            // q/k/v seqlenth array size equals to batch*beam.
            size_t sla_size = 1;    
            for(int j=0; j<CUDNN_SEQDATA_DIM_COUNT; j++){
                // CUDNN_SEQDATA_VECT_DIM equals 3
                if(axes[i][j] == CUDNN_SEQDATA_VECT_DIM)
                    size_QKVO[i] = dimA[i][j];
                // CUDNN_SEQDATA_BATCH_DIM equals 1
                if(axes[i][j] == CUDNN_SEQDATA_BATCH_DIM){
                    if(dimA[i][j] > maxbatchsize)
                        maxbatchsize = dimA[i][j];
                    sla_size *= dimA[i][j];                    
                }
                // CUDNN_SEQDATA_BEAM_DIM equals 2
                if(axes[i][j] == CUDNN_SEQDATA_BEAM_DIM){
                    if(dimA[i][j] > maxbeamsize)
                        maxbatchsize = dimA[i][j];
                    sla_size *= dimA[i][j];
                }
            }
            seqlentharray_sizes[i] = sla_size;
        }

        // seqlength, probably all aligned.
        int* sl_q = pd->seqlength_Q();
        int* sl_k = pd->seqlength_K();
        int* sl_v = pd->seqlength_V();
        int* sl_o = pd->seqlength_O();

        get_seqlenth_utils(io::q, sl_q, &maxseqlength_qo);
        get_seqlenth_utils(io::k, sl_k, &maxseqlength_kv);
        get_seqlenth_utils(io::v, sl_v);
        get_seqlenth_utils(io::o, sl_o);

        p_currIdx = pd->currIdx();
        loWinIdx = pd->loWinIdxArray();
        hiWinIdx = pd->hiWinIdxArray();

        attndropout = pd->attn_dropout();
        postattndropout = pd->post_attn_dropout();
        dropoutseed = pd->dropout_seed();
        postdropoutseed = pd->post_dropout_seed();

        return status::success;
    }

    status_t create_cudnn_descs(impl::engine_t *engine, const multi_head_attn_pd_t *pd) {
        auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(engine);
        impl::stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream = utils::downcast<nvidia::stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

		// dropout descriptor
		// since we didn't got the state storage here, 
		// DropoutDescriptor will be set when this primitive is executed.
		CUDNN_EXECUTE_FUNC_V(cudnnCreateDropoutDescriptor, &attnDropoutDesc);
		CUDNN_EXECUTE_FUNC_V(cudnnCreateDropoutDescriptor, &postDropoutDesc);
		CUDNN_EXECUTE_FUNC_V(cudnnDropoutGetStatesSize, handle, &Dropout_stateSize);
		
		CUDNN_EXECUTE_FUNC_V(cudnnCreateAttnDescriptor, &attnDesc);
		CUDNN_EXECUTE_FUNC_V(cudnnSetAttnDescriptor, attnDesc, attnMode, nHeads, smScaler, 
			    weight_type, computePrec, mathType, attnDropoutDesc, postDropoutDesc, size_QKVO[io::q],
                size_QKVO[io::k], size_QKVO[io::v], projsize_QKVO[io::q], projsize_QKVO[io::k],
                projsize_QKVO[io::v], projsize_QKVO[io::o], maxseqlength_qo, maxseqlength_kv,
                maxbatchsize, maxbeamsize);

        for(int i=0; i<NUM_IO; i++) {
            CUDNN_EXECUTE_FUNC_V(cudnnCreateSeqDataDescriptor, &SeqDataDescs[i]);
            CUDNN_EXECUTE_FUNC_V(cudnnSetSeqDataDescriptor, SeqDataDescs[i], data_types[i], 
                    CUDNN_SEQDATA_DIM_COUNT, dimA[i], (cudnnSeqDataAxis_t*)(axes[i]), seqlentharray_sizes[i],
                    seqlenthArray[i].data(), NULL);
        }

        return status::success;
    }

    virtual status_t init_scratchpad(
            impl::engine_t *engine, multi_head_attn_pd_t *pd) {
        
        auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(engine);
        impl::stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream = utils::downcast<nvidia::stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        CUDNN_EXECUTE_FUNC_V(cudnnGetMultiHeadAttnBuffers, handle, attnDesc, 
                &weightSizeInBytes, &workSpaceSizeInBytes, &reserveSpaceSizeInBytes);
        
        // buffers
        if(weightSizeInBytes > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_attn_weight, weightSizeInBytes,
                    size_t(1));
        if(workSpaceSizeInBytes > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_attn_workspace, workSpaceSizeInBytes,
                    size_t(1));
        if(reserveSpaceSizeInBytes > 0)
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

        return status::success;
    };

    virtual void execute(
            cudnnHandle_t handle, const std::vector<void *> &args) const = 0;
};

struct cudnn_multi_head_attn_fwd_impl_t : public cudnn_multi_head_attn_impl_base_t {
public:
    // virtual ~cudnn_multi_head_attn_impl_fwd_t() {}

    void execute(cudnnHandle_t handle, const std::vector<void *> &args) const override {
        auto devSeqLengthsQO = args[0], devSeqLengthsKV = args[1],
                queries = args[2], residuals = args[3], keys = args[4], 
                values = args[5], out = args[6];
        
        void* weightbias[8];
        for(int i=7; i<7+8; i++)
            weightbias[i-4] = args[i];
        // buffers
        void* weightspace = args[15];
        void* workspace = args[16];
        void* reservespace = args[17];

        void* attn_dropout_states = args[18];
        void* attn_post_dropout_states = args[19];

        // dropout
        CUDNN_EXECUTE_FUNC_V(cudnnSetDropoutDescriptor, attnDropoutDesc, handle,
                attndropout, attn_dropout_states, Dropout_stateSize, dropoutseed);
        CUDNN_EXECUTE_FUNC_V(cudnnSetDropoutDescriptor, postDropoutDesc, handle,
                postattndropout, attn_post_dropout_states, Dropout_stateSize, 
                postdropoutseed);

        // copy to weight buffer
        for(int i=0; i<8; i++) {
            if(!proj_disabled[i]){
                CUDNN_EXECUTE_FUNC_V(cudnnGetMultiHeadAttnWeights, handle, attnDesc, 
                        cudnnMultiHeadAttnWeightKind_t(i), weightSizeInBytes, 
                        weightspace, weightbias_tdesc[i], wAddr + i);

                CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)(wAddr[i]),
                        (CUdeviceptr)(weightbias[i]), wSize[i]);
                // CUDA_EXECUTE_FUNC(cudaMemcpy, wAddr[i], weightbias[i], 
                //         wSize[i], cudaMemcpyDeviceToDevice);
            }
        }

        // forward
        CUDNN_EXECUTE_FUNC_V(cudnnMultiHeadAttnForward, handle, attnDesc, *p_currIdx, loWinIdx, 
                hiWinIdx, (int*)devSeqLengthsQO, (int*)devSeqLengthsKV, SeqDataDescs[io::q],
                queries, residuals, SeqDataDescs[io::k], keys, SeqDataDescs[io::v], values,
                SeqDataDescs[io::o], out, weightSizeInBytes, weightspace, workSpaceSizeInBytes,
                workspace, reserveSpaceSizeInBytes, reservespace);
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
