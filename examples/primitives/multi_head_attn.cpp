/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/// @example multi_head_attn.cpp
/// > Annotated version: @ref multi_head_attn_example_cpp
///
/// @page multi_head_attn_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [multi_head_attn](@ref dev_guide_multi_head_attn) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page multi_head_attn_example_cpp multi_head_attn Primitive Example
/// @copydetails multi_head_attn_example_cpp_short
///
/// @include multi_head_attn.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include "example_utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

enum platform {
    cuda = 0,
    rocm = 1,
};

platform p = rocm;

void maulti_head_attn_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim TIME = 8, 
            BATCH = 8, BEAM = 8, VECT = 64, PROJ = 8;
    
    int num_head = 8;
    double smScalar = 0.01;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims devSeqLengthsQO_dims = {BATCH*BEAM};
    memory::dims devSeqLengthsKV_dims = {BATCH*BEAM};

    memory::dims query_dims = {TIME, BATCH, BEAM, VECT};
    memory::dims key_dims = {TIME, BATCH, BEAM, VECT};
    memory::dims value_dims = {TIME, BATCH, BEAM, VECT};
    memory::dims output_dims = {TIME, BATCH, BEAM, VECT};

    memory::dims qw_dims = {num_head, PROJ, VECT};
	memory::dims qb_dims = {num_head, PROJ, 1};
	memory::dims kw_dims = {num_head, PROJ, VECT};
	memory::dims kb_dims = {num_head, PROJ, 1};
	memory::dims vw_dims = {num_head, PROJ, VECT};
	memory::dims vb_dims = {num_head, PROJ, 1};
	memory::dims ow_dims = {num_head, PROJ, VECT};
	memory::dims ob_dims = {num_head, PROJ, 1};

	int qkvo_axes[4] = {0, 1, 2, 3};
	int* seqlength_QKVO = (int*)malloc(sizeof(int)*BATCH*BEAM);
	for(int i=0; i<BATCH*BEAM; i++){
		seqlength_QKVO[i] = TIME;
	}

	int currIdx = -1;
	int* p_currIdx = &currIdx;
	std::vector<int> loWinIdx(TIME, 0);
	std::vector<int> hiWinIdx(TIME, TIME);

	float dropout = 0.5;
	unsigned long long seed = 1;

    // Allocate buffers.
    std::vector<float> qkv_data(product(query_dims));
    std::vector<float> qkv_proj_weights_data(product(qw_dims));
    std::vector<float> qkv_proj_bias_data(product(qb_dims));
	std::vector<float> output_data(product(output_dims));

    // Initialize src, weights, bias.
    std::generate(qkv_data.begin(), qkv_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(qkv_proj_weights_data.begin(), qkv_proj_weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(qkv_proj_bias_data.begin(), qkv_proj_bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
	auto SeqLengths_md = memory::desc(devSeqLengthsQO_dims, dt::s32, tag::a);
    auto qkvo_md = memory::desc(query_dims, dt::f32, tag::abcd);
    auto weights_md = memory::desc(qw_dims, dt::f32, tag::abc);
    auto bias_md = memory::desc(qb_dims, dt::f32, tag::abc);

	auto devSeqLengthsQO_mem = sycl_interop::make_memory(SeqLengths_md, engine, sycl_interop::memory_kind::buffer);
	auto devSeqLengthsKV_mem = sycl_interop::make_memory(SeqLengths_md, engine, sycl_interop::memory_kind::buffer);
	auto query_mem = sycl_interop::make_memory(qkvo_md, engine, sycl_interop::memory_kind::buffer);
	auto residuals_mem = memory();
	auto key_mem = sycl_interop::make_memory(qkvo_md, engine, sycl_interop::memory_kind::buffer);
	auto value_mem = sycl_interop::make_memory(qkvo_md, engine, sycl_interop::memory_kind::buffer);
	auto output_mem = sycl_interop::make_memory(qkvo_md, engine, sycl_interop::memory_kind::buffer);

    auto qw_mem = sycl_interop::make_memory(weights_md, engine, sycl_interop::memory_kind::buffer);
    auto qb_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::buffer);
    auto kw_mem = sycl_interop::make_memory(weights_md, engine, sycl_interop::memory_kind::buffer);
    auto kb_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::buffer);
    auto vw_mem = sycl_interop::make_memory(weights_md, engine, sycl_interop::memory_kind::buffer);
    auto vb_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::buffer);
    auto ow_mem = sycl_interop::make_memory(weights_md, engine, sycl_interop::memory_kind::buffer);
    auto ob_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::buffer);

    // Write data to memory object's handles.
    write_to_dnnl_memory(qkv_data.data(), query_mem);
	write_to_dnnl_memory(qkv_data.data(), key_mem);
	write_to_dnnl_memory(qkv_data.data(), value_mem);
	write_to_dnnl_memory(qkv_proj_weights_data.data(), qw_mem);
	write_to_dnnl_memory(qkv_proj_bias_data.data(), qb_mem);
	write_to_dnnl_memory(qkv_proj_weights_data.data(), kw_mem);
	write_to_dnnl_memory(qkv_proj_bias_data.data(), kb_mem);
	write_to_dnnl_memory(qkv_proj_weights_data.data(), vw_mem);
    write_to_dnnl_memory(qkv_proj_bias_data.data(), vb_mem);
	write_to_dnnl_memory(qkv_proj_weights_data.data(), ow_mem);
	write_to_dnnl_memory(qkv_proj_bias_data.data(), ob_mem);

	write_to_dnnl_memory(seqlength_QKVO, devSeqLengthsQO_mem);
	write_to_dnnl_memory(seqlength_QKVO, devSeqLengthsKV_mem);

    // Create primitive descriptor.
    multi_head_attn_forward::primitive_desc attn_pd;
    if(p == platform::cuda)
        attn_pd = multi_head_attn_forward::primitive_desc(
            engine, prop_kind::forward_training, algorithm::attn_querymap_one2one, SeqLengths_md, SeqLengths_md, qkvo_md, qkvo_md, 
			qkvo_md, qkvo_md, qkvo_md, weights_md, bias_md, weights_md, bias_md, weights_md, bias_md,
			weights_md, bias_md, num_head, smScalar, qkvo_axes, seqlength_QKVO, qkvo_axes, seqlength_QKVO,
			qkvo_axes, seqlength_QKVO, qkvo_axes, seqlength_QKVO, p_currIdx, loWinIdx.data(), 
			hiWinIdx.data(), dropout, seed, dropout, seed);
    if(p == platform::rocm)
        // Create primitive descriptor.
        attn_pd = multi_head_attn_forward::primitive_desc(
                engine, prop_kind::forward_training, qkvo_md, memory::desc(), qkvo_md, 
                qkvo_md, qkvo_md, weights_md, bias_md, weights_md, bias_md, 
                weights_md, bias_md, weights_md, bias_md, num_head, smScalar, 
                qkvo_axes, qkvo_axes, qkvo_axes, qkvo_axes, dropout, 
                seed, dropout, seed);

    // workspace
    memory::dim ws_size = attn_pd.query_s64(query::workspace_md);
    // reservespace for dropout layer
    memory::dim rs_size = attn_pd.query_s64(query::scratchpad_md);
    printf("attn workspace size:%ld, reservespace size:%ld\n", ws_size, rs_size);

    auto ws_md = memory::desc({ws_size}, dt::s8, tag::a);
	auto ws_mem = sycl_interop::make_memory(ws_md, engine, sycl_interop::memory_kind::buffer);
    auto rs_md = memory::desc({rs_size}, dt::s8, tag::a);
	auto rs_mem = sycl_interop::make_memory(rs_md, engine, sycl_interop::memory_kind::buffer);
    
    memory wei_pack_mem;
    if(p == platform::cuda) {
        memory::dim wei_size = attn_pd.query_s64(query::weights_md);
        auto wei_pack_md = memory::desc({wei_size}, dt::s8, tag::a);
        wei_pack_mem = sycl_interop::make_memory(wei_pack_md, 
                engine, sycl_interop::memory_kind::buffer);
    }

    // Create the primitive.
    auto attn_prim = multi_head_attn_forward(attn_pd);

    // return ;
    // Primitive arguments.
    std::unordered_map<int, memory> attn_args;
    if(p == platform::cuda) {
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC, devSeqLengthsQO_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 1, devSeqLengthsKV_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 2, query_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 3, residuals_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 4, key_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 5, value_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 6, qw_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 7, qb_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 8, kw_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 9, kb_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 10, vw_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 11, vb_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 12, ow_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 13, ob_mem});

        attn_args.insert({DNNL_ARG_WEIGHTS, wei_pack_mem});
    }
    else if(p == platform::rocm) {
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 0, query_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 1, residuals_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 2, key_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 3, value_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 4, qw_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 5, qb_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 6, kw_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 7, kb_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 8, vw_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 9, vb_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 10, ow_mem});
        attn_args.insert({DNNL_ARG_MULTIPLE_SRC + 11, ob_mem});
    }

    attn_args.insert({DNNL_ARG_WORKSPACE, ws_mem});
    attn_args.insert({DNNL_ARG_SCRATCHPAD, rs_mem});
	attn_args.insert({DNNL_ARG_DST, output_mem});

    // Primitive execution: matrix multiplication with ReLU.
    attn_prim.execute(engine_stream, attn_args);

    // Wait for the computation to finalize.
    engine_stream.wait();
    
    // Read data from memory object's handle.
    read_from_dnnl_memory(output_data.data(), output_mem);

    printf("forward over\n");
    
    // backward data
    auto attn_bd_pd = multi_head_attn_backward_data::primitive_desc(engine, attn_pd);
    auto attn_bd_prim = multi_head_attn_backward_data(attn_bd_pd);

    std::unordered_map<int, memory> attn_bd_args;
    if(p == platform::cuda) {

    }
    else if(p == platform::rocm) {
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 0, output_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 1, qw_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 2, qb_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 3, kw_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 4, kb_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 5, vw_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 6, vb_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 7, ow_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_SRC + 8, ob_mem});

        attn_bd_args.insert({DNNL_ARG_MULTIPLE_DST + 0, query_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_DST + 1, key_mem});
        attn_bd_args.insert({DNNL_ARG_MULTIPLE_DST + 2, value_mem});
    }
    attn_bd_args.insert({DNNL_ARG_WORKSPACE, ws_mem});
    attn_bd_args.insert({DNNL_ARG_SCRATCHPAD, rs_mem});

    attn_bd_prim.execute(engine_stream, attn_bd_args);
    engine_stream.wait();
    printf("backward_data over\n");

    auto dqw_mem = sycl_interop::make_memory(weights_md, engine, sycl_interop::memory_kind::buffer);
    auto dqb_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::buffer);
    auto dkw_mem = sycl_interop::make_memory(weights_md, engine, sycl_interop::memory_kind::buffer);
    auto dkb_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::buffer);
    auto dvw_mem = sycl_interop::make_memory(weights_md, engine, sycl_interop::memory_kind::buffer);
    auto dvb_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::buffer);
    auto dow_mem = sycl_interop::make_memory(weights_md, engine, sycl_interop::memory_kind::buffer);
    auto dob_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::buffer);

    auto attn_bw_pd = multi_head_attn_backward_weights::primitive_desc(engine, 
            algorithm::attn_wgrad_add, attn_pd);
    auto attn_bw_prim = multi_head_attn_backward_weights(attn_bw_pd);

    std::unordered_map<int, memory> attn_bw_args;
    if(p == platform::cuda) {

    }
    else if(p == platform::rocm) {
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 0, query_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 1, key_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 2, value_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 3, output_mem});   // dout

        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 4, qw_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 5, qb_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 6, kw_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 7, kb_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 8, vw_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 9, vb_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 10, ow_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_SRC + 11, ob_mem});

        attn_bw_args.insert({DNNL_ARG_MULTIPLE_DST + 0, dqw_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_DST + 1, dqb_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_DST + 2, dkw_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_DST + 3, dkb_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_DST + 4, dvw_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_DST + 5, dvb_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_DST + 6, dow_mem});
        attn_bw_args.insert({DNNL_ARG_MULTIPLE_DST + 7, dob_mem});
    }
    attn_bw_args.insert({DNNL_ARG_WORKSPACE, ws_mem});
    attn_bw_args.insert({DNNL_ARG_SCRATCHPAD, rs_mem});

    attn_bw_prim.execute(engine_stream, attn_bw_args);
    engine_stream.wait();
    printf("backward_weights over\n");

    free(seqlength_QKVO);
}
