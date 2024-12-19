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

/// @example matmul.cpp
/// > Annotated version: @ref matmul_example_cpp
///
/// @page matmul_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page matmul_example_cpp Matmul Primitive Example
/// @copydetails matmul_example_cpp_short
///
/// @include matmul.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void maulti_head_attn_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim TIME = 8, 
            BATCH = 8, BEAM = 8, VECT = 8, PROJ = 8;
    
    int num_head = 4;
    double smScalar;

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
	int* seqlength_QKVO = (int*)malloc(sizeof(int)*BATCH*BEAM)
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

	write_to_dnnl_memory(seqlength_QKVO.data(), devSeqLengthsQO_mem);
	write_to_dnnl_memory(seqlength_QKVO.data(), devSeqLengthsKV_mem);

    // Create primitive descriptor.
    auto attn_pd = multi_head_attn_forward::primitive_desc(
            engine, algorithm::attn_querymap_one2one, SeqLengths_md, SeqLengths_md, qkvo_md, qkvo_md, 
			qkvo_md, qkvo_md, qkvo_md, weights_md, bias_md, weights_md, bias_md, weights_md, bias_md,
			weights_md, bias_md, num_head, smScalar, qkvo_axes, seqlength_QKVO, qkvo_axes, seqlength_QKVO,
			qkvo_axes, seqlength_QKVO, qkvo_axes, seqlength_QKVO, p_currIdx, loWinIdx.data(), 
			hiWinIdx.data(), dropout, seed, dropout, seed);

    // Create the primitive.
    auto attn_prim = multi_head_attn_forward(attn_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> attn_args;
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

	attn_args.insert({DNNL_ARG_DST, output_mem});

    // Primitive execution: matrix multiplication with ReLU.
    attn_prim.execute(engine_stream, attn_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(output_data.data(), output_mem);
    free(seqlength_QKVO);
}

int main(int argc, char **argv) {
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}
