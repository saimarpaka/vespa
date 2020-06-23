// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

struct Inputs {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char *> names;
    std::vector<std::vector<int64_t>> dim_sizes;
    Inputs(Ort::Session &session) {
        size_t num_inputs = session.GetInputCount();
        fprintf(stderr, "Number of inputs = %zu\n", num_inputs);
        names.resize(num_inputs, nullptr);
        dim_sizes.resize(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            names[i] = session.GetInputName(i, allocator);
            fprintf(stderr, "Input %zu : name=%s\n", i, names[i]);
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto element_type = tensor_info.GetElementType();
            fprintf(stderr, "Input %zu : type=%d\n", i, element_type);
            size_t num_dims = tensor_info.GetDimensionsCount();
            fprintf(stderr, "Input %zu : num_dims=%zu\n", i, num_dims);
            dim_sizes[i].resize(num_dims);
            tensor_info.GetDimensions(dim_sizes[i].data(), num_dims);
            for (size_t j = 0; j < num_dims; ++j) {
                fprintf(stderr, "Input %zu : dim %zu=%ld\n", i, j, dim_sizes[i][j]);
            }
        }
    }
    ~Inputs() {
        for (const char *name: names) {
            allocator.Free(const_cast<char*>(name));
        }
    }
};

struct Outputs {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char *> names;
    std::vector<std::vector<int64_t>> dim_sizes;
    Outputs(Ort::Session &session) {
        size_t num_outputs = session.GetOutputCount();
        fprintf(stderr, "Number of outputs = %zu\n", num_outputs);
        names.resize(num_outputs, nullptr);
        dim_sizes.resize(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            names[i] = session.GetOutputName(i, allocator);
            fprintf(stderr, "Output %zu : name=%s\n", i, names[i]);
            auto type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto element_type = tensor_info.GetElementType();
            fprintf(stderr, "Output %zu : type=%d\n", i, element_type);
            size_t num_dims = tensor_info.GetDimensionsCount();
            fprintf(stderr, "Output %zu : num_dims=%zu\n", i, num_dims);
            dim_sizes[i].resize(num_dims);
            tensor_info.GetDimensions(dim_sizes[i].data(), num_dims);
            for (size_t j = 0; j < num_dims; ++j) {
                fprintf(stderr, "Output %zu : dim %zu=%ld\n", i, j, dim_sizes[i][j]);
            }
        }
    }
    ~Outputs() {
        for (const char *name: names) {
            allocator.Free(const_cast<char*>(name));
        }
    }
};

int main(int, char**) {
    // setup model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session session(env, "squeezenet.onnx", session_options);
    Inputs inputs(session);
    Outputs outputs(session);

    // prepare input
    size_t input_tensor_size = 224 * 224 * 3;
    std::vector<float> input_tensor_values(input_tensor_size);
    for (size_t i = 0; i < input_tensor_size; ++i) {
        input_tensor_values[i] = float(i) / (input_tensor_size + 1);
    }

    // run model
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor(memory_info, input_tensor_values.data(), input_tensor_size, inputs.dim_sizes[0].data(), inputs.dim_sizes[0].size());
    auto result = session.Run(nullptr, inputs.names.data(), &input_tensor, 1, outputs.names.data(), 1);
    assert(result.size() == 1);
    assert(result[0].IsTensor());

    // print result
    auto raw = result[0].GetTensorMutableData<float>();
    assert(std::abs(raw[0] - 0.000045) < 1e-6);
    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "Score for class [%d] =  %f\n", i, raw[i]);
    }
    return 0;
}
