#ifndef MODEL_H
#define MODEL_H

#include "Tensor.hpp"
#include "LinearLayer.hpp"
#include "ConvLayer.hpp"
#include "MaxPoolLayer.hpp"
#include "AvgPoolLayer.hpp"
#include <string>
#include <memory>

const int n = 1;
const int c = 3;
const int h = 224, w = 224;
extern std::shared_ptr<Tensor<float>> cudainput;
extern std::shared_ptr<Tensor<float>> cudaoutput;
extern float* cpu_result;
extern ConvLayer* conv1;
extern MaxPoolLayer* mp1;
extern ConvLayer* layer1_0_conv1;
extern ConvLayer* layer1_0_conv2;
    // block2
extern ConvLayer* layer1_1_conv1;
extern ConvLayer* layer1_1_conv2;

// layer2

    // block1
extern ConvLayer* layer2_0_conv1;
extern ConvLayer* layer2_0_conv2;

    // downsample
extern ConvLayer* layer2_0_downsample_0_conv;

    // block2
extern ConvLayer* layer2_1_conv1;
extern ConvLayer* layer2_1_conv2;

// layer3

    // block1
extern ConvLayer* layer3_0_conv1;
extern ConvLayer* layer3_0_conv2;

    // downsample
extern ConvLayer* layer3_0_downsample_0_conv;

    // block2
extern ConvLayer* layer3_1_conv1;
extern ConvLayer* layer3_1_conv2;

// layer3

    // block1
extern ConvLayer* layer4_0_conv1;
extern ConvLayer* layer4_0_conv2;

    // downsample
extern ConvLayer* layer4_0_downsample_0_conv;

    // block2
extern ConvLayer* layer4_1_conv1;
extern ConvLayer* layer4_1_conv2;

// head
extern AvgPoolLayer* avg_pool;
extern LinearLayer* linear;

void initModel(int argc, const char** argv);
void inference(float *input, float *output);

#endif //CUDA_PROJ_MODEL_H





