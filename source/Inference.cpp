#include "Model.hpp"
#include "helper_functions.cuh"
#include "argparse.hpp"
#include <iostream>

std::shared_ptr<Tensor<float>> cudainput;
std::shared_ptr<Tensor<float>> cudaoutput;
float* cpu_result;
ConvLayer* conv1;
MaxPoolLayer* mp1;
ConvLayer* layer1_0_conv1;
ConvLayer* layer1_0_conv2;
    // block2
ConvLayer* layer1_1_conv1;
ConvLayer* layer1_1_conv2;

// layer2

    // block1
ConvLayer* layer2_0_conv1;
ConvLayer* layer2_0_conv2;

    // downsample
ConvLayer* layer2_0_downsample_0_conv;

    // block2
ConvLayer* layer2_1_conv1;
ConvLayer* layer2_1_conv2;

// layer3

    // block1
ConvLayer* layer3_0_conv1;
ConvLayer* layer3_0_conv2;

    // downsample
ConvLayer* layer3_0_downsample_0_conv;

    // block2
ConvLayer* layer3_1_conv1;
ConvLayer* layer3_1_conv2;

// layer3

    // block1
ConvLayer* layer4_0_conv1;
ConvLayer* layer4_0_conv2;

    // downsample
ConvLayer* layer4_0_downsample_0_conv;

    // block2
ConvLayer* layer4_1_conv1;
ConvLayer* layer4_1_conv2;

// head
AvgPoolLayer* avg_pool;
LinearLayer* linear;


void initModel(int argc, const char** argv){
    argparse::ArgumentParser parser;
    try {
        parser.addArgument("--weights_dir", 1, false);
        parser.parse(argc, argv);
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    InitializeCUDA(0);

    cudainput = std::make_shared< Tensor<float> >(Size({n, c, h, w}));
    std::string w_path = parser.retrieve<std::string>("weights_dir").c_str();

    conv1 = new ConvLayer(w_path+"conv1", 3, 2, true, true);
    mp1 = new MaxPoolLayer(3, 2, 1);

    conv1->set_input(cudainput);
    mp1->set_input(conv1->get_output());


    // layer1

        // block1
    layer1_0_conv1 = new ConvLayer(w_path+"layer1.0.conv1", 1, 1, true, true);
    layer1_0_conv2 = new ConvLayer(w_path+"layer1.0.conv2", 1, 1, true);

    layer1_0_conv1->set_input(mp1->get_output());
    layer1_0_conv2->set_input(layer1_0_conv1->get_output());

        // block2
    layer1_1_conv1 = new ConvLayer(w_path+"layer1.1.conv1", 1, 1, true, true);
    layer1_1_conv2 = new ConvLayer(w_path+"layer1.1.conv2", 1, 1, true);

    layer1_1_conv1->set_input(layer1_0_conv2->get_output());
    layer1_1_conv2->set_input(layer1_1_conv1->get_output());


    // layer2

        // block1
    layer2_0_conv1 = new ConvLayer(w_path+"layer2.0.conv1", 1, 2, true, true);
    layer2_0_conv2 = new ConvLayer(w_path+"layer2.0.conv2", 1, 1, true);

        // downsample
    layer2_0_downsample_0_conv = new ConvLayer(w_path+"layer2.0.downsample.0", 0, 2, true);

    layer2_0_conv1->set_input(layer1_1_conv2->get_output());
    layer2_0_conv2->set_input(layer2_0_conv1->get_output());

        // downsample
    layer2_0_downsample_0_conv->set_input(layer1_1_conv2->get_output());

        // block2
    layer2_1_conv1 = new ConvLayer(w_path+"layer2.1.conv1", 1, 1, true, true);
    layer2_1_conv2 = new ConvLayer(w_path+"layer2.1.conv2", 1, 1, true);

    layer2_1_conv1->set_input(layer2_0_conv2->get_output());
    layer2_1_conv2->set_input(layer2_1_conv1->get_output());


    // layer3

        // block1
    layer3_0_conv1 = new ConvLayer(w_path+"layer3.0.conv1", 1, 2, true, true);
    layer3_0_conv2 = new ConvLayer(w_path+"layer3.0.conv2", 1, 1, true);

        // downsample
    layer3_0_downsample_0_conv = new ConvLayer(w_path+"layer3.0.downsample.0", 0, 2, true);

    layer3_0_conv1->set_input(layer2_1_conv2->get_output());
    layer3_0_conv2->set_input(layer3_0_conv1->get_output());

    layer3_0_downsample_0_conv->set_input(layer2_1_conv2->get_output());

        // block2
    layer3_1_conv1 = new ConvLayer(w_path+"layer3.1.conv1", 1, 1, true, true);
    layer3_1_conv2 = new ConvLayer(w_path+"layer3.1.conv2", 1, 1, true);

    layer3_1_conv1->set_input(layer3_0_conv2->get_output());
    layer3_1_conv2->set_input(layer3_1_conv1->get_output());
    // layer3

        // block1
    layer4_0_conv1 = new ConvLayer(w_path+"layer4.0.conv1", 1, 2, true, true);
    layer4_0_conv2 = new ConvLayer(w_path+"layer4.0.conv2", 1, 1, true);

        // downsample
    layer4_0_downsample_0_conv = new ConvLayer(w_path+"layer4.0.downsample.0", 0, 2, true);

    layer4_0_conv1->set_input(layer3_1_conv2->get_output());
    layer4_0_conv2->set_input(layer4_0_conv1->get_output());

    layer4_0_downsample_0_conv->set_input(layer3_1_conv2->get_output());

        // block2
    layer4_1_conv1 = new ConvLayer(w_path+"layer4.1.conv1", 1, 1, true, true);
    layer4_1_conv2 = new ConvLayer(w_path+"layer4.1.conv2", 1, 1, true);

    layer4_1_conv1->set_input(layer4_0_conv2->get_output());
    layer4_1_conv2->set_input(layer4_1_conv1->get_output());


    // head
    avg_pool = new AvgPoolLayer(7);
    linear = new LinearLayer(w_path+"fc", true);

    avg_pool->set_input(layer4_1_conv2->get_output());
    linear->set_input(avg_pool->get_output());


    cudaoutput = linear->get_output();
    cpu_result = (float*) malloc(cudaoutput->count()*sizeof(float));   
}


void inference(float *input, float *output){
    
    cudainput->from_cpu(input);
    conv1->forward();
    mp1->forward();
      
    // layer1
    layer1_0_conv1->forward();
    layer1_0_conv2->forward();
    layer1_0_conv2->get_output()->add_relu(*(mp1->get_output()));
 
    layer1_1_conv1->forward();
    layer1_1_conv2->forward();
    layer1_1_conv2->get_output()->add_relu(*(layer1_0_conv2->get_output()));

    // layer2

    
    
    layer2_0_conv1->forward();
    layer2_0_conv2->forward();
    layer2_0_downsample_0_conv->forward();
    layer2_0_conv2->get_output()->add_relu(*(layer2_0_downsample_0_conv->get_output()));

    
    
    layer2_1_conv1->forward();
    layer2_1_conv2->forward();
    layer2_1_conv2->get_output()->add_relu(*(layer2_0_conv2->get_output()));

    
    
    // layer3

    layer3_0_conv1->forward();
    layer3_0_conv2->forward();
    layer3_0_downsample_0_conv->forward();
    layer3_0_conv2->get_output()->add_relu(*(layer3_0_downsample_0_conv->get_output()));

    
    
    layer3_1_conv1->forward();
    layer3_1_conv2->forward();
    layer3_1_conv2->get_output()->add_relu(*(layer3_0_conv2->get_output()));

    // layer4

    
    
    layer4_0_conv1->forward();
    layer4_0_conv2->forward();
    layer4_0_downsample_0_conv->forward();
    layer4_0_conv2->get_output()->add_relu(*(layer4_0_downsample_0_conv->get_output()));


    layer4_1_conv1->forward();
    layer4_1_conv2->forward();
    layer4_1_conv2->get_output()->add_relu(*(layer4_0_conv2->get_output()));


    
    
    // head
    avg_pool->forward();
    avg_pool->get_output()->reshape({n, avg_pool->get_output()->count()/n});

    linear->forward();

    
    
    cudaDeviceSynchronize();
    cudaoutput->to_cpu(output);
}