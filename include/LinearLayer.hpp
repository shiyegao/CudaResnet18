#ifndef CUDA_PROJ_LINEARLAYER_CUH
#define CUDA_PROJ_LINEARLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>
#include <memory>


class LinearLayer: public Layer {
public:
    LinearLayer(const std::string& w_path, bool bias=true);
    ~LinearLayer();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);
    std::shared_ptr<Tensor<float>> get_output();
    int get_output_dim();

private:
    int batch_size, input_dim, output_dim;
    std::shared_ptr<Tensor<float>> _input, _w, _b, _res, _tmp, _w2;
    std::vector<float> data_b;
    bool _bias;

};

#endif //CUDA_PROJ_LINEARLAYER_CUH
