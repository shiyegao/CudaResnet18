#ifndef CUDA_PROJ_CONVLAYER_CUH
#define CUDA_PROJ_CONVLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>
#include <memory>


class ConvLayer: public Layer {
public:
    ConvLayer(const std::string& w_path, int stride=1, int pad=0, bool bias=true, bool Relu=false, bool extra_down=false);
    ~ConvLayer();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);
    std::shared_ptr<Tensor<float>> get_output();
    std::shared_ptr<Tensor<float>> get_extra_down();

private:
    std::shared_ptr<Tensor<float>> _input, _w, _b, _res;
    std::shared_ptr<Tensor<float>> _imcol, _imcol2, _wcol, _bcol, _tmp, _extra_down;
    std::vector<float> data_b;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int batch_size, N, C, H, W, _pad, _stride;
    int m;
    int n;
    int k;
    bool Relu;
    bool extra_down;
    bool input_set, _bias;

};

#endif //CUDA_PROJ_CONVLAYER_CUH
