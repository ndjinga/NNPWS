#ifndef FAST_INFERENCE_HXX
#define FAST_INFERENCE_HXX

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <cmath>
#include <torch/script.h>

struct FastLayer {
    int rows;
    int cols;
    std::vector<double> weights;
    std::vector<double> biases;
};

struct RegionData {
    bool is_valid = false;
    std::vector<FastLayer> layers;

    std::vector<double> in_mean; //[T_mean, P_mean]
    std::vector<double> in_std;
    std::vector<double> out_mean;
    std::vector<double> out_std;
};

struct FastResult {
    double G;
    double dG_dP;
    double dG_dT;

    double d2G_dP2;
    double d2G_dT2;
    double d2G_dPdT;
};

class FastInference {
public:
    FastInference();
    ~FastInference();

    void load_from_module(std::shared_ptr<torch::jit::script::Module> module, const std::vector<int>& regions_to_load);


    FastResult compute(int region_id, double p, double T) const;

private:
    std::map<int, RegionData> regions_map_;

    mutable std::vector<double> buf_val, buf_dp, buf_dt;
    mutable std::vector<double> buf_d2p, buf_d2t, buf_d2pt;


    mutable std::vector<double> next_val, next_dp, next_dt;
    mutable std::vector<double> next_d2p, next_d2t, next_d2pt;

    void ensure_buffers_size(size_t size) const;
};

#endif // FAST_INFERENCE_HXX