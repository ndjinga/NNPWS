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

    size_t max_width = 0;

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
    void load_secondary_from_module(std::shared_ptr<torch::jit::script::Module> module);

    FastResult compute(int region_id, double p, double T) const;
    double compute_val(double in1, double in2) const;

private:
    std::map<int, RegionData> regions_map_;

    struct Thread_data {
        size_t cap = 0;
        std::vector<double> buf_val, buf_dp, buf_dt;
        std::vector<double> buf_d2p, buf_d2t, buf_d2pt;
        std::vector<double> next_val, next_dp, next_dt;
        std::vector<double> next_d2p, next_d2t, next_d2pt;

        void ensure(size_t n) {
            if (cap >= n) return;
            cap = n;
            buf_val.resize(n);  buf_dp.resize(n);  buf_dt.resize(n);
            buf_d2p.resize(n);  buf_d2t.resize(n); buf_d2pt.resize(n);
            next_val.resize(n); next_dp.resize(n); next_dt.resize(n);
            next_d2p.resize(n); next_d2t.resize(n); next_d2pt.resize(n);
        }
    };

    static Thread_data& tls_workspace();
};

#endif // FAST_INFERENCE_HXX