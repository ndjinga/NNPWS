#include "FastInference.hxx"
#include <fstream>
#include <iostream>

#include "json.hpp"
using json = nlohmann::json;

#ifdef NNPWS_USE_OPENMP
  #include <omp.h>
#endif

FastInference::ThreadData& FastInference::tls() {
    static thread_local ThreadData td;
    return td;
}

static std::vector<double> j2vec(const json& a) {
    std::vector<double> v;
    v.reserve(a.size());
    for (const auto& x : a) v.push_back(x.get<double>());
    return v;
}

static void load_region_into(RegionData& data, const json& m, const std::string& label) {
    data.is_valid = true;

    data.in_mean  = j2vec(m.at("input_norm").at("mean"));
    data.in_std   = j2vec(m.at("input_norm").at("std"));
    data.out_mean = j2vec(m.at("output_norm").at("mean"));
    data.out_std  = j2vec(m.at("output_norm").at("std"));

    if (data.in_mean.size() != 2 || data.in_std.size() != 2) {
        throw std::runtime_error("[FastInference] Bad input_norm size for " + label);
    }
    /*
    if (data.out_mean.size() != 6 || data.out_std.size() != 6) {
        throw std::runtime_error("[FastInference] Bad output_norm size for " + label);
    }
    */

    data.layers.clear();
    data.max_width = 0;

    for (const auto& L : m.at("layers")) {
        if (const std::string t = L.at("type").get<std::string>(); t != "Linear") continue;

        FastLayer fl;
        fl.rows = L.at("out").get<int>();
        fl.cols = L.at("in").get<int>();
        fl.weights = j2vec(L.at("W"));
        fl.biases  = j2vec(L.at("b"));

        if (static_cast<int>(fl.weights.size()) != fl.rows * fl.cols) {
            throw std::runtime_error("[FastInference] Bad W size for " + label);
        }
        if (static_cast<int>(fl.biases.size()) != fl.rows) {
            throw std::runtime_error("[FastInference] Bad b size for " + label);
        }

        data.max_width = std::max(data.max_width, static_cast<size_t>(fl.rows));
        data.layers.push_back(std::move(fl));
    }

    if (data.layers.empty()) {
        throw std::runtime_error("[FastInference] No Linear layers found for " + label);
    }

    data.max_width = std::max<size_t>(data.max_width, 2);
}

void FastInference::load_from_json_file(const std::string& json_path, const std::vector<int>& regions_to_load) {
    std::ifstream f(json_path);
    if (!f) throw std::runtime_error("[FastInference] Cannot open JSON: " + json_path);

    json root;
    f >> root;

    const json& models = root.at("models");

    regions_map_.clear();

    for (int r : regions_to_load) {
        const std::string rid = std::to_string(r);
        if (!models.contains(rid)) continue;

        RegionData data;
        load_region_into(data, models.at(rid), "region " + rid);
        regions_map_[r] = std::move(data);
    }

    if (regions_map_.empty()) {
        throw std::runtime_error("[FastInference] No regions loaded from: " + json_path);
    }
}

void FastInference::load_secondary_from_json_file(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f) throw std::runtime_error("[FastInference] Cannot open PH JSON: " + json_path);

    json root;
    f >> root;

    const json& models = root.at("models");
    if (!models.contains("0")) {
        throw std::runtime_error("[FastInference] PH JSON must contain models['0']");
    }

    regions_map_.clear();

    RegionData data;
    load_region_into(data, models.at("0"), "PH model (0)");
    regions_map_[0] = std::move(data);
}

FastResult FastInference::compute(int region_id, double p_real, double T_real) const {
    const auto it = regions_map_.find(region_id);
    if (it == regions_map_.end() || !it->second.is_valid) {
        throw std::runtime_error("[FastInference] Region not loaded: " + std::to_string(region_id));
    }
    const RegionData& data = it->second;

    ThreadData& ws = tls();
    ws.ensure(data.max_width);

    auto& buf_val  = ws.buf_val;
    auto& buf_dp   = ws.buf_dp;
    auto& buf_dt   = ws.buf_dt;
    auto& buf_d2p  = ws.buf_d2p;
    auto& buf_d2t  = ws.buf_d2t;
    auto& buf_d2pt = ws.buf_d2pt;

    auto& next_val  = ws.next_val;
    auto& next_dp   = ws.next_dp;
    auto& next_dt   = ws.next_dt;
    auto& next_d2p  = ws.next_d2p;
    auto& next_d2t  = ws.next_d2t;
    auto& next_d2pt = ws.next_d2pt;

    // Inputs order: [T, P]
    const double xT = (T_real - data.in_mean[0]) / data.in_std[0];
    const double xP = (p_real - data.in_mean[1]) / data.in_std[1];

    // neuron 0 = T_norm
    buf_val[0] = xT;
    buf_dp[0]  = 0.0;
    buf_dt[0]  = 1.0 / data.in_std[0];
    buf_d2p[0] = 0.0; buf_d2t[0] = 0.0; buf_d2pt[0] = 0.0;

    // neuron 1 = P_norm
    buf_val[1] = xP;
    buf_dp[1]  = 1.0 / data.in_std[1];
    buf_dt[1]  = 0.0;
    buf_d2p[1] = 0.0; buf_d2t[1] = 0.0; buf_d2pt[1] = 0.0;

    int n_curr = 2;

    for (size_t l_idx = 0; l_idx < data.layers.size(); ++l_idx) {
        const FastLayer& layer = data.layers[l_idx];
        const bool is_last = (l_idx == data.layers.size() - 1);
        const int n_next = layer.rows;

// #ifdef NNPWS_USE_OPENMP
//         #pragma omp parallel for schedule(static)
// #endif
        for (int r = 0; r < n_next; ++r) {
            double sum_val = layer.biases[r];
            double sum_dp = 0.0, sum_dt = 0.0;
            double sum_d2p = 0.0, sum_d2t = 0.0, sum_d2pt = 0.0;

            for (int c = 0; c < n_curr; ++c) {
                const double w = layer.weights[r * n_curr + c];
                sum_val  += w * buf_val[c];
                sum_dp   += w * buf_dp[c];
                sum_dt   += w * buf_dt[c];
                sum_d2p  += w * buf_d2p[c];
                sum_d2t  += w * buf_d2t[c];
                sum_d2pt += w * buf_d2pt[c];
            }

            if (is_last) {
                next_val[r]  = sum_val;
                next_dp[r]   = sum_dp;
                next_dt[r]   = sum_dt;
                next_d2p[r]  = sum_d2p;
                next_d2t[r]  = sum_d2t;
                next_d2pt[r] = sum_d2pt;
            } else {
                const double t = std::tanh(sum_val);
                const double f1 = 1.0 - t * t;          // f'(z) = 1 - tanh^2(z)
                const double f2 = -2.0 * t * f1;        // f''(z) = -2 * tanh(z) * (1 - tanh^2(z))

                next_val[r] = t;
                next_dp[r]  = f1 * sum_dp;
                next_dt[r]  = f1 * sum_dt;

                next_d2p[r]  = f1 * sum_d2p  + f2 * (sum_dp * sum_dp);
                next_d2t[r]  = f1 * sum_d2t  + f2 * (sum_dt * sum_dt);
                next_d2pt[r] = f1 * sum_d2pt + f2 * (sum_dp * sum_dt);
            }
        }

// #ifdef NNPWS_USE_OPENMP
//         #pragma omp parallel for schedule(static)
// #endif
        for (int i = 0; i < n_next; ++i) {
            buf_val[i]  = next_val[i];
            buf_dp[i]   = next_dp[i];
            buf_dt[i]   = next_dt[i];
            buf_d2p[i]  = next_d2p[i];
            buf_d2t[i]  = next_d2t[i];
            buf_d2pt[i] = next_d2pt[i];
        }

        n_curr = n_next;
    }

    const double out_std  = data.out_std[0];
    const double out_mean = data.out_mean[0];

    FastResult res{};
    res.G        = buf_val[0] * out_std + out_mean;
    res.dG_dP    = buf_dp[0]  * out_std;
    res.dG_dT    = buf_dt[0]  * out_std;
    res.d2G_dP2  = buf_d2p[0] * out_std;
    res.d2G_dT2  = buf_d2t[0] * out_std;
    res.d2G_dPdT = buf_d2pt[0]* out_std;
    return res;
}

double FastInference::compute_val(double p_real, double h_real) const {
    auto it = regions_map_.find(0);
    if (it == regions_map_.end() || !it->second.is_valid) {
        throw std::runtime_error("[FastInference] PH model not loaded (region 0)");
    }
    const RegionData& data = it->second;

    // network inputs: [H, P]
    const double xH = (h_real - data.in_mean[0]) / data.in_std[0];
    const double xP = (p_real - data.in_mean[1]) / data.in_std[1];

    std::vector<double> buf(data.max_width);
    std::vector<double> next_buf(data.max_width);

    buf[0] = xH;
    buf[1] = xP;

    int n_curr = 2;

    for (size_t l_idx = 0; l_idx < data.layers.size(); ++l_idx) {
        const FastLayer& layer = data.layers[l_idx];
        const bool is_last = (l_idx == data.layers.size() - 1);
        const int n_next = layer.rows;

// #ifdef NNPWS_USE_OPENMP
//         #pragma omp parallel for schedule(static)
// #endif
        for (int r = 0; r < n_next; ++r) {
            double sum = layer.biases[r];
            for (int c = 0; c < n_curr; ++c) {
                sum += layer.weights[r * n_curr + c] * buf[c];
            }
            next_buf[r] = is_last ? sum : std::tanh(sum);
        }

// #ifdef NNPWS_USE_OPENMP
//         #pragma omp parallel for schedule(static)
// #endif
        for (int i = 0; i < n_next; ++i) buf[i] = next_buf[i];

        n_curr = n_next;
    }

    // output denorm
    return buf[0] * data.out_std[0] + data.out_mean[0];
}
