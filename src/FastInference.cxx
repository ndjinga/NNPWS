#include "FastInference.hxx"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <regex>

#ifdef NNPWS_USE_OPENMP
  #include <omp.h>
#endif


FastInference::FastInference() {
}

FastInference::~FastInference() {}

FastInference::Thread_data& FastInference::tls_workspace() {
    static thread_local Thread_data ws;
    return ws;
}

static bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

void FastInference::load_from_module(std::shared_ptr<torch::jit::script::Module> module, const std::vector<int>& regions_to_load) {
    std::cout << "[FastInference] Initialisation..." << std::endl;
    regions_map_.clear();

    auto all_params = module->named_parameters(true);

    for (int r : regions_to_load) {
        RegionData r_data;
        std::string rid = std::to_string(r);

        std::string prefix_model   = "models." + rid + ".layers.";
        std::string prefix_scalers = "r" + rid + "_";

        try {
            if (module->hasattr(prefix_scalers + "in_mean")) {
                auto to_vec = [](torch::Tensor t) {
                    t = t.cpu().to(torch::kDouble).contiguous();
                    return std::vector<double>(t.data_ptr<double>(), t.data_ptr<double>() + t.numel());
                };
                r_data.in_mean  = to_vec(module->attr(prefix_scalers + "in_mean").toTensor());
                r_data.in_std   = to_vec(module->attr(prefix_scalers + "in_std").toTensor());
                r_data.out_mean = to_vec(module->attr(prefix_scalers + "out_mean").toTensor());
                r_data.out_std  = to_vec(module->attr(prefix_scalers + "out_std").toTensor());
            } else {
                continue;
            }
        } catch (...) { continue; }

        std::map<int, std::pair<torch::Tensor, torch::Tensor>> layers_map;

        for (const auto& p : all_params) {
            std::string name = p.name;

            if (starts_with(name, prefix_model)) {
                std::string suffix = name.substr(prefix_model.size());
                size_t dot_pos = suffix.find('.');
                if (dot_pos == std::string::npos) continue;

                std::string idx_str = suffix.substr(0, dot_pos);
                std::string type    = suffix.substr(dot_pos + 1);

                try {
                    int layer_idx = std::stoi(idx_str);
                    if (type == "weight") layers_map[layer_idx].first = p.value.cpu().contiguous();
                    if (type == "bias")   layers_map[layer_idx].second = p.value.cpu().contiguous();
                } catch (...) { continue; }
            }
        }

        if (layers_map.empty()) {
            std::cerr << "[FastInference] ERREUR: Aucun poids trouve pour region " << rid << std::endl;
            continue;
        }

        for (auto const& [idx, tensors] : layers_map) {
            torch::Tensor w = tensors.first;
            torch::Tensor b = tensors.second;

            if (!w.defined()) continue;

            FastLayer l;
            l.rows = w.size(0);
            l.cols = w.size(1);

            auto w_ptr = w.to(torch::kDouble).data_ptr<double>();
            l.weights.assign(w_ptr, w_ptr + l.rows * l.cols);

            if (b.defined()) {
                auto b_ptr = b.to(torch::kDouble).data_ptr<double>();
                l.biases.assign(b_ptr, b_ptr + l.rows);
            } else {
                l.biases.resize(l.rows, 0.0);
            }
            r_data.layers.push_back(l);

            r_data.max_width = std::max(r_data.max_width, static_cast<size_t>(l.rows));
        }

        r_data.max_width = 0;
        for (const auto& layer : r_data.layers) {
            r_data.max_width = std::max(r_data.max_width, static_cast<size_t>(layer.rows));
        }

        r_data.is_valid = true;
        r_data.max_width = std::max(r_data.max_width, static_cast<size_t>(2));
        regions_map_[r] = r_data;
        std::cout << "[FastInference] Region " << r << " chargee (" << r_data.layers.size() << " couches)." << std::endl;
    }
}

void FastInference::load_secondary_from_module(std::shared_ptr<torch::jit::script::Module> module) {
    std::cout << "[FastInference] Chargement du modele secondaire (PH)..." << std::endl;
    regions_map_.clear();

    int id = 0;
    RegionData& data = regions_map_[id];
    data.is_valid = true;

    auto get_attr = [&](const std::string& name) -> std::vector<double> {
        if (module->hasattr(name)) {
            torch::Tensor t = module->attr(name).toTensor().to(torch::kCPU).to(torch::kFloat64);
            return std::vector<double>(t.data_ptr<double>(), t.data_ptr<double>() + t.numel());
        }
        data.is_valid = false;
        std::cerr << "[WARN] Attribut '" << name << "' manquant dans le modele JIT." << std::endl;
        return {0.0, 1.0};
    };

    data.in_mean  = get_attr("in_mean");
    data.in_std   = get_attr("in_std");
    data.out_mean = get_attr("out_mean");
    data.out_std  = get_attr("out_std");

    if (data.is_valid)
        std::cout << "   > Normalisation chargee. Mean In: [" << data.in_mean[0] << ", " << data.in_mean[1] << "]" << std::endl;

    std::map<int, FastLayer> layers_map;

    auto params = module->named_parameters(true);
    for (const auto& p : params) {
        std::string name = p.name;

        std::regex layer_regex(".*\\.(\\d+)\\.(weight|bias)");
        std::smatch match;
        if (std::regex_search(name, match, layer_regex)) {
            int layer_idx = std::stoi(match[1]);
            int stored_idx = layer_idx / 2;

            torch::Tensor t = p.value.to(torch::kCPU).to(torch::kFloat64);
            std::vector<double> vals(t.data_ptr<double>(), t.data_ptr<double>() + t.numel());

            if (name.find("weight") != std::string::npos) {
                layers_map[stored_idx].weights = vals;
                layers_map[stored_idx].rows = t.size(0);
                layers_map[stored_idx].cols = t.size(1);
            } else if (name.find("bias") != std::string::npos) {
                layers_map[stored_idx].biases = vals;
            }
        }
    }

    for (auto const& [key, layer] : layers_map) {
        data.layers.push_back(layer);
    }

    data.max_width = 0;
    for (const auto& layer : data.layers) {
        data.max_width = std::max(data.max_width, static_cast<size_t>(layer.rows));
    }
    data.max_width = std::max(data.max_width, static_cast<size_t>(2));
    std::cout << "   > " << data.layers.size() << " couches extraites." << std::endl;
}

FastResult FastInference::compute(int region_id, double p_real, double T_real) const {
    auto it = regions_map_.find(region_id);
    if (it == regions_map_.end() || !it->second.is_valid) throw std::runtime_error("erreur");
    const RegionData& data = it->second;

    Thread_data& ws = tls_workspace();
    ws.ensure(std::max<size_t>(2, data.max_width));

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

    double x0 = (T_real - data.in_mean[0]) / data.in_std[0];
    double x1 = (p_real - data.in_mean[1]) / data.in_std[1];

    // Neurone 0 est la Température
    buf_val[0] = x0;
    buf_dp[0]  = 0.0;                    // T ne dépend pas de P
    buf_dt[0]  = 1.0 / data.in_std[0];   // d(T_norm)/dT = 1/std_T
    buf_d2p[0] = 0.0; buf_d2t[0] = 0.0; buf_d2pt[0] = 0.0;

    // Neurone 1 est la Pression
    buf_val[1] = x1;
    buf_dp[1]  = 1.0 / data.in_std[1];   // d(P_norm)/dP = 1/std_P
    buf_dt[1]  = 0.0;
    buf_d2p[1] = 0.0; buf_d2t[1] = 0.0; buf_d2pt[1] = 0.0;

    int n_curr = 2;

    for (size_t l_idx = 0; l_idx < data.layers.size(); ++l_idx) {
        const FastLayer& layer = data.layers[l_idx];
        bool is_last = (l_idx == data.layers.size() - 1);
        int n_next = layer.rows;
        ws.ensure(std::max<size_t>(static_cast<size_t>(n_next), 2));

#ifdef NNPWS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < n_next; ++r) {
            double sum_val = layer.biases[r];
            double sum_dp = 0.0; double sum_dt = 0.0;
            double sum_d2p = 0.0; double sum_d2t = 0.0; double sum_d2pt = 0.0;

            for (int c = 0; c < n_curr; ++c) {
                double w = layer.weights[r * n_curr + c];

                sum_val  += w * buf_val[c];
                sum_dp   += w * buf_dp[c];
                sum_dt   += w * buf_dt[c];
                sum_d2p  += w * buf_d2p[c];
                sum_d2t  += w * buf_d2t[c];
                sum_d2pt += w * buf_d2pt[c];
            }

            if (is_last) {
                next_val[r] = sum_val;
                next_dp[r]  = sum_dp;   next_dt[r]  = sum_dt;
                next_d2p[r] = sum_d2p;  next_d2t[r] = sum_d2t; next_d2pt[r] = sum_d2pt;
            } else {
                double tanh_z = std::tanh(sum_val);

                // Terme 1 : Dérivée première de tanh
                // f'(z) = 1 - tanh^2(z)
                double term1 = 1.0 - tanh_z * tanh_z; // f'

                // Terme 2 : Dérivée seconde de tanh
                // f''(z) = -2 * tanh(z) * (1 - tanh^2(z))
                double term2 = -2.0 * tanh_z * term1; // f''

                next_val[r] = tanh_z;
                next_dp[r] = term1 * sum_dp;
                next_dt[r] = term1 * sum_dt;

                next_d2p[r]  = term1 * sum_d2p  + term2 * (sum_dp * sum_dp);
                next_d2t[r]  = term1 * sum_d2t  + term2 * (sum_dt * sum_dt);
                next_d2pt[r] = term1 * sum_d2pt + term2 * (sum_dp * sum_dt);
            }
        }

#ifdef NNPWS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int i=0; i<n_next; ++i) {
            buf_val[i] = next_val[i];
            buf_dp[i] = next_dp[i]; buf_dt[i] = next_dt[i];
            buf_d2p[i] = next_d2p[i]; buf_d2t[i] = next_d2t[i]; buf_d2pt[i] = next_d2pt[i];
        }
        n_curr = n_next;
    }


    double out_std = data.out_std[0];
    double out_mean = data.out_mean[0];

    FastResult res;
    res.G = buf_val[0] * out_std + out_mean;
    res.dG_dP = buf_dp[0] * out_std;
    res.dG_dT = buf_dt[0] * out_std;
    res.d2G_dP2  = buf_d2p[0] * out_std;
    res.d2G_dT2  = buf_d2t[0] * out_std;
    res.d2G_dPdT = buf_d2pt[0] * out_std;

    return res;
}

double FastInference::compute_val(double val1_real, double val2_real) const {
    auto it = regions_map_.find(0);
    if (it == regions_map_.end() || !it->second.is_valid) throw std::runtime_error("erreur");
    const RegionData& data = it->second;

    size_t max_size = 0;
    for(const auto& l : data.layers) max_size = std::max(max_size, static_cast<size_t>(l.rows));
    max_size = std::max(max_size, static_cast<size_t>(2));

    std::vector<double> buf(max_size);
    std::vector<double> next_buf(max_size);

    double H_val = val2_real;
    double P_val = val1_real;

    buf[0] = (H_val - data.in_mean[0]) / data.in_std[0];
    buf[1] = (P_val - data.in_mean[1]) / data.in_std[1];


    int n_curr = 2;

    for (size_t l_idx = 0; l_idx < data.layers.size(); ++l_idx) {
        const FastLayer& layer = data.layers[l_idx];
        int n_next = layer.rows;
        bool is_last = (l_idx == data.layers.size() - 1);

#ifdef NNPWS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < n_next; ++r) {
            double sum = layer.biases[r];

            for (int c = 0; c < n_curr; ++c) {
                sum += layer.weights[r * n_curr + c] * buf[c];
            }

            if (is_last) {
                next_buf[r] = sum;
            } else {
                next_buf[r] = std::tanh(sum);
            }
        }

#ifdef NNPWS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int i=0; i<n_next; ++i) buf[i] = next_buf[i];
        n_curr = n_next;
    }

    double result = buf[0] * data.out_std[0] + data.out_mean[0];

    return result;
}