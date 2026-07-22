#include "NNPWS.hxx"

#ifdef NNPWS_USE_OPENMP
  #include <omp.h>
#endif

NNPWS::NNPWS(inputPair varnames) : inputPr_(varnames) {}

#if __cplusplus >= 201703L
NNPWS::NNPWS(inputPair varnames, double var1, double var2,
             const std::string& path_main_model,
             const std::optional<const std::string>& path_secondary_model)
: inputPr_(varnames)
{
    setNeuralNetworks(path_main_model, path_secondary_model);

    switch(varnames) {
        case PT: setPT(var1, var2); break;
        case PH: setPH(var1, var2); break;
        default: throw std::runtime_error("not implemented");
    }
}
#else
NNPWS::NNPWS(inputPair varnames, double var1, double var2,
             const std::string& path_main_model,
             const std::string& path_secondary_model)
: inputPr_(varnames)
{
    setNeuralNetworks(path_main_model, path_secondary_model);

    switch(varnames) {
        case PT: setPT(var1, var2); break;
        case PH: setPH(var1, var2); break;
        default: throw std::runtime_error("not implemented");
    }
}
#endif

#if __cplusplus >= 201703L
NNPWS::NNPWS(inputPair varnames,
             const std::string& path_main_model,
             const std::optional<const std::string>& path_secondary_model)
: inputPr_(varnames)
{
    setNeuralNetworks(path_main_model, path_secondary_model);
}
#else
NNPWS::NNPWS(inputPair varnames,
             const std::string& path_main_model,
             const std::string& path_secondary_model)
: inputPr_(varnames)
{
    setNeuralNetworks(path_main_model, path_secondary_model);
}
#endif

#if __cplusplus >= 201703L
void NNPWS::setNeuralNetworks(const std::string& path_main_model,
                              const std::optional<const std::string>& path_secondary_model)
{
    // JSON is the default for FastInference
    if (ends_with(path_main_model, ".json")) {
        fast_engine_.load_from_json_file(path_main_model, {1,2,3,4,5});
        is_initialized_ = true;
        path_main_model_ = path_main_model;

        if (inputPr_ == PH) {
            if (!path_secondary_model.has_value())
                throw std::runtime_error("PH requires secondary model JSON");
            if (!ends_with(*path_secondary_model, ".json"))
                throw std::runtime_error("PH secondary must be .json");

            fast_engine_backward_.load_secondary_from_json_file(*path_secondary_model);
            path_secondary_model_ = *path_secondary_model;
        }
        return;
    }

    throw std::runtime_error("provide .json model paths for single calls");
}
#else
void NNPWS::setNeuralNetworks(const std::string& path_main_model,
                              const std::string& path_secondary_model)
{
    // JSON is the default for FastInference
    if (ends_with(path_main_model, ".json")) {
        fast_engine_.load_from_json_file(path_main_model, {1,2,3,4,5});
        is_initialized_ = true;
        path_main_model_ = path_main_model;

        if (inputPr_ == PH) {
            if (!path_secondary_model.compare(""))
                throw std::runtime_error("PH requires secondary model JSON");
            if (!ends_with(path_secondary_model, ".json"))
                throw std::runtime_error("PH secondary must be .json");

            fast_engine_backward_.load_secondary_from_json_file(path_secondary_model);
            path_secondary_model_ = path_secondary_model;
        }
        return;
    }

    throw std::runtime_error("provide .json model paths for single calls");
}
#endif

void NNPWS::setPT(double p, double T) {
    if (p_ != p || T_ != T || !valid_) {
        p_ = p;
        T_ = T;
        calculateG_derivatives();
    }
}

void NNPWS::setPH(double p, double h) {
    if (p_ != p || h_ != h || !valid_) {
        p_ = p;
        h_ = h;
        calculateT();
        setPT(p_, T_);
    }
}

void NNPWS::calculateT() {
    valid_ = false;
    if (!is_initialized_) throw std::runtime_error("Models not set: call setNeuralNetworks()");
    if (inputPr_ != PH) throw std::runtime_error("input pair != PH");

    T_ = fast_engine_backward_.compute_val(p_, h_);
}

void NNPWS::calculateG_derivatives() {
    valid_ = false;
    if (!is_initialized_) throw std::runtime_error("Models not set: call setNeuralNetworks()");

    const Region r = Regions_Boundaries::determine_region(T_, p_);
    if (r == out_of_regions) throw std::runtime_error("TP out of region");

    g_derivatives_ = fast_engine_.compute((int)r, p_, T_);

    if (std::abs(g_derivatives_.dG_dP) < precision_)
        throw std::runtime_error("volume close to 0");

    valid_ = true;
}

void NNPWS::compute_batch_PT(const std::vector<double>& p_list,
                             const std::vector<double>& T_list,
                             std::vector<NNPWS>& results,
                             const std::string& path_main_model)
{
    if (p_list.size() != T_list.size())
        throw std::invalid_argument("compute_batch_PT: P and T size mismatch");

    const size_t n = p_list.size();
    results.clear();
    results.resize(n, NNPWS(Undefined));

    // JSON batch (CPU)
    if (ends_with(path_main_model, ".json")) {
        FastInference eng;
        eng.load_from_json_file(path_main_model, {1,2,3,4,5});

#ifdef NNPWS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < n; ++i) {
            const Region r = Regions_Boundaries::determine_region(T_list[i], p_list[i]);
            if (r == out_of_regions) { results[i].valid_ = false; continue; }

            try {
                results[i].p_ = p_list[i];
                results[i].T_ = T_list[i];
                results[i].g_derivatives_ = eng.compute(r, p_list[i], T_list[i]);
                results[i].valid_ = true;
            } catch (...) {
                results[i].valid_ = false;
            }
        }
        return;
    }

#ifdef NNPWS_WITH_TORCH
    // Torch batch if .pt is provided
    if (!ends_with(path_main_model, ".pt"))
        throw std::runtime_error("compute_batch_PT: expected .json or .pt path");

    if (!ModelLoader::instance().load(path_main_model))
        throw std::runtime_error("Torch model load failed: " + path_main_model);

    auto module_pt = ModelLoader::instance().get_model(path_main_model);
    if (!module_pt) throw std::runtime_error("Torch model null: " + path_main_model);

    torch::Device device = getDevice();
    module_pt->to(device);

    // Group indices by IAPWS region
    std::map<int, std::vector<size_t>> region_indices;
    for (size_t i = 0; i < n; ++i) {
        Region r = Regions_Boundaries::determine_region(T_list[i], p_list[i]);
        if (r == out_of_regions) { results[i].valid_ = false; continue; }
        region_indices[static_cast<int>(r)].push_back(i);
    }

    for (auto const& [reg_id, indices] : region_indices) {
        const size_t n_reg = indices.size();
        if (n_reg == 0) continue;

        std::vector<double> flat_input;
        flat_input.reserve(n_reg * 2);

        for (size_t idx : indices) {
            flat_input.push_back(T_list[idx]); // T
            flat_input.push_back(p_list[idx]); // P
        }

        torch::Tensor input_cpu = torch::tensor(flat_input, torch::dtype(torch::kDouble)).reshape({static_cast<long>(n_reg), 2});
        torch::Tensor input_dev = input_cpu.to(device).detach();
        input_dev.set_requires_grad(true);

        try {
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(input_dev);
            inputs.emplace_back(reg_id);

            torch::Tensor out_dev = module_pt->get_method("compute_derivatives_batch")(inputs).toTensor();
            torch::Tensor out_cpu = out_dev.to(torch::kCPU);

            auto acc = out_cpu.accessor<double, 2>();
            for (size_t k = 0; k < n_reg; ++k) {
                const size_t i0 = indices[k];
                NNPWS& obj = results[i0];

                obj.valid_ = true;
                obj.p_ = p_list[i0];
                obj.T_ = T_list[i0];

                obj.g_derivatives_ = FastResult{
                    acc[k][0], acc[k][2], acc[k][1],
                    acc[k][5], acc[k][3], acc[k][4]
                };
            }
        } catch (const c10::Error& e) {
            std::cerr << "[NNPWS/Torch] batch failed for region " << reg_id << ": " << e.what() << "\n";
            for (size_t idx : indices) results[idx].valid_ = false;
        }
    }
    return;
#else
    throw std::runtime_error("Built without Torch: compute_batch_PT requires JSON path.");
#endif
}

void NNPWS::compute_batch_PH(const std::vector<double>& p_list,
                             const std::vector<double>& h_list,
                             std::vector<NNPWS>& results,
                             const std::string& path_main_model,
                             const std::string& path_secondary_model)
{
    if (p_list.size() != h_list.size())
        throw std::invalid_argument("compute_batch_PH: P and H size mismatch");

    const size_t n = p_list.size();
    results.clear();
    results.resize(n, NNPWS(Undefined));

    // JSON batch: PH -> compute T list then call batch_PT JSON
    if (ends_with(path_main_model, ".json") && ends_with(path_secondary_model, ".json")) {
        FastInference ph;
        ph.load_secondary_from_json_file(path_secondary_model);

        std::vector<double> T_list(n);

#ifdef NNPWS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < n; ++i) {
            T_list[i] = ph.compute_val(p_list[i], h_list[i]);
        }

        compute_batch_PT(p_list, T_list, results, path_main_model);
        return;
    }

#ifdef NNPWS_WITH_TORCH

    if (!ends_with(path_main_model, ".pt") || !ends_with(path_secondary_model, ".pt"))
        throw std::runtime_error("compute_batch_PH: expected (.json,.json) or (.pt,.pt)");

    if (!ModelLoader::instance().load(path_secondary_model))
        throw std::runtime_error("Torch PH load failed: " + path_secondary_model);

    auto model_ph = ModelLoader::instance().get_model(path_secondary_model);
    if (!model_ph) throw std::runtime_error("Torch PH model null");

    torch::Device device = getDevice();
    model_ph->to(device);

    torch::Tensor input_cpu = torch::empty({(long)n, 2}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU));
    auto input_acc = input_cpu.accessor<double, 2>();
    for (size_t i = 0; i < n; ++i) {
        input_acc[i][0] = h_list[i]; // H
        input_acc[i][1] = p_list[i]; // P
    }

    torch::Tensor input_dev = input_cpu.to(device);
    torch::Tensor out_dev;
    {
        torch::NoGradGuard ng;
        std::vector<torch::jit::IValue> in;
        in.emplace_back(input_dev);
        out_dev = model_ph->forward(in).toTensor();
    }

    torch::Tensor out_cpu = out_dev.to(torch::kCPU);
    std::vector<double> T_list(n);
    if (out_cpu.dim() == 2) {
        auto out_acc = out_cpu.accessor<double, 2>();
        for (size_t i = 0; i < n; ++i) T_list[i] = out_acc[i][0];
    } else {
        auto out_acc = out_cpu.accessor<double, 1>();
        for (size_t i = 0; i < n; ++i) T_list[i] = out_acc[i];
    }

    compute_batch_PT(p_list, T_list, results, path_main_model);
    return;
#else
    throw std::runtime_error("Built without Torch: compute_batch_PH requires (.json,.json).");
#endif
}
