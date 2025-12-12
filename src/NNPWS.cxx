#include "NNPWS.hxx"
#include <iostream>
#include <map>
#include <stdexcept>

FastInference NNPWS::fast_engine_;
std::shared_ptr<torch::jit::script::Module> NNPWS::module_pt_ = nullptr;
bool NNPWS::is_initialized_ = false;

NNPWS::NNPWS() : valid_(false), path_model_pt_("ressources/"); {}

NNPWS::NNPWS(double p, double T), path_model_pt_("ressources/") {
    setPT(p, T);
}

NNPWS(const std::string& path_model_pt, const std::string& path_model_ph)
{
	setNNPath( path_model_pt, path_model_ph);
}

NNPWS::~NNPWS() = default;

int NNPWS::init(const std::string& path_model_pt, const std::string& path_model_ph) {
    if (is_initialized_) return 0;

    if (!ModelLoader::instance().load(path_model_pt)) {

        std::cerr << "[NNPWS] Erreur chargement modele PT." << std::endl;
        return -1;
    }
    module_pt_ = ModelLoader::instance().get_model(path_model_pt);

    try {
        std::vector<int> regions = {1, 2, 3, 4, 5};
        fast_engine_.load_from_module(module_pt_, regions);
        is_initialized_ = true;

    } catch (const std::exception& e) {

        std::cerr << "[NNPWS] Erreur init FastInference: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

void NNPWS::setPT(double p, double T) {
	
	if( this->p_ != p || this->T_ != T )
    {
        this->p_ = p;
        this->T_ = T;
        this->calculate();
    }
}

void setNNPath(const std::string& path_model_pt, const std::string& path_model_ph)
{
	path_model_pt_ = path_model_pt;
	path_model_ph_ = path_model_ph;
}

void NNPWS::calculate() {
    valid_ = false;
    G_ = 0; S_ = 0; V_ = 0; Rho_ = 0; Cp_ = 0; Kappa_ = 0;

    if (!is_initialized_) return;

    Region r = Regions_Boundaries::determine_region(T_, p_);
    if (r == out_of_regions) return;

    FastResult res = fast_engine_.compute((int)r, p_, T_);

    G_ = res.G;
    S_ = -res.dG_dT;

    const double vol = res.dG_dP * 1e-3; //G[kJ/kg], P[MPa] -> V[m3/kg]
    V_ = vol;

    if (std::abs(vol) > 1e-12) {
        Rho_ = 1.0 / vol;
        double dV_dP = res.d2G_dP2 * 1e-3;
        Kappa_ = -(1.0 / vol) * dV_dP;
    }

    Cp_ = -T_ * res.d2G_dT2;
    valid_ = true;
}

void NNPWS::compute_batch(const std::vector<double>& p_list,
                          const std::vector<double>& T_list,
                          std::vector<NNPWS>& results) {

    if (!is_initialized_ || !module_pt_) {
        std::cerr << "[NNPWS] Erreur Critique : Le modèle n'est pas initialisé. Appelez NNPWS::init() d'abord." << std::endl;
        throw std::runtime_error("Modèle non initialisé");
    }

    if (p_list.size() != T_list.size()) {
        std::string err_msg = "[NNPWS] Erreur de dimension : p_list (" + std::to_string(p_list.size()) +
                              ") et T_list (" + std::to_string(T_list.size()) + ") doivent avoir la même taille.";
        std::cerr << err_msg << std::endl;
        throw std::invalid_argument(err_msg);
    }

    size_t n = p_list.size();
    results.clear();
    results.resize(n);

    std::map<int, std::vector<size_t>> region_indices;

    for (size_t i = 0; i < n; ++i) {
        Region r = Regions_Boundaries::determine_region(T_list[i], p_list[i]);
        if (r == out_of_regions) {
            results[i].valid_ = false;
            continue;
        }
        region_indices[(int)r].push_back(i);
    }

    for (auto const& [reg_id, indices] : region_indices) {
        size_t n_reg = indices.size();
        if (n_reg == 0) continue;

        std::vector<double> flat_input;
        flat_input.reserve(n_reg * 2);

        for (size_t idx : indices) {
            flat_input.push_back(T_list[idx]); // T en colonne 0
            flat_input.push_back(p_list[idx]); // P en colonne 1
        }

        torch::Tensor input_tensor = torch::from_blob(flat_input.data(), {(long)n_reg, 2}, torch::kDouble).clone();

        try {
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(input_tensor);
            inputs.emplace_back(reg_id);

            torch::Tensor output = module_pt_->get_method("compute_derivatives_batch")(inputs).toTensor();
            auto acc = output.accessor<double, 2>();

            for (size_t k = 0; k < n_reg; ++k) {
                size_t original_idx = indices[k];
                NNPWS& obj = results[original_idx];

                obj.p_ = p_list[original_idx];
                obj.T_ = T_list[original_idx];

                double res_G     = acc[k][0];
                double res_G_T   = acc[k][1];
                double res_G_P   = acc[k][2];
                double res_G_TT  = acc[k][3];
                double res_G_PP  = acc[k][5];

                obj.G_ = res_G;
                obj.S_ = -res_G_T;

                double vol = res_G_P * 1e-3;
                obj.V_ = vol;

                if (std::abs(vol) > 1e-12) {
                    obj.Rho_ = 1.0 / vol;
                    obj.Kappa_ = -(1.0 / vol) * (res_G_PP * 1e-3);
                } else {
                    obj.Rho_ = 0; obj.Kappa_ = 0;
                }

                obj.Cp_ = -obj.T_ * res_G_TT;
                obj.valid_ = true;
            }

        } catch (const c10::Error& e) {
            std::cerr << "[NNPWS] Erreur Batch LibTorch (Region " << reg_id << "): " << e.what() << std::endl;
            // throw std::runtime_error("Erreur fatale dans le modèle Torch");
        }
    }
}
