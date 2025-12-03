#include "NNPWS.hxx"

NNPWS::NNPWS() {
}

NNPWS::NNPWS(const NNPWS &) {
}

NNPWS::~NNPWS() {
}

int NNPWS::init() {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_Ph(const char * const property_name, double in1, double in2) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_PT(const char * const property_name, double in1, double in2) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_Ph(const char * const property_name, int property_number, double in1, double in2) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_PT(const char * const property_name, int property_number, double in1, double in2) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_T_sat_p(double p) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_p_sat_T(double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_rho_l_sat_p(double p) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_rho_v_sat_p(double p) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_rho_l_sat_T(double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_rho_v_sat_T(double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_h_l_sat_p(double p) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_h_v_sat_p(double p) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_cp_l_sat_p(double p) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_cp_v_sat_p(double p) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_cp_l_sat_T(double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_cp_v_sat_T(double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_h_l_sat_T(double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_h_v_sat_T(double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_pr_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_h_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_T_ph(double p, double h) const {
    if (!is_initialized_ph()) return -1.0;
    const auto& results = run_inference_ph(p, h);

    return results[0];
}

double NNPWS::compute_rho_pT(double p, double T) const {
    if (!is_initialized_pt()) return -1.0;

    const auto& results = run_inference_pt(p, T);

    double v = results[IDX_G_DP] * 1e-3;

    // Rho = 1 / v
    if (v != 0.0) return 1.0 / v;
    return 0.0;
}

double NNPWS::compute_rho_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_u_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_u_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_s_pT(double p, double T) const {

    if (!is_initialized_pt()) return -1.0;
    const auto& results = run_inference_pt(p, T);
    return -results[IDX_G_DT];
}

double NNPWS::compute_s_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_mu_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_mu_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_lambda_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_lambda_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_cp_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_cp_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_cv_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_cv_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_sigma_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_sigma_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_w_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_w_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_g_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_g_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_f_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_f_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_beta_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_beta_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_gamma_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_gamma_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_pr_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_T_min() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_T_max() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_p_max() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_p_min() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_h_min() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_h_max() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_T_crit() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_p_crit() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_h_crit() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::get_mm() const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_d_T_d_p_h_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_d_T_d_h_p_ph(double p, double h) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_d_h_d_p_T_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_d_h_d_T_p_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
}

Region NNPWS::determine_region(const double T, const double P) {
    return Regions_Boundaries::determine_region(T, P);
}


int NNPWS::init(const std::string& path_pt, const std::string& path_ph) {

    bool pt_loaded = ModelLoader::instance().load(path_pt);
    bool ph_loaded = ModelLoader::instance().load(path_ph);

    if (!pt_loaded) {
        std::cerr << "NNPWS: Échec du chargement du modèle PT via ModelLoader." << std::endl;
        return -1;
    }

    if (ph_loaded) {
        has_ph_model_ = true;

        this->module_ph_ = ModelLoader::instance().get_model(path_ph);
        if (!module_ph_) {
            std::cerr << "NNPWS: Erreur lors de la récupération des pointeurs de modèles." << std::endl;
            return -1;
        }

        cached_results_ph_.resize(IDX_COUNT, 0.0);
    }
    else if (!path_ph.empty()) {
        std::cerr << "NNPWS: Échec du chargement de modèle PH via ModelLoader." << std::endl;
        return -1;
    }

    this->module_pt_ = ModelLoader::instance().get_model(path_pt);

    if (!module_pt_) {
        std::cerr << "NNPWS: Erreur lors de la récupération des pointeurs de modèles." << std::endl;
        return -1;
    }

    cached_results_pt_.resize(IDX_COUNT, 0.0);

    is_initialized_ = true;
    return 0;
}


const std::vector<double>& NNPWS::run_inference_pt(double p, double T) const {
    if (p == cache_p_pt_ && T == cache_T_pt_) {
        return cached_results_pt_;
    }

    //torch::NoGradGuard no_grad;

    int region = Regions_Boundaries::determine_region(T, p);

    torch::Tensor inputs = torch::tensor({{T, p}}, torch::kDouble);

    auto result = module_pt_->get_method("compute_derivatives_batch")({inputs, region});

    torch::Tensor out_tensor = result.toTensor();

    auto data_ptr = out_tensor.accessor<double, 2>();
    for(int i=0; i < IDX_COUNT; ++i) {
        cached_results_pt_[i] = data_ptr[0][i];
    }

    cache_p_pt_ = p;
    cache_T_pt_ = T;

    return cached_results_pt_;
}

const std::vector<double>& NNPWS::run_inference_ph(double p, double h) const {
    if (p == cache_p_ph_ && h == cache_h_ph_) {
        return cached_results_ph_;
    }
    
    cache_p_ph_ = p;
    cache_h_ph_ = h;

    return cached_results_ph_;
}

bool NNPWS::is_initialized_pt() const {
    if (!is_initialized_) {
        std::cerr << "NNPWS: classe non initialisé avec init()" << std::endl;
    }

    return is_initialized_;
}

bool NNPWS::is_initialized_ph() const {
    if (!is_initialized_pt()) {
        return false;
    }

    if (!has_ph_model_) {
        std::cerr << "NNPWS: modèle PH non initialisé avec init()" << std::endl;
    }

    return has_ph_model_;
}