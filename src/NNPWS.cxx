#include "NNPWS.hxx"
#include <iostream>
#include <vector>
#include <torch/script.h>
#include "ModelLoader.hxx"

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
    throw std::runtime_error("Not implemented yet.");
}

double NNPWS::compute_rho_pT(double p, double T) const {
    throw std::runtime_error("Not implemented yet.");
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
    throw std::runtime_error("Not implemented yet.");
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
    return r1;
}

double NNPWS::compute_gibbs(double T, double P) const {
    auto model = ModelLoader::instance().get_model();

    if (!model) {
        std::cerr << "[GibbsPredictor] Erreur : Modèle non chargé ! Appelez load() au début." << std::endl;
        return 0.0;
    }

    int region_id = determine_region(T, P);

    torch::NoGradGuard no_grad;

    try {
        torch::Tensor inputs = torch::tensor({{T, P}}, at::kDouble);

        std::vector<torch::jit::IValue> args;
        args.emplace_back(inputs);
        args.emplace_back(static_cast<int64_t>(region_id));

        at::Tensor output = model->forward(args).toTensor();

        try {
            at::Tensor out = output.toType(at::kDouble).cpu();

            if (!out.defined() || out.numel() == 0) {
                throw std::runtime_error("sortie du modèle vide ou indéfinie");
            }

            at::Tensor flat = out.reshape({-1});
            return flat.index({0}).item<double>();
        } catch (const std::exception& ie) {
            std::ostringstream oss;
            oss << "extraction du scalaire échouée: " << ie.what();
            throw std::runtime_error(oss.str());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[GibbsPredictor] Erreur calcul : " << e.what() << std::endl;
        return 0.0;
    }
}