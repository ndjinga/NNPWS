#ifndef NNPWS_NNPWS_HXX
#define NNPWS_NNPWS_HXX

#include <vector>
#include <string>
#include <memory>
#include <torch/script.h>
#include "FastInference.hxx"
#include "Regions.hxx"
#include "ModelLoader.hxx"

class NNPWS {
public:
    NNPWS();
    NNPWS(double p, double T);
    NNPWS(const std::string& path_model_pt, const std::string& path_model_ph = "");
    ~NNPWS();

    static int init(const std::string& path_model_pt="ressources/", const std::string& path_model_ph = "");

    void setPT(double p, double T);

    double getPressure() const { return p_; }     // MPa
    double getTemperature() const { return T_; }  // K
    double getGibbs() const { return g_derivatives_.G; }        // kJ/kg
    double getEntropy() const { return -g_derivatives_.dG_dT; }      // kJ/(kg.K)
    double getVolume() const { return g_derivatives_.dG_dP * 1e-3; }       // m3/kg
    double getDensity() const { return Rho_; }    // kg/m3
    double getCp() const { return -T_ * g_derivatives_.d2G_dT2; }          // kJ/(kg.K)
    double getKappa() const { return Kappa_; }    // 1/MPa
    bool isValid() const { return valid_; }


    static void compute_batch(const std::vector<double>& p_list, 
                              const std::vector<double>& T_list, 
                              std::vector<NNPWS>& results);

private:
    static FastInference fast_engine_; 
    FastResult g_derivatives_;
    static std::shared_ptr<torch::jit::script::Module> module_pt_;
    static bool is_initialized_;

    double p_ = 0.0;
    double T_ = 0.0;
    
    double Rho_ = 0.0;
    double Kappa_ = 0.0;
    
    std::string& path_model_pt_ = "ressources/";
    std::string& path_model_ph_;

    bool valid_ = false;
    double precision_ = 1e-12;

    void calculate();
};

#endif // NNPWS_NNPWS_HXX
