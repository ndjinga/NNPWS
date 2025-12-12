#ifndef NNPWS_NNPWS_HXX
#define NNPWS_NNPWS_HXX

#include <vector>
#include <string>
#include <memory>
#include <torch/script.h>
#include "FastInference.hxx"
#include "Regions.hxx"
#include "ModelLoader.hxx"

enum inputPair { PT, PH, RhoE };

class NNPWS {
public:
    NNPWS();
    NNPWS(inputPair varnames, double p, double T, const std::string& path_model_pt="ressources/PT.jit", const std::string& path_model_ph = "");
    NNPWS(const std::string& path_model_pt="ressources/PT.jit", const std::string& path_model_ph = "");
    ~NNPWS();

    int init(const std::string& path_model_pt="ressources/PT.jit", const std::string& path_model_ph = "");

    void setPT(double p, double T);

    double getPressure() const { return p_; }     // MPa
    double getTemperature() const { return T_; }  // K
    double getGibbs() const { return g_derivatives_.G; }        // kJ/kg
    double getEntropy() const { return -g_derivatives_.dG_dT; }      // kJ/(kg.K)
    double getVolume() const { return g_derivatives_.dG_dP * 1e-3; }       // m3/kg
    double getDensity() const { return 1/getVolume(); }    // kg/m3  // volume is non zero because of earlier check in function  calculate
    double getCp() const { return -T_ * g_derivatives_.d2G_dT2; }          // kJ/(kg.K)
    double getdV_dP() const { return g_derivatives_.d2G_dP2 * 1e-3;}
    double getKappa() const { return -getDensity() * getdV_dP(); }    // 1/MPa
    bool isValid() const { return valid_; }


    static void compute_batch(const std::vector<double>& p_list, 
                              const std::vector<double>& T_list, 
                              std::vector<NNPWS>& results,
                              const std::string& path_model_pt="ressources/PT.jit");

private:
    FastInference fast_engine_; 
    FastResult g_derivatives_;
    std::shared_ptr<torch::jit::script::Module> module_pt_;
    bool is_initialized_;

    double p_ = 0.0;//For P-T and P-H inputs
    double T_ = 0.0;//For P-T inputs
    double h_ = 0.0;//For P-H inputs
    
    double Rho_ = 0.0;
    double Kappa_ = 0.0;
    
    std::string& path_model_pt_ = "ressources/";
    std::string& path_model_ph_;

    bool valid_ = false;
    double precision_ = 1e-12;

    void calculate();
};

#endif // NNPWS_NNPWS_HXX
