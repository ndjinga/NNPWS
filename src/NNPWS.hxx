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
    NNPWS(inputPair varnames, double p, double T, const std::string& path_main_model_pt="ressources/PT.jit", const std::string& path_secondary_model = "");
    NNPWS(const std::string& path_main_model_pt="ressources/PT.jit", const std::string& path_secondary_model = "");
    ~NNPWS();

    int setNeuralNetworks(const std::string& path_main_model_pt="ressources/PT.jit", const std::string& path_secondary_model = "");//memory allocations are performed here. Called by most constructors

    void setPT(double p, double T);
    //void setPH(double p, double h);    //computes        T_, then call setPT(p_,T_)
    //void setRhoE(double rho, double e);//computes p_ and T_, then call setPT(p_,T_)

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
                              const std::string& path_main_model_pt="ressources/PT.jit");

private:
    FastInference fast_engine_; //For single calculations
    FastResult g_derivatives_;
    std::shared_ptr<torch::jit::script::Module> module_pt_;//for batch or gpu calculations
    bool is_initialized_;

    double p_ = 0.0;
    double T_ = 0.0;
    
    double Rho_ = 0.0;
    double Kappa_ = 0.0;
    
    std::string& path_main_model_pt_  = "";//Path to the main neural network, the one that computes g and its derivatives from P and T
    std::string& path_secondary_model_= "";//Path to the secondary neural network, the one that computes (P,T) from (P,h) or (rho, e) depending on the enum inputPair

    bool valid_ = false;
    double precision_ = 1e-12;

    void calculate();
};

#endif // NNPWS_NNPWS_HXX
