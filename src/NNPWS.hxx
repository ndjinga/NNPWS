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
    ~NNPWS();

    static int init(const std::string& path_model_pt, const std::string& path_model_ph = "");

    void setPT(double p, double T);

    double getPressure() const { return p_; }     // MPa
    double getTemperature() const { return T_; }  // K
    double getGibbs() const { return G_; }        // kJ/kg
    double getEntropy() const { return S_; }      // kJ/(kg.K)
    double getVolume() const { return V_; }       // m3/kg
    double getDensity() const { return Rho_; }    // kg/m3
    double getCp() const { return Cp_; }          // kJ/(kg.K)
    double getKappa() const { return Kappa_; }    // 1/MPa
    bool isValid() const { return valid_; }


    static void compute_batch(const std::vector<double>& p_list, 
                              const std::vector<double>& T_list, 
                              std::vector<NNPWS>& results);

private:
    static FastInference fast_engine_; 
    static std::shared_ptr<torch::jit::script::Module> module_pt_;
    static bool is_initialized_;

    double p_ = 0.0;
    double T_ = 0.0;
    
    double G_ = 0.0;
    double S_ = 0.0;
    double V_ = 0.0;
    double Rho_ = 0.0;
    double Cp_ = 0.0;
    double Kappa_ = 0.0;
    
    bool valid_ = false;

    void calculate();
};

#endif // NNPWS_NNPWS_HXX
