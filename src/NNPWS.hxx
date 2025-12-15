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
    /* Default constructor. Does not load the NN model. Used to save memory. */
    NNPWS();
    /* Full data constructor. Loads the NN model in memory and computes fluid properties. */
    NNPWS(inputPair varnames, double p, double T, const std::string& path_main_model_pt="ressources/PT.jit", const std::string& path_secondary_model = "");
    /* Half data constructor. Loads the NN model in memory and wait for values of p and T to compute fluid properties. */
    NNPWS(const std::string& path_main_model_pt="ressources/PT.jit", const std::string& path_secondary_model = "");
    ~NNPWS();

    int setNeuralNetworks(const std::string& path_main_model_pt="ressources/PT.jit", const std::string& path_secondary_model = "");//memory allocations are performed here. Called by most constructors

    void setPT(double p, double T);
    //void setPH(double p, double h);    //computes        T_, then call setPT(p_,T_)
    //void setRhoE(double rho, double e);//computes p_ and T_, then call setPT(p_,T_)

    /* Pressure in MPa */
    double getPressure() const { if (valid_) return p_; else throw exception; }     // MPa
    /* Temperature in Kelvin */
    double getTemperature() const { if (valid_) return T_; else throw exception; }  // K
    /* Gibbs Free energy in KJ/Kg */
    double getGibbs() const { if (valid_) return g_derivatives_.G; else throw exception; }        // kJ/kg
    /* Entropy in kJ/(kg.K) */
    double getEntropy() const { if (valid_) return -g_derivatives_.dG_dT; else throw exception; }      // kJ/(kg.K)
    /* Volume in m3/kg */
    double getVolume() const { if (valid_) return g_derivatives_.dG_dP * 1e-3; else throw exception; }       // m3/kg
    /* Density in kg/m3 */
    double getDensity() const { if (valid_) return 1/getVolume(); else throw exception; }    // kg/m3  // volume is non zero because of earlier check in function  calculate
    /* Enthalpy in kJ/kg */
    double getEnthalpy() const { if (valid_) return g_derivatives_.G - T_*g_derivatives_.dG_dT; else throw exception; }      // kJ/(kg.K)
    /* Internal Energy in kJ/kg */
    double getInternalEnergy() const { if (valid_) return getEnthalpy() - P_*getVolume(); else throw exception; }      // kJ/(kg.K)
    /* Isobaric heat capacity in kJ/(kg.K) */
    double getCp() const { if (valid_) return -T_ * g_derivatives_.d2G_dT2; else throw exception; }          // kJ/(kg.K)
    //To do : getCrho() et getSoundSpeed()
    
    double getdV_dP() const { if (valid_) return g_derivatives_.d2G_dP2 * 1e-3; else throw exception; }
    /* Compressibilité isotherme in 1/MPa */
    double getCompressibiliteIsotherme() const { if (valid_) return -getDensity() * getdV_dP(); else throw exception; }    // 1/MPa
    double getKappa() const { if (valid_) return -getDensity() * getdV_dP(); else throw exception; }    // 1/MPa
    
    bool isValid() const { return valid_; }

    /* Load the model and compute many (P,T) pairs at once. output contain g derivatives and can be used to retrieve other fluid properties. Model is not loaded in order to save memory. */
    static void compute_batch(const std::vector<double>& p_list, 
                              const std::vector<double>& T_list, 
                              std::vector<NNPWS>& results,
                              const std::string& path_main_model_pt="ressources/PT.jit",
                              const std::string& path_secondary_model);

private:
    FastInference fast_engine_; //For single calculations
    FastResult g_derivatives_;
    std::shared_ptr<torch::jit::script::Module> module_pt_;//for batch or gpu calculations
    bool is_initialized_;

    double p_ = 0.0;
    double T_ = 0.0;
    
    std::string& path_main_model_pt_  = "";//Path to the main neural network, the one that computes g and its derivatives from P and T
    std::string& path_secondary_model_= "";//Path to the secondary neural network, the one that computes (P,T) from (P,h) or (rho, e) depending on the enum inputPair

    bool valid_ = false;
    double precision_ = 1e-12;

    void calculateG_derivatives();
};

#endif // NNPWS_NNPWS_HXX
