#ifndef NNPWS_NNPWS_HXX
#define NNPWS_NNPWS_HXX

#include <vector>
#include <string>
#include <memory>
#include <torch/script.h>
#include "FastInference.hxx"
#include "Regions.hxx"
#include "ModelLoader.hxx"

enum inputPair { PT, PH, RhoE, Undefined };

class NNPWS {
public:
    /* Basic constructor. Does not load the NN model. Used to save memory. */
    NNPWS();
    /* Full data constructor. Loads the NN model in memory and computes g derivatives. */
    NNPWS(inputPair varnames, double first, double second, const std::string& path_main_model_pt, const std::string& path_secondary_model);
    /* Half-way data constructor. Loads the NN model in memory and wait for values of p and T to compute g derivatives. */
    NNPWS(const std::string& path_main_model_pt, const std::string& path_secondary_model);
    ~NNPWS();

    /* Memory allocations are performed here. Called by most constructors */
    void setNeuralNetworks(const std::string& path_main_model_pt, const std::string& path_secondary_model);

    /* Set P and T and calculate g derivatives */
    void setPT(double p, double T);
    //void setPH(double p, double h);    //computes        T_, then call setPT(p_,T_)
    //void setRhoE(double rho, double e);//computes p_ and T_, then call setPT(p_,T_)
    void setInputPair( inputPair inputPr){ inputPr_=inputPr; }
    inputPair getInputPair( ){ return inputPr_ ; }

    /* Pressure in MPa */
    double getPressure() const { if (valid_) return p_;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // MPa

    /* Temperature in Kelvin */
    double getTemperature() const { if (valid_) return T_;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // K

    /* Gibbs Free energy in KJ/Kg */
    double getGibbs() const { if (valid_) return g_derivatives_.G;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/kg

    /* Entropy in kJ/(kg.K) */
    double getEntropy() const { if (valid_) return -g_derivatives_.dG_dT;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    /* Volume in m3/kg */
    double getVolume() const { if (valid_) return g_derivatives_.dG_dP * 1e-3;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // m3/kg

    /* Density in kg/m3 */
    double getDensity() const { if (valid_) return 1/getVolume();
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kg/m3 volume is non zero because of earlier check in function calculate

    /* Enthalpy in kJ/kg */
    double getEnthalpy() const { if (valid_) return g_derivatives_.G - T_*g_derivatives_.dG_dT;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    /* Internal Energy in kJ/kg */
    double getInternalEnergy() const { if (valid_) return getEnthalpy() - p_ * getVolume();
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    /* Isobaric heat capacity in kJ/(kg.K) */
    double getCp() const { if (valid_) return -T_ * g_derivatives_.d2G_dT2;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    //To do : getCrho() et getSoundSpeed()
    
    double getdV_dP() const { if (valid_) return g_derivatives_.d2G_dP2 * 1e-3;
        throw std::runtime_error("variable PT not set use setPT() first"); }

    /* Compressibilité isotherme in 1/MPa */
    double getCompressibiliteIsotherme() const { if (valid_) return -getDensity() * getdV_dP();
        throw std::runtime_error("variable PT not set use setPT() first"); }    // 1/MPa

    double getKappa() const { if (valid_) return -getDensity() * getdV_dP() * 1e-3;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // 1/MPa
    
    bool isValid() const { return valid_; }

    /* Load the model and compute many (P,T) pairs at once. output contain g derivatives and can be used to retrieve other fluid properties. Model is not loaded in order to save memory. */
    static void compute_batch_PT(const std::vector<double>& p_list,
                              const std::vector<double>& T_list, 
                              std::vector<NNPWS>& results,
                              const std::string& path_main_model_pt);

private:
    FastInference fast_engine_; //For single calculations
    FastResult g_derivatives_;
    std::shared_ptr<torch::jit::script::Module> module_pt_; //for batch or gpu calculations
    bool is_initialized_ = false;

    double p_ = 0.0;
    double T_ = 0.0;
    inputPair inputPr_ = Undefined;

    //Path to the main neural network, the one that computes g and its derivatives from P and T
    std::string path_main_model_pt_ = "resources/models/DNN_TP_v6.pt";
    //Path to the secondary neural network, the one that computes (P,T) from (P,h) or (rho, e) depending on the enum inputPair
    std::string path_secondary_model_;

    bool valid_ = false;
    double precision_ = 1e-12;

    void calculateG_derivatives();
};

#endif // NNPWS_NNPWS_HXX
