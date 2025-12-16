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
    /** \fn NNPWS()
     * \brief Basic constructor
     * \details Does not load the NN model. Used to save memory
     * \param [in] inputPair
     *  */
    NNPWS(inputPair varnames);
    /** \fn NNPWS
     * \brief Full data constructor
     * \details Loads the NN model in memory and computes g derivatives
     * \param [in] inputPair, enum indicating P-T, P-H or Rho-e
     * \param [in] double, first variable (pressure or density depending on input pair
     * \param [in] double, second variable (temperature, enthalpy or internal energy depending on input pair
     * \param [in] string, path to the main neural network file
     * \param [in] string, (optional) path to the auxilliary neural network file used to convert P-H or RhoE to P-T
     *  */
    NNPWS(inputPair varnames, double first, double second, const std::string& path_main_model_pt, const std::string& path_secondary_model);
    /** \fn NNPWS
     * \brief Half-way data constructor
     * \details Loads the NN model in memory and wait for values of p and T to compute g derivatives
     * \param [in] string, path to the main neural network file
     * \param [in] string, (optional) path to the auxilliary neural network file used to convert P-H or RhoE to P-T
     *  */
    NNPWS(const std::string& path_main_model_pt, const std::string& path_secondary_model);
    ~NNPWS();

    /** \fn setNeuralNetworks
     * \brief set the neural network file. Load the Neural Networks models
     * \details Memory allocations are performed here. Called by most constructors
     * \param [in] string, path to the main neural network file
     * \param [in] string, (optional) path to the auxilliary neural network file used to convert P-H or RhoE to P-T
     * \param [out] void
     *  */
    void setNeuralNetworks(const std::string& path_main_model_pt, const std::string& path_secondary_model);

    /** \fn setPT
     * \brief  Set Pressure and Temperature and calculate g derivatives
     * \details Path to NN should be set before, otherwise exception raised
     * \param [in] double, pressure
     * \param [in] double, temperature
     *  */
    /* Set P and T and calculate g derivatives */
    void setPT(double p, double T);
    /** \fn setPT
     * \brief  Set Pressure and Enthalpy and calculate g derivatives
     * \details Path to NN should be set before, otherwise exception raised
     * \param [in] double, pressure
     * \param [in] double, enthalpy
     *  */
    //void setPH(double p, double h);    //computes        T_, then call setPT(p_,T_)
    /** \fn setRhoE
     * \brief  Set Density and Internal energy and calculate g derivatives
     * \details Path to NN should be set before, otherwise exception raised
     * \param [in] double, density
     * \param [in] double, internal energy
     *  */
    //void setRhoE(double rho, double e);//computes p_ and T_, then call setPT(p_,T_)
    /** \fn setInputPair
     * \brief  Set the enum InputPair to specify input variables
     * \details Input pair can be pressure-density, pressure-enthalpy or Density-Internal energy
     * \param [in] inputPair, the enum corresponding to the input variables
     *  */
    void setInputPair( inputPair inputPr){ inputPr_=inputPr; }
    /** \fn getInputPair
     * \brief  get the enum InputPair that specify input variables
     * \details Input pair can be pressure-density, pressure-enthalpy or Density-Internal energy
     * \param [out] inputPair, the enum corresponding to the input variables
     *  */
    inputPair getInputPair( ){ return inputPr_ ; }

    /** \fn getPressure
     * \brief  get the pressure (in MPa) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the pressure
     *  */
    double getPressure() const { if (valid_) return p_;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // MPa

    /** \fn getTemperature
     * \brief  get the temperature (in Kelvin) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the temperature
     *  */
    double getTemperature() const { if (valid_) return T_;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // K

    /** \fn getGibbs
     * \brief  get the Gibbs free energy (in KJ/Kg) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the Gibbs free energy
     *  */
    double getGibbs() const { if (valid_) return g_derivatives_.G;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/kg

    /** \fn getEntropy
     * \brief  get the Entropy (in kJ/(kg.K)) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the Entropy
     *  */
    double getEntropy() const { if (valid_) return -g_derivatives_.dG_dT;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    /** \fn getVolume
     * \brief  get the Volume (in m3/kg)) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the Volume
     *  */
    double getVolume() const { if (valid_) return g_derivatives_.dG_dP * 1e-3;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // m3/kg

    /** \fn getDensity
     * \brief  get the Density (in kg/m3)) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the Density
     *  */
    double getDensity() const { if (valid_) return 1/getVolume();
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kg/m3 volume is non zero because of earlier check in function calculate

    /** \fn getEnthalpy
     * \brief  get the Enthalpy (in kJ/kg) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the Enthalpy
     *  */
    double getEnthalpy() const { if (valid_) return g_derivatives_.G - T_*g_derivatives_.dG_dT;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    /** \fn getInternalEnergy
     * \brief  get the Internal Energy (in kJ/kg) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the Internal Energy
     *  */
    double getInternalEnergy() const { if (valid_) return getEnthalpy() - p_ * getVolume();
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    /** \fn getCp
     * \brief  get the Isobaric heat capacity (in kJ/(kg.K)) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, Isobaric heat capacity
     *  */
    double getCp() const { if (valid_) return -T_ * g_derivatives_.d2G_dT2;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    /** \fn getCrho
     * \brief  get the Isochoric heat capacity (in kJ/(kg.K)) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, Isochoric heat capacity
     *  */
    //To do : getCrho()

    /** \fn getSoundSpeed
     * \brief  get the sound speed (in m/s) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the sound speed
     *  */
    //To do getSoundSpeed()
    
    double getdV_dP() const { if (valid_) return g_derivatives_.d2G_dP2 * 1e-3;
        throw std::runtime_error("variable PT not set use setPT() first"); }

    /** \fn getCompressibiliteIsotherme
     * \brief  get the Isothermal Compressibility (in 1/MPa) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, Isothermal Compressibility
     *  */
    /* Compressibilité isotherme in 1/MPa */
    double getIsothermalCompressibility() const { if (valid_) return -getDensity() * getdV_dP();
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
