#ifndef NNPWS_NNPWS_HXX
#define NNPWS_NNPWS_HXX

#include <vector>
#include <string>
#include <memory>
#include <torch/script.h>
#include <torch/cuda.h>
#include <optional>
#include "FastInference.hxx"
#include "Regions.hxx"
#include "ModelLoader.hxx"

enum inputPair { PT, PH, RhoE, Undefined };

class NNPWS {
public:
    /** \fn NNPWS
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
    NNPWS(inputPair varnames, double first, double second, const std::string& path_main_model_pt, const std::optional<const std::string>& path_secondary_model);
    /** \fn NNPWS
     * \brief Half-way data constructor
     * \details Loads the NN model in memory and wait for values of p and T to compute g derivatives
     * \param [in] string, path to the main neural network file
     * \param [in] string, (optional) path to the auxilliary neural network file used to convert P-H or RhoE to P-T
     *  */
    NNPWS(inputPair varnames, const std::string& path_main_model_pt, const std::optional<const std::string>& path_secondary_model);
    ~NNPWS();

    /** \fn setNeuralNetworks
     * \brief set the neural network file. Load the Neural Networks models
     * \details Memory allocations are performed here. Called by most constructors
     * \param [in] string, path to the main neural network file
     * \param [in] string, (optional) path to the auxilliary neural network file used to convert P-H or RhoE to P-T
     * \param [out] void
     *  */
    void setNeuralNetworks(const std::string &path_main_model_pt, const std::optional<const std::string> &path_secondary_model);

    /** \fn setPT
     * \brief  Set Pressure and Temperature and calculate g derivatives
     * \details Path to the neural network file should be set before, otherwise exception raised
     * \param [in] double, pressure  in MPa
     * \param [in] double, temperature in Kelvin
     *  */
    void setPT(double p, double T);
    /** \fn setPH
     * \brief  Set Pressure and Enthalpy and calculate g derivatives
     * \details Path to the neural network file should be set before, otherwise exception raised
     * \param [in] double, pressure in MPa
     * \param [in] double, enthalpy in kJ/Kg
     *  */
    void setPH(double p, double h);
    /** \fn setRhoE
     * \brief  Set Density and Internal energy and calculate g derivatives
     * \details Path to the neural network file should be set before, otherwise exception raised
     * \param [in] double, density in  kg/m3
     * \param [in] double, internal energy in kJ/Kg
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
     * \param [out] double, the pressure in MPa
     *  */
    double getPressure() const { if (valid_) return p_;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // MPa

    /** \fn getTemperature
     * \brief  get the temperature (in Kelvin) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the temperature in Kelvin
     *  */
    double getTemperature() const { if (valid_) return T_;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // K

    /** \fn getGibbs
     * \brief  get the Gibbs free energy (in kJ/Kg) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the Gibbs free energy in kJ/Kg
     *  */
    double getGibbs() const { if (valid_) return g_derivatives_.G;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/kg

    /** \fn getEntropy
     * \brief  get the Entropy (in kJ/(kg.K)) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the entropy in kJ/(kg.K)
     *  */
    double getEntropy() const { if (valid_) return -g_derivatives_.dG_dT;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kJ/(kg.K)

    /** \fn getVolume
     * \brief  get the Volume (in m3/kg) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the volume in m3/kg
     *  */
    double getVolume() const { if (valid_) return g_derivatives_.dG_dP * 1e-3;
        throw std::runtime_error("variable PT not set use setPT() first"); }    // m3/kg

    /** \fn getDensity
     * \brief  get the Density (in kg/m3) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the density in kg/m3
     *  */
    double getDensity() const { if (valid_) return 1/getVolume();
        throw std::runtime_error("variable PT not set use setPT() first"); }    // kg/m3 volume is non zero because of earlier check in function calculate

    /** \fn getEnthalpy
     * \brief  get the Enthalpy (in kJ/kg) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the enthalpy in kJ/kg
     *  */
    double getEnthalpy() const { if (valid_) return g_derivatives_.G - T_*g_derivatives_.dG_dT;
        throw std::runtime_error("variable PT not set use setPT() first"); } 

    /** \fn getInternalEnergy
     * \brief  get the Internal Energy (in kJ/kg) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the internal energy in kJ/kg
     *  */
    double getInternalEnergy() const { if (valid_) return getEnthalpy() - p_ * getVolume();
        throw std::runtime_error("variable PT not set use setPT() first"); }

    /** \fn getCp
     * \brief  get the Isobaric heat capacity (in kJ/(kg.K)) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, isobaric heat capacity in kJ/(kg.K)
     *  */
    double getCp() const { if (valid_) return -T_ * g_derivatives_.d2G_dT2;
        throw std::runtime_error("variable PT not set use setPT() first"); }

    /** \fn getCrho
     * \brief  get the Isochoric heat capacity (in kJ/(kg.K)) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, Isochoric heat capacity in kJ/(kg.K)
     *  */
    //To do : getCrho()

    /** \fn getSoundSpeed
     * \brief  get the sound speed (in m/s) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, the sound speed in m/s
     *  */
    //To do getSoundSpeed()
    
    double getdV_dP() const { if (valid_) return g_derivatives_.d2G_dP2 * 1e-3;
        throw std::runtime_error("variable PT not set use setPT() first"); }

    /** \fn getCompressibiliteIsotherme
     * \brief  get the Isothermal Compressibility (in 1/MPa) associated to the fluid state
     * \details Should be called after loading of model and setting of input pair value
     * \param [out] double, isothermal compressibility in 1/MPa
     *  */
    double getIsothermalCompressibility() const { if (valid_) return -getDensity() * getdV_dP();
        throw std::runtime_error("variable PT not set use setPT() first"); }    // 1/MPa

    double getKappa() const { if (valid_) return -getDensity() * getdV_dP();
        throw std::runtime_error("variable PT not set use setPT() first"); }    // 1/MPa

    static void setUseGPU(bool use_gpu) {
        if (use_gpu && torch::cuda::is_available()) {
            target_device_ = torch::kCUDA;
            std::cout << "[NNPWS] GPU active (CUDA)" << std::endl;
        } else {
            target_device_ = torch::kCPU;
            std::cout << "[NNPWS] CPU active" << std::endl;
        }
    }

    static torch::Device getDevice() { return target_device_; }
    
    bool isValid() const { return valid_; }

    /* Load the model and compute many (P,T) pairs at once. output contain g derivatives and can be used to retrieve other fluid properties. Model is not loaded in order to save memory. */
    static void compute_batch_PT(const std::vector<double>& p_list,
                              const std::vector<double>& T_list, 
                              std::vector<NNPWS>& results,
                              const std::string& path_main_model_pt);

    static void compute_batch_PH(const std::vector<double>& p_list,
                                 const std::vector<double>& h_list,
                                 std::vector<NNPWS>& results,
                                 const std::string& path_main_model_pt,
                                 const std::string& path_secondary_model);

private:
    FastInference fast_engine_; //For single calculations
    FastInference fast_engine_backward_;
    FastResult g_derivatives_;
    //std::shared_ptr<torch::jit::script::Module> module_pt_; //for batch or gpu calculations
    bool is_initialized_ = false;
    static inline torch::Device target_device_ = torch::kCPU;

    double h_ = 0.0;
    double p_ = 0.0;
    double T_ = 0.0;
    inputPair inputPr_ = Undefined;

    //Path to the main neural network, the one that computes g and its derivatives from P and T
    std::string path_main_model_pt_ = "../resources/models/DNN_TP_v8.pt";
    //Path to the secondary neural network, the one that computes (P,T) from (P,h) or (rho, e) depending on the enum inputPair
    std::string path_secondary_model_ = "../resources/models/DNN_Backward_PH_noRegion.pt";

    bool valid_ = false;
    double precision_ = 1e-12;

    void calculateG_derivatives();
    void calculateT();
};

#endif // NNPWS_NNPWS_HXX
