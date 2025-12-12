#include "NNPWS.hxx"
#include <iostream>
#include <map>
#include <stdexcept>


NNPWS::NNPWS() : valid_(false), path_model_pt_("ressources/PT.jit"); {}

NNPWS::NNPWS(inputPair varnames, double var1, double var2, const std::string& path_model_pt, const std::string& path_model_ph ) : valid_(false) {
    setNNPath( path_model_pt, path_model_ph);
    
    switch(varnames)
    {
        case PT   : setPT(  var1, var2); break;
        //case PH   : setPH(  var1, var2); break;
        //case RhoE : setRhoE(var1, var2); break;
        default : throw exception not yet implemented
    }
        
}

NNPWS(const std::string& path_model_pt, const std::string& path_model_ph) ) : valid_(false) 
{
	setNNPath( path_model_pt, path_model_ph);
}

NNPWS::~NNPWS() = default;

int NNPWS::init(const std::string& path_model_pt, const std::string& path_model_ph) {
    if (is_initialized_) return 0;

    if (!ModelLoader::instance().load(path_model_pt)) {

        std::cerr << "[NNPWS] Erreur chargement modele PT." << std::endl;
        return -1;//throw exception
    }
    module_pt_ = ModelLoader::instance().get_model(path_model_pt);

    try {
        std::vector<int> regions = {1, 2, 3, 4, 5};
        fast_engine_.load_from_module(module_pt_, regions);
        is_initialized_ = true;

    } catch (const std::exception& e) {

        std::cerr << "[NNPWS] Erreur init FastInference: " << e.what() << std::endl;
        return -1;//throw exception
    }

    return 0;
}

void NNPWS::setPT(double p, double T) {
	
	if( this->p_ != p || this->T_ != T )
    {
        this->p_ = p;
        this->T_ = T;
        this->calculate();
    }
}

void setNNPath(const std::string& path_model_pt, const std::string& path_model_ph)
{
	path_model_pt_ = path_model_pt;
	path_model_ph_ = path_model_ph;
}

void NNPWS::calculate() {
    valid_ = false;

    if (!is_initialized_) return;//throw exception

    Region r = Regions_Boundaries::determine_region(T_, p_);
    if (r == out_of_regions) return;//throw exception

    g_derivatives_ = fast_engine_.compute((int)r, p_, T_);
 
    /* Check volume is non zero */
    //if (std::abs(vol) < precision_) 
    //throw exception

    valid_ = true;
}

void NNPWS::compute_batch(const std::vector<double>& p_list,
                          const std::vector<double>& T_list,
                          std::vector<NNPWS>& results,
                          const std::string& path_model_pt) {

    if (!is_initialized_ || !module_pt_) {
        std::cerr << "[NNPWS] Erreur Critique : Le modèle n'est pas initialisé. Appelez NNPWS::init() d'abord." << std::endl;
        throw std::runtime_error("Modèle non initialisé");
    }

    if (p_list.size() != T_list.size()) {
        std::string err_msg = "[NNPWS] Erreur de dimension : p_list (" + std::to_string(p_list.size()) +
                              ") et T_list (" + std::to_string(T_list.size()) + ") doivent avoir la même taille.";
        std::cerr << err_msg << std::endl;
        throw std::invalid_argument(err_msg);
    }

    size_t n = p_list.size();
    results.clear();
    results.resize(n);

    std::map<int, std::vector<size_t>> region_indices;
    Region r;
    double vol;
    
    for (size_t i = 0; i < n; ++i) {
        r = Regions_Boundaries::determine_region(T_list[i], p_list[i]);
        if (r == out_of_regions) {
            results[i].valid_ = false;
            continue;
        }
        region_indices[(int)r].push_back(i);
    }

    for (auto const& [reg_id, indices] : region_indices) {
        size_t n_reg = indices.size();
        if (n_reg == 0) continue;

        std::vector<double> flat_input;
        flat_input.reserve(n_reg * 2);

        for (size_t idx : indices) {
            flat_input.push_back(T_list[idx]); // T en colonne 0
            flat_input.push_back(p_list[idx]); // P en colonne 1
        }

        torch::Tensor input_tensor = torch::from_blob(flat_input.data(), {(long)n_reg, 2}, torch::kDouble).clone();

        try {
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(input_tensor);
            inputs.emplace_back(reg_id);

            torch::Tensor output = module_pt_->get_method("compute_derivatives_batch")(inputs).toTensor();
            auto acc = output.accessor<double, 2>();

            for (size_t k = 0; k < n_reg; ++k) {
                size_t original_idx = indices[k];
                NNPWS& obj = results[original_idx];

                obj.p_ = p_list[original_idx];
                obj.T_ = T_list[original_idx];
                
                obj.g_derivatives = FastResult(acc[k][0],acc[k][1],acc[k][2],acc[k][3],acc[k][4],acc[k][5])

                vol = obj.getVolume() * 1e-3;//now check if non zero

                /* check volume is non zero */
                //if (std::abs(vol) < precision_) 
                    //throw exception
                obj.valid_ = true;
            }

        } catch (const c10::Error& e) {
            std::cerr << "[NNPWS] Erreur Batch LibTorch (Region " << reg_id << "): " << e.what() << std::endl;
            // throw std::runtime_error("Erreur fatale dans le modèle Torch");
        }
    }
}
