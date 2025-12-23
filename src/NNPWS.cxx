#include "NNPWS.hxx"
#include <iostream>
#include <map>
#include <stdexcept>

//This constructor requires explicit call to setNeuralNetworks by the user, unlike the other constructors. Use only if you want to delay the memory allocation
NNPWS::NNPWS(inputPair varnames) : valid_(false), is_initialized_(false), inputPr_(varnames){}

NNPWS::NNPWS(inputPair varnames, double var1, double var2, const std::string& path_main_model_pt, const std::optional<const std::string>& path_secondary_model)
: valid_(false), is_initialized_(false), inputPr_(varnames) {
    setNeuralNetworks(path_main_model_pt, path_secondary_model);
        
    switch(varnames)
    {
        case PT : p_ = var1; T_ = var2; setPT(  p_, T_);   break;
        case PH : setPH(  var1, var2); break;
        //case RhoE : setRhoE(var1, var2); break;
        default : throw std::runtime_error("not implemented");
    }
}

NNPWS::NNPWS(inputPair varnames, const std::string& path_main_model_pt, const std::optional<const std::string>& path_secondary_model) : valid_(false), is_initialized_(false), inputPr_(varnames)
{
	setNeuralNetworks(path_main_model_pt, path_secondary_model);
}

NNPWS::~NNPWS() = default;

void NNPWS::setNeuralNetworks(const std::string& path_main_model_pt, const std::optional<const std::string>& path_secondary_model) {
    if (!ModelLoader::instance().load(path_main_model_pt)) {

        std::cerr << "[NNPWS] Erreur chargement modele PT." << std::endl;
        throw std::runtime_error("impossible de charger le modele pt");
    }
    module_pt_ = ModelLoader::instance().get_model(path_main_model_pt);

    try {
        std::vector<int> regions = {1, 2, 3, 4, 5};
        fast_engine_.load_from_module(module_pt_, regions);
        is_initialized_ = true;

    } catch (const std::exception& e) {

        std::cerr << "[NNPWS] Erreur init FastInference: " << e.what() << std::endl;
        throw std::runtime_error("fichier .pt non valide");
    }

    if (inputPr_ == PH) {
        if (!path_secondary_model.has_value()) throw std::runtime_error("PH input requires a secondary model");

        if (!ModelLoader::instance().load(*path_secondary_model)) {

            std::cerr << "[NNPWS] Erreur chargement modele PH." << std::endl;
            throw std::runtime_error("impossible de charger le modele ph");
        }

        try {
            fast_engine_backward_.load_secondary_from_module(ModelLoader::instance().get_model(*path_secondary_model));
        } catch (const std::exception& e) {

            std::cerr << "[NNPWS] Erreur init FastInferenceBackward: " << e.what() << std::endl;
            throw std::runtime_error("fichier .pt non valide");
        }

        path_secondary_model_ = *path_secondary_model;
    }
	path_main_model_pt_   = path_main_model_pt;
}

void NNPWS::setPT(double p, double T) {
	
	if(this->p_ != p || this->T_ != T || !isValid())
    {
        this->p_ = p;
        this->T_ = T;
        //inputPr_ = PT;
        this->calculateG_derivatives();
    }
}

void NNPWS::setPH(double p, double h)
{
    if(this->p_ != p || this->h_ != h || !isValid())
    {
        this->p_ = p;
        this->h_ = h;
        //inputPr_ = PH;
        this->calculateT();

        setPT(p, this->T_);
    }
}

void NNPWS::calculateT() {
    valid_ = false;

    if (!is_initialized_)
        throw std::runtime_error("model not set call setNeuralNetworks");

    if (inputPr_ != PH)
        throw std::runtime_error("input pair != PH");

    T_ = fast_engine_backward_.compute_val(p_, h_);
}



void NNPWS::calculateG_derivatives() {
    valid_ = false;

    if (!is_initialized_) throw std::runtime_error("model not set call setNeuralNetworks(path_main_model_pt, path_secondary_model)");//throw exception or call setNeuralNetworks( path_main_model_pt, path_secondary_model)

    const Region r = Regions_Boundaries::determine_region(T_, p_);
    if (r == out_of_regions) throw std::runtime_error("TP out of region"); 

    g_derivatives_ = fast_engine_.compute(r, p_, T_);
 
    /* Check volume is non zero */
    if (std::abs(g_derivatives_.dG_dP) < precision_)
        throw std::runtime_error("volume close to 0");

    valid_ = true;
}

void NNPWS::compute_batch_PT(const std::vector<double>& p_list,
                          const std::vector<double>& T_list,
                          std::vector<NNPWS>& results,
                          const std::string& path_main_model_pt) {

    if (!ModelLoader::instance().load(path_main_model_pt)) {

        std::cerr << "[NNPWS] Erreur chargement modele PT." << std::endl;
        throw std::runtime_error("impossible de charger le modele pt");
    }
    std::shared_ptr<torch::jit::script::Module> module_pt = ModelLoader::instance().get_model(path_main_model_pt);


    if (p_list.size() != T_list.size()) {
        std::string err_msg = "[NNPWS] Erreur de dimension : p_list (" + std::to_string(p_list.size()) +
                              ") et T_list (" + std::to_string(T_list.size()) + ") doivent avoir la même taille.";
        std::cerr << err_msg << std::endl;
        throw std::invalid_argument(err_msg);
    }

    size_t n = p_list.size();
    results.clear();
    results.resize(n, NNPWS(Undefined));

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

            torch::Tensor output = module_pt->get_method("compute_derivatives_batch")(inputs).toTensor();
            auto acc = output.accessor<double, 2>();

            for (size_t k = 0; k < n_reg; ++k) {
                size_t original_idx = indices[k];
                NNPWS& obj = results[original_idx];
                obj.is_initialized_ = false;
                obj.valid_ = true;

                obj.p_ = p_list[original_idx];
                obj.T_ = T_list[original_idx];
                
                obj.g_derivatives_ = FastResult{acc[k][0], acc[k][2], acc[k][1], acc[k][5], acc[k][3], acc[k][4]};

                vol = obj.getVolume() * 1e-3;   // now check if non zero

                /* check volume is non zero */
                if (std::abs(vol) < obj.precision_)
                    throw std::runtime_error("volume close to 0");

            	obj.path_main_model_pt_   = path_main_model_pt;
                // obj.path_secondary_model_ = path_secondary_model;
                

            }

        } catch (const c10::Error& e) {
            std::cerr << "[NNPWS] Erreur Batch LibTorch (Region " << reg_id << "): " << e.what() << std::endl;
            // throw std::runtime_error("Erreur fatale dans le modèle Torch");
        }
    }
}

void NNPWS::compute_batch_PH(const std::vector<double>& p_list,
                             const std::vector<double>& h_list,
                             std::vector<NNPWS>& results,
                             const std::string& path_main_model_pt,
                             const std::string& path_secondary_model)
{
    if (p_list.size() != h_list.size()) {
        throw std::runtime_error("Taille des vecteurs P et H differente dans compute_batch_PH");
    }
    size_t n = p_list.size();

    if (results.size() != n) {
        results.resize(n, NNPWS(Undefined));
    }

    if (!ModelLoader::instance().load(path_secondary_model)) {
        throw std::runtime_error("Impossible de charger le modele secondaire : " + path_secondary_model);
    }
    auto model_ph = ModelLoader::instance().get_model(path_secondary_model);

    auto options = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor input_tensor = torch::empty({(long)n, 2}, options);

    auto input_acc = input_tensor.accessor<double, 2>();
    for (size_t i = 0; i < n; ++i) {
        input_acc[i][0] = p_list[i];
        input_acc[i][1] = h_list[i];
    }

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input_tensor);

    torch::Tensor output_T = model_ph->forward(inputs).toTensor();

    std::vector<double> T_list(n);

    if (output_T.dim() == 2) {
        auto out_acc = output_T.accessor<double, 2>();
        for (size_t i = 0; i < n; ++i) {
            T_list[i] = out_acc[i][0];
        }
    } else {
        auto out_acc = output_T.accessor<double, 1>();
        for (size_t i = 0; i < n; ++i) {
            T_list[i] = out_acc[i];
        }
    }

    compute_batch_PT(p_list, T_list, results, path_main_model_pt);
}
