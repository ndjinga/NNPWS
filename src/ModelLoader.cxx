#include "ModelLoader.hxx"
#include <iostream>

ModelLoader& ModelLoader::instance() {
    static ModelLoader _instance;
    return _instance;
}

bool ModelLoader::load(const std::string& path) {
    try {
        torch::NoGradGuard no_grad;

        auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(path));

        module->eval();
        module->to(torch::kDouble);

        this->model_ptr = module;
        std::cout << "[ModelRepository] Modèle chargé avec succès : " << path << std::endl;
        return true;
    }
    catch (const c10::Error& e) {
        //std::cerr << "[ModelRepository] Erreur critique de chargement : " << e.what() << std::endl;
        std::cerr << "[ModelRepository] Erreur critique de chargement" << std::endl;
        return false;
    }
}

std::shared_ptr<torch::jit::script::Module> ModelLoader::get_model() const {
    return model_ptr;
}

bool ModelLoader::is_ready() const {
    return model_ptr != nullptr;
}