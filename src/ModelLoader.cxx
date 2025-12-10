#include "ModelLoader.hxx"
#include <iostream>

// Singleton Pattern
ModelLoader& ModelLoader::instance() {
    static ModelLoader _instance;
    return _instance;
}

bool ModelLoader::load(const std::string& path) {

    if (models_map.find(path) != models_map.end()) {
        return true;
    }

    try {
        torch::NoGradGuard no_grad;

        auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(path, torch::kCPU));

        module->eval();
        module->to(torch::kDouble);

        models_map[path] = module;

        std::cout << "[ModelLoader] Modèle chargé et mis en cache : " << path << std::endl;
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "[ModelLoader] Erreur critique de chargement pour " << path << std::endl;
        //std::cerr << "[ModelLoader] Erreur critique: " << e.what() << std::endl;
        return false;
    }
}

std::shared_ptr<torch::jit::script::Module> ModelLoader::get_model(const std::string& path) const {
    auto it = models_map.find(path);
    if (it != models_map.end()) {
        return it->second;
    }
    return nullptr;
}

bool ModelLoader::is_loaded(const std::string& path) const {
    return models_map.find(path) != models_map.end();
}