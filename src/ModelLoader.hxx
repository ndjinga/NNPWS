#ifndef NNPWS_MODELLOADER_HXX
#define NNPWS_MODELLOADER_HXX

#ifdef NNPWS_WITH_TORCH

#include <torch/script.h>
#include <string>
#include <memory>
#include <map>

class ModelLoader {
private:
    std::map<std::string, std::shared_ptr<torch::jit::script::Module>> models_map;
    ModelLoader() = default;

public:
    static ModelLoader& instance();

    ModelLoader(const ModelLoader&) = delete;
    void operator=(const ModelLoader&) = delete;

    bool load(const std::string& path);
    std::shared_ptr<torch::jit::script::Module> get_model(const std::string& path) const;
    bool is_loaded(const std::string& path) const;
};

#endif // NNPWS_WITH_TORCH

#endif // NNPWS_MODELLOADER_HXX
