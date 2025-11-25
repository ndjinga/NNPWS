#ifndef NNPWS_MODELLOADER_HXX
#define NNPWS_MODELLOADER_HXX

#pragma once

#include <torch/script.h>
#include <string>
#include <memory>

class ModelLoader {
private:
    std::shared_ptr<torch::jit::script::Module> model_ptr;

    ModelLoader() = default;

public:
    static ModelLoader& instance();

    ModelLoader(const ModelLoader&) = delete;
    void operator=(const ModelLoader&) = delete;

    bool load(const std::string& path);

    std::shared_ptr<torch::jit::script::Module> get_model() const;

    bool is_ready() const;
};

#endif //NNPWS_MODELLOADER_HXX