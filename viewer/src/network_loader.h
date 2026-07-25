#pragma once
#include <string>
#include <vector>

struct LayerWeights {
    std::string name;
    int in_features;
    int out_features;
    std::vector<std::vector<float>> weights; // [out][in]
    std::vector<float> bias;                  // [out]
};

struct NetworkArchitecture {
    std::vector<int> layer_sizes;
    std::vector<LayerWeights> layers;
    bool valid = false;
};

NetworkArchitecture loadNetworkFromJSON(const std::string& path);
