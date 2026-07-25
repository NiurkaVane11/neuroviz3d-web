#pragma once
#include <string>
#include <vector>

struct LayerWeights {
    std::string name;
    int in_features;
    int out_features;
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
};

struct NetworkArchitecture {
    std::vector<int> layer_sizes;
    std::vector<LayerWeights> layers;
    bool valid = false;
};

struct ActivationSample {
    std::vector<float> input;
    int true_label = -1;
    int predicted_label = -1;
    std::vector<std::vector<float>> activations;
};

struct ActivationSet {
    std::vector<ActivationSample> samples;
    bool valid = false;
};

struct TrainingHistory {
    std::vector<int> epochs;
    std::vector<float> train_loss;
    std::vector<float> test_accuracy;
    bool valid = false;
};

NetworkArchitecture loadNetworkFromJSON(const std::string& path);
ActivationSet loadActivationsFromJSON(const std::string& path);
TrainingHistory loadTrainingHistoryFromJSON(const std::string& path);
