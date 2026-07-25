#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>

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

struct EmbeddingSample {
    int index = -1;
    int true_label = -1;
    int predicted_label = -1;
    glm::vec3 pca_position;
    glm::vec3 tsne_position;
};
struct EmbeddingSet {
    std::vector<EmbeddingSample> samples;
    int num_classes = 3;
    bool valid = false;
};
EmbeddingSet loadEmbeddingsFromJSON(const std::string& path);
