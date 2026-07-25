#include "network_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <stdexcept>

namespace {

struct JsonParser {
    const std::string& s;
    size_t pos = 0;

    JsonParser(const std::string& str) : s(str) {}

    void skipWhitespace() {
        while (pos < s.size() && std::isspace((unsigned char)s[pos])) pos++;
    }

    bool match(char c) {
        skipWhitespace();
        if (pos < s.size() && s[pos] == c) { pos++; return true; }
        return false;
    }

    void expect(char c) {
        if (!match(c)) {
            throw std::runtime_error(std::string("JSON: se esperaba '") + c + "' en pos " + std::to_string(pos));
        }
    }

    std::string parseString() {
        skipWhitespace();
        expect('"');
        std::string out;
        while (pos < s.size() && s[pos] != '"') {
            out += s[pos++];
        }
        expect('"');
        return out;
    }

    double parseNumber() {
        skipWhitespace();
        size_t start = pos;
        if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) pos++;
        while (pos < s.size() && (std::isdigit((unsigned char)s[pos]) || s[pos] == '.' ||
               s[pos] == 'e' || s[pos] == 'E' || s[pos] == '-' || s[pos] == '+')) pos++;
        return std::stod(s.substr(start, pos - start));
    }

    std::string parseKey() {
        std::string key = parseString();
        expect(':');
        return key;
    }

    std::vector<float> parseFloatArray() {
        std::vector<float> arr;
        skipWhitespace();
        expect('[');
        skipWhitespace();
        if (match(']')) return arr;
        while (true) {
            arr.push_back((float)parseNumber());
            skipWhitespace();
            if (match(',')) continue;
            expect(']');
            break;
        }
        return arr;
    }

    std::vector<std::vector<float>> parseFloatMatrix() {
        std::vector<std::vector<float>> mat;
        skipWhitespace();
        expect('[');
        skipWhitespace();
        if (match(']')) return mat;
        while (true) {
            mat.push_back(parseFloatArray());
            skipWhitespace();
            if (match(',')) continue;
            expect(']');
            break;
        }
        return mat;
    }

    std::vector<int> parseIntArray() {
        std::vector<int> arr;
        auto floats = parseFloatArray();
        for (float f : floats) arr.push_back((int)f);
        return arr;
    }
    glm::vec3 parseVec3() {
        auto v = parseFloatArray();
        if (v.size() != 3) throw std::runtime_error("JSON: se esperaba vec3 de 3 componentes");
        return glm::vec3(v[0], v[1], v[2]);
    }
};

} // namespace

NetworkArchitecture loadNetworkFromJSON(const std::string& path) {
    NetworkArchitecture arch;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[network_loader] No se pudo abrir: " << path << std::endl;
        return arch;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    try {
        JsonParser p(content);
        p.expect('{');

        while (true) {
            std::string key = p.parseKey();

            if (key == "layer_sizes") {
                arch.layer_sizes = p.parseIntArray();
            } else if (key == "layers") {
                p.skipWhitespace();
                p.expect('[');
                p.skipWhitespace();
                if (!p.match(']')) {
                    while (true) {
                        LayerWeights layer;
                        p.expect('{');
                        while (true) {
                            std::string lkey = p.parseKey();
                            if (lkey == "name") {
                                layer.name = p.parseString();
                            } else if (lkey == "in_features") {
                                layer.in_features = (int)p.parseNumber();
                            } else if (lkey == "out_features") {
                                layer.out_features = (int)p.parseNumber();
                            } else if (lkey == "weights") {
                                layer.weights = p.parseFloatMatrix();
                            } else if (lkey == "bias") {
                                layer.bias = p.parseFloatArray();
                            }
                            p.skipWhitespace();
                            if (p.match(',')) continue;
                            p.expect('}');
                            break;
                        }
                        arch.layers.push_back(layer);
                        p.skipWhitespace();
                        if (p.match(',')) continue;
                        p.expect(']');
                        break;
                    }
                }
            }

            p.skipWhitespace();
            if (p.match(',')) continue;
            p.expect('}');
            break;
        }

        arch.valid = true;
        std::cout << "[network_loader] Cargado: " << arch.layers.size()
                  << " capas desde " << path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[network_loader] Error parseando JSON: " << e.what() << std::endl;
        arch.valid = false;
    }

    return arch;
}

ActivationSet loadActivationsFromJSON(const std::string& path) {
    ActivationSet set;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[network_loader] No se pudo abrir: " << path << std::endl;
        return set;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    try {
        JsonParser p(content);
        p.expect('{');

        while (true) {
            std::string key = p.parseKey();

            if (key == "samples") {
                p.skipWhitespace();
                p.expect('[');
                p.skipWhitespace();
                if (!p.match(']')) {
                    while (true) {
                        ActivationSample sample;
                        p.expect('{');
                        while (true) {
                            std::string skey = p.parseKey();
                            if (skey == "input") {
                                sample.input = p.parseFloatArray();
                            } else if (skey == "true_label") {
                                sample.true_label = (int)p.parseNumber();
                            } else if (skey == "predicted_label") {
                                sample.predicted_label = (int)p.parseNumber();
                            } else if (skey == "activations") {
                                sample.activations = p.parseFloatMatrix();
                            }
                            p.skipWhitespace();
                            if (p.match(',')) continue;
                            p.expect('}');
                            break;
                        }
                        set.samples.push_back(sample);
                        p.skipWhitespace();
                        if (p.match(',')) continue;
                        p.expect(']');
                        break;
                    }
                }
            }

            p.skipWhitespace();
            if (p.match(',')) continue;
            p.expect('}');
            break;
        }

        set.valid = true;
        std::cout << "[network_loader] Cargados " << set.samples.size()
                  << " samples de activaciones desde " << path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[network_loader] Error parseando activations: " << e.what() << std::endl;
        set.valid = false;
    }

    return set;
}

TrainingHistory loadTrainingHistoryFromJSON(const std::string& path) {
    TrainingHistory hist;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[network_loader] No se pudo abrir: " << path << std::endl;
        return hist;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    try {
        JsonParser p(content);
        p.expect('{');

        while (true) {
            std::string key = p.parseKey();

            if (key == "epochs") {
                hist.epochs = p.parseIntArray();
            } else if (key == "train_loss") {
                hist.train_loss = p.parseFloatArray();
            } else if (key == "test_accuracy") {
                hist.test_accuracy = p.parseFloatArray();
            }

            p.skipWhitespace();
            if (p.match(',')) continue;
            p.expect('}');
            break;
        }

        hist.valid = true;
        std::cout << "[network_loader] Historial de entrenamiento cargado: " << hist.epochs.size()
                  << " epocas desde " << path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[network_loader] Error parseando training history: " << e.what() << std::endl;
        hist.valid = false;
    }

    return hist;
}

EmbeddingSet loadEmbeddingsFromJSON(const std::string& path) {
    EmbeddingSet result;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir " << path << "\n";
        return result;
    }

    std::stringstream buf;
    buf << file.rdbuf();
    std::string content = buf.str();

    try {
        JsonParser p(content);
        p.expect('{');
        while (true) {
            std::string key = p.parseKey();

            if (key == "numSamples") {
                p.parseNumber();
            } else if (key == "numClasses") {
                result.num_classes = (int)p.parseNumber();
            } else if (key == "samples") {
                p.expect('[');
                p.skipWhitespace();
                if (!p.match(']')) {
                    while (true) {
                        p.expect('{');
                        EmbeddingSample sample;
                        while (true) {
                            std::string sk = p.parseKey();
                            if (sk == "index") sample.index = (int)p.parseNumber();
                            else if (sk == "label") sample.true_label = (int)p.parseNumber();
                            else if (sk == "predicted") sample.predicted_label = (int)p.parseNumber();
                            else if (sk == "pca") sample.pca_position = p.parseVec3();
                            else if (sk == "tsne") sample.tsne_position = p.parseVec3();
                            p.skipWhitespace();
                            if (p.match(',')) continue;
                            p.expect('}');
                            break;
                        }
                        result.samples.push_back(sample);
                        p.skipWhitespace();
                        if (p.match(',')) continue;
                        p.expect(']');
                        break;
                    }
                }
            }

            p.skipWhitespace();
            if (p.match(',')) continue;
            p.expect('}');
            break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parseando " << path << ": " << e.what() << "\n";
        result.samples.clear();
        result.valid = false;
        return result;
    }

    result.valid = !result.samples.empty();
    return result;
}
