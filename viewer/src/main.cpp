#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "shader.h"
#include "mesh.h"
#include "camera.h"
#include "network_loader.h"

const unsigned int SCR_WIDTH = 900;
const unsigned int SCR_HEIGHT = 700;

OrbitCamera camera(glm::vec3(0.0f), 12.0f);
bool dragging = false;
double lastX = 0.0, lastY = 0.0;

bool spacePressedLastFrame = false;
bool animating = false;
double animStartTime = 0.0;
int activeSampleIdx = -1;

bool tPressedLastFrame = false;
bool trainingAnimating = false;
double trainingAnimStartTime = 0.0;
const float TRAINING_ANIM_DURATION = 4.0f; // segundos para revelar toda la curva

const float SEGMENT_DURATION = 0.8f;
const float INPUT_MIN_GLOW = 0.18f;

const char* IRIS_CLASS_NAMES[3] = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

// Rectangulo de la grafica en coordenadas de pantalla (pixeles), esquina superior derecha
const float GRAPH_MARGIN = 20.0f;
const float GRAPH_WIDTH = 260.0f;
const float GRAPH_HEIGHT = 150.0f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            dragging = true;
            glfwGetCursorPos(window, &lastX, &lastY);
        } else if (action == GLFW_RELEASE) {
            dragging = false;
        }
    }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    if (dragging) {
        float dx = static_cast<float>(xpos - lastX);
        float dy = static_cast<float>(ypos - lastY);
        camera.rotate(dx * 0.3f, -dy * 0.3f);
        lastX = xpos;
        lastY = ypos;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.zoom(static_cast<float>(yoffset) * 0.8f);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

glm::vec3 weightColor(float w) {
    float mag = std::min(std::fabs(w), 1.0f);
    if (w >= 0.0f) {
        return glm::mix(glm::vec3(0.05f, 0.25f, 0.3f), glm::vec3(0.0f, 0.85f, 0.95f), mag);
    } else {
        return glm::mix(glm::vec3(0.35f, 0.15f, 0.1f), glm::vec3(0.95f, 0.35f, 0.25f), mag);
    }
}

float normalizeActivation(float value, const std::vector<float>& layerActivations) {
    float maxAbs = 0.0001f;
    for (float v : layerActivations) maxAbs = std::max(maxAbs, std::fabs(v));
    return std::clamp(std::fabs(value) / maxAbs, 0.0f, 1.0f);
}

glm::vec3 emissiveColor(float intensity) {
    glm::vec3 low(0.0f, 0.0f, 0.0f);
    glm::vec3 high(1.4f, 1.1f, 0.5f);
    return glm::mix(low, high, intensity);
}

void updateWindowTitle(GLFWwindow* window, const ActivationSample* sample) {
    if (!sample) {
        glfwSetWindowTitle(window, "NeuroViz3D - Fase 6 (SPACE = forward pass, T = training)");
        return;
    }
    std::ostringstream oss;
    oss << "NeuroViz3D - Fase 6 | Real: " << IRIS_CLASS_NAMES[sample->true_label]
        << " | Prediccion: " << IRIS_CLASS_NAMES[sample->predicted_label]
        << (sample->true_label == sample->predicted_label ? " (correcto)" : " (incorrecto)");
    glfwSetWindowTitle(window, oss.str().c_str());
}

// Construye los vertices (pos + color) de una curva normalizada dentro del rectangulo de la grafica.
// values: serie completa; minV/maxV: rango para normalizar verticalmente; color: color fijo de la linea.
std::vector<float> buildGraphCurve(const std::vector<float>& values, float minV, float maxV,
                                    float graphX, float graphY, glm::vec3 color) {
    std::vector<float> verts;
    if (values.empty()) return verts;
    float range = std::max(maxV - minV, 0.0001f);
    for (size_t i = 0; i < values.size(); ++i) {
        float px = graphX + (float)i / (float)(values.size() - 1) * GRAPH_WIDTH;
        float norm = (values[i] - minV) / range;
        float py = graphY + GRAPH_HEIGHT - norm * GRAPH_HEIGHT; // invertido: mayor valor = mas arriba
        verts.push_back(px); verts.push_back(py); verts.push_back(0.0f);
        verts.push_back(color.r); verts.push_back(color.g); verts.push_back(color.b);
    }
    return verts;
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "NeuroViz3D - Fase 6", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "ERROR: no se pudo crear la ventana GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "ERROR: no se pudo inicializar GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Shader sphereShader("shaders/sphere.vert", "shaders/sphere.frag");
    Shader lineShader("shaders/line.vert", "shaders/line.frag");

    std::vector<int> layerSizes = {4, 8, 6, 3};
    float layerSpacing = 3.5f;
    float neuronSpacing = 1.0f;

    std::vector<std::vector<glm::vec3>> positions(layerSizes.size());
    for (size_t l = 0; l < layerSizes.size(); ++l) {
        int count = layerSizes[l];
        float layerX = l * layerSpacing - (layerSizes.size() - 1) * layerSpacing * 0.5f;
        float totalHeight = (count - 1) * neuronSpacing;
        for (int i = 0; i < count; ++i) {
            float y = i * neuronSpacing - totalHeight * 0.5f;
            positions[l].push_back(glm::vec3(layerX, y, 0.0f));
        }
    }

    NetworkArchitecture net = loadNetworkFromJSON("network.json");
    bool useRealWeights = net.valid && net.layers.size() == positions.size() - 1;
    if (useRealWeights) {
        std::cout << "[main] Usando pesos reales de PyTorch (network.json)" << std::endl;
    } else {
        std::cout << "[main] network.json no disponible; usando pesos random (fallback, seed 42)" << std::endl;
    }

    ActivationSet activations = loadActivationsFromJSON("activations.json");
    if (activations.valid && !activations.samples.empty()) {
        std::cout << "[main] " << activations.samples.size()
                  << " samples de activaciones listos (SPACE)" << std::endl;
    }

    TrainingHistory history = loadTrainingHistoryFromJSON("training_history.json");
    if (history.valid) {
        std::cout << "[main] Historial de entrenamiento listo (" << history.epochs.size()
                  << " epocas, tecla T)" << std::endl;
    } else {
        std::cout << "[main] training_history.json no disponible; T no hara nada" << std::endl;
    }

    updateWindowTitle(window, nullptr);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> weightDist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> sampleDist(0, activations.valid && !activations.samples.empty()
        ? (int)activations.samples.size() - 1 : 0);

    std::vector<float> lineVerts;
    for (size_t l = 0; l + 1 < positions.size(); ++l) {
        for (size_t i0 = 0; i0 < positions[l].size(); ++i0) {
            for (size_t i1 = 0; i1 < positions[l + 1].size(); ++i1) {
                glm::vec3 p0 = positions[l][i0];
                glm::vec3 p1 = positions[l + 1][i1];
                float w = useRealWeights
                    ? std::max(-1.0f, std::min(1.0f, net.layers[l].weights[i1][i0]))
                    : weightDist(rng);
                glm::vec3 c = weightColor(w);
                lineVerts.push_back(p0.x); lineVerts.push_back(p0.y); lineVerts.push_back(p0.z);
                lineVerts.push_back(c.r); lineVerts.push_back(c.g); lineVerts.push_back(c.b);
                lineVerts.push_back(p1.x); lineVerts.push_back(p1.y); lineVerts.push_back(p1.z);
                lineVerts.push_back(c.r); lineVerts.push_back(c.g); lineVerts.push_back(c.b);
            }
        }
    }

    SphereMesh sphere(0.22f, 24, 16);
    LineMesh lines(lineVerts);
    SphereMesh pulseSphere(0.09f, 16, 10);

    // --- Overlay 2D: grafica de entrenamiento ---
    float graphX = (float)SCR_WIDTH - GRAPH_WIDTH - GRAPH_MARGIN;
    float graphY = GRAPH_MARGIN;

    glm::vec3 lossColor(0.95f, 0.35f, 0.25f);   // mismo tono que peso negativo
    glm::vec3 accColor(0.0f, 0.85f, 0.95f);     // mismo tono que peso positivo

    // Marco de la grafica (estatico): un rectangulo simple en gris tenue
    std::vector<float> frameVerts = {
        graphX, graphY, 0.0f,                         0.4f, 0.4f, 0.45f,
        graphX + GRAPH_WIDTH, graphY, 0.0f,            0.4f, 0.4f, 0.45f,
        graphX + GRAPH_WIDTH, graphY, 0.0f,            0.4f, 0.4f, 0.45f,
        graphX + GRAPH_WIDTH, graphY + GRAPH_HEIGHT, 0.0f, 0.4f, 0.4f, 0.45f,
        graphX + GRAPH_WIDTH, graphY + GRAPH_HEIGHT, 0.0f, 0.4f, 0.4f, 0.45f,
        graphX, graphY + GRAPH_HEIGHT, 0.0f,           0.4f, 0.4f, 0.45f,
        graphX, graphY + GRAPH_HEIGHT, 0.0f,           0.4f, 0.4f, 0.45f,
        graphX, graphY, 0.0f,                          0.4f, 0.4f, 0.45f,
    };
    LineMesh graphFrame(frameVerts);

    // Curvas dinamicas (se llenan solo si el historial cargo bien)
    std::vector<float> lossVertsFull, accVertsFull;
    if (history.valid) {
        float maxLoss = *std::max_element(history.train_loss.begin(), history.train_loss.end());
        lossVertsFull = buildGraphCurve(history.train_loss, 0.0f, maxLoss, graphX, graphY, lossColor);
        accVertsFull = buildGraphCurve(history.test_accuracy, 0.0f, 1.0f, graphX, graphY, accColor);
    }
    // Se crean con al menos un punto dummy si no hay historial, para que el LineMesh no falle con buffer vacio
    std::vector<float> dummy = { 0,0,0, 0,0,0, 0,0,0, 0,0,0 };
    LineMesh lossCurve(history.valid ? lossVertsFull : dummy, true);
    LineMesh accCurve(history.valid ? accVertsFull : dummy, true);

    glm::vec3 lightPos(4.0f, 6.0f, 8.0f);
    size_t numLayerGaps = positions.size() - 1;

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        bool spacePressed = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spacePressed && !spacePressedLastFrame && activations.valid && !activations.samples.empty()) {
            activeSampleIdx = sampleDist(rng);
            animating = true;
            animStartTime = glfwGetTime();
            const auto& s = activations.samples[activeSampleIdx];
            std::cout << "[forward pass] sample #" << activeSampleIdx
                      << " | label real=" << s.true_label
                      << " | prediccion=" << s.predicted_label << std::endl;
            updateWindowTitle(window, &s);
        }
        spacePressedLastFrame = spacePressed;

        bool tPressed = glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS;
        if (tPressed && !tPressedLastFrame && history.valid) {
            trainingAnimating = true;
            trainingAnimStartTime = glfwGetTime();
            std::cout << "[training] reproduciendo historial (" << history.epochs.size() << " epocas)" << std::endl;
        }
        tPressedLastFrame = tPressed;

        double elapsed = animating ? (glfwGetTime() - animStartTime) : 0.0;
        float totalDuration = SEGMENT_DURATION * (float)numLayerGaps;

        glClearColor(0.04f, 0.04f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
            (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.getViewMatrix();
        glm::vec3 viewPos = camera.getPosition();

        lineShader.use();
        lineShader.setMat4("uMVP", projection * view);
        lines.draw();

        std::vector<bool> layerReached(positions.size(), false);
        layerReached[0] = animating;
        int activeGap = -1;
        float gapProgress = 0.0f;

        if (animating && activations.valid && activeSampleIdx >= 0) {
            for (size_t gap = 0; gap < numLayerGaps; ++gap) {
                double gapStart = gap * SEGMENT_DURATION;
                double gapEnd = gapStart + SEGMENT_DURATION;
                if (elapsed >= gapEnd) {
                    layerReached[gap + 1] = true;
                } else if (elapsed >= gapStart) {
                    activeGap = (int)gap;
                    gapProgress = (float)((elapsed - gapStart) / SEGMENT_DURATION);
                    break;
                }
            }
            if (elapsed >= totalDuration) layerReached[numLayerGaps] = true;
            if (elapsed > totalDuration + 1.5) {
                animating = false;
                updateWindowTitle(window, activeSampleIdx >= 0 ? &activations.samples[activeSampleIdx] : nullptr);
            }
        }

        sphereShader.use();
        sphereShader.setMat4("uView", view);
        sphereShader.setMat4("uProjection", projection);
        sphereShader.setVec3("uLightPos", lightPos);
        sphereShader.setVec3("uViewPos", viewPos);

        const ActivationSample* activeSample = (activations.valid && activeSampleIdx >= 0)
            ? &activations.samples[activeSampleIdx] : nullptr;

        for (size_t l = 0; l < positions.size(); ++l) {
            for (size_t i = 0; i < positions[l].size(); ++i) {
                glm::mat4 model = glm::translate(glm::mat4(1.0f), positions[l][i]);
                sphereShader.setMat4("uModel", model);
                sphereShader.setVec3("uColor", glm::vec3(0.85f, 0.85f, 0.88f));

                glm::vec3 emissive(0.0f);
                if (activeSample && layerReached[l] && l < activeSample->activations.size()) {
                    float val = activeSample->activations[l][i];
                    float intensity = normalizeActivation(val, activeSample->activations[l]);
                    if (l == 0) intensity = INPUT_MIN_GLOW + intensity * (1.0f - INPUT_MIN_GLOW);
                    emissive = emissiveColor(intensity);
                }
                sphereShader.setVec3("uEmissive", emissive);
                sphere.draw();
            }
        }

        if (activeGap >= 0 && activeSample) {
            size_t l = (size_t)activeGap;
            for (size_t i0 = 0; i0 < positions[l].size(); ++i0) {
                for (size_t i1 = 0; i1 < positions[l + 1].size(); ++i1) {
                    glm::vec3 p0 = positions[l][i0];
                    glm::vec3 p1 = positions[l + 1][i1];
                    glm::vec3 pulsePos = glm::mix(p0, p1, gapProgress);
                    float srcVal = (l < activeSample->activations.size()) ? activeSample->activations[l][i0] : 0.0f;
                    float intensity = normalizeActivation(srcVal, activeSample->activations[l]);
                    if (l == 0) intensity = INPUT_MIN_GLOW + intensity * (1.0f - INPUT_MIN_GLOW);
                    glm::mat4 model = glm::translate(glm::mat4(1.0f), pulsePos);
                    sphereShader.setMat4("uModel", model);
                    sphereShader.setVec3("uColor", glm::vec3(0.0f));
                    sphereShader.setVec3("uEmissive", emissiveColor(std::max(intensity, 0.5f)));
                    pulseSphere.draw();
                }
            }
        }

        // --- Overlay 2D: grafica de entrenamiento (siempre encima, sin depth test) ---
        if (history.valid) {
            glDisable(GL_DEPTH_TEST);
            glm::mat4 ortho = glm::ortho(0.0f, (float)SCR_WIDTH, (float)SCR_HEIGHT, 0.0f, -1.0f, 1.0f);
            lineShader.use();
            lineShader.setMat4("uMVP", ortho);

            graphFrame.draw();

            size_t totalPoints = history.epochs.size();
            size_t visiblePoints = totalPoints;
            if (trainingAnimating) {
                double tElapsed = glfwGetTime() - trainingAnimStartTime;
                float progress = std::clamp((float)(tElapsed / TRAINING_ANIM_DURATION), 0.0f, 1.0f);
                visiblePoints = std::max((size_t)2, (size_t)(progress * totalPoints));
                if (progress >= 1.0f) trainingAnimating = false;
            } else if (animating || activeSample == nullptr) {
                // si nunca se presiono T, no mostramos la curva completa de una vez;
                // solo se revela tras presionar T al menos una vez.
                visiblePoints = 0;
            }

            if (visiblePoints > 0) {
                lossCurve.draw((unsigned int)visiblePoints);
                accCurve.draw((unsigned int)visiblePoints);
            }

            glEnable(GL_DEPTH_TEST);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
