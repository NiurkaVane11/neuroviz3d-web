#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
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

// Mapea un peso [-1, 1] a un color: cian si es positivo, naranja/rojo si es negativo.
// La magnitud del peso controla que tan intenso/brillante es el color.
glm::vec3 weightColor(float w) {
    float mag = std::min(std::fabs(w), 1.0f);
    if (w >= 0.0f) {
        return glm::mix(glm::vec3(0.05f, 0.25f, 0.3f), glm::vec3(0.0f, 0.85f, 0.95f), mag);
    } else {
        return glm::mix(glm::vec3(0.35f, 0.15f, 0.1f), glm::vec3(0.95f, 0.35f, 0.25f), mag);
    }
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "NeuroViz3D - Fase 4", nullptr, nullptr);
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

    // Fase 4: intenta cargar pesos reales de PyTorch; si no existe o no coincide, cae a random (seed 42)
    NetworkArchitecture net = loadNetworkFromJSON("network.json");
    bool useRealWeights = net.valid && net.layers.size() == positions.size() - 1;

    if (net.valid && !useRealWeights) {
        std::cerr << "[main] network.json cargado pero no coincide con layerSizes fijo {4,8,6,3}; "
                     "usando fallback random." << std::endl;
    }
    if (useRealWeights) {
        std::cout << "[main] Usando pesos reales de PyTorch (network.json)" << std::endl;
    } else {
        std::cout << "[main] network.json no disponible; usando pesos random (fallback, seed 42)" << std::endl;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> weightDist(-1.0f, 1.0f);

    std::vector<float> lineVerts;
    for (size_t l = 0; l + 1 < positions.size(); ++l) {
        for (size_t i0 = 0; i0 < positions[l].size(); ++i0) {
            for (size_t i1 = 0; i1 < positions[l + 1].size(); ++i1) {
                glm::vec3 p0 = positions[l][i0];
                glm::vec3 p1 = positions[l + 1][i1];

                float w;
                if (useRealWeights) {
                    // weights[out][in] -> out = neurona destino (i1), in = neurona origen (i0)
                    w = net.layers[l].weights[i1][i0];
                    w = std::max(-1.0f, std::min(1.0f, w));
                } else {
                    w = weightDist(rng);
                }

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

    glm::vec3 lightPos(4.0f, 6.0f, 8.0f);

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.04f, 0.04f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
            (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.getViewMatrix();
        glm::vec3 viewPos = camera.getPosition();

        // Conexiones
        lineShader.use();
        lineShader.setMat4("uMVP", projection * view);
        lines.draw();

        // Neuronas con iluminacion Phong
        sphereShader.use();
        sphereShader.setMat4("uView", view);
        sphereShader.setMat4("uProjection", projection);
        sphereShader.setVec3("uLightPos", lightPos);
        sphereShader.setVec3("uViewPos", viewPos);
        sphereShader.setVec3("uColor", glm::vec3(0.85f, 0.85f, 0.88f));

        for (auto& layer : positions) {
            for (auto& pos : layer) {
                glm::mat4 model = glm::translate(glm::mat4(1.0f), pos);
                sphereShader.setMat4("uModel", model);
                sphere.draw();
            }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
