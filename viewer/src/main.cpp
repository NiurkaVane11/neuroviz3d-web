#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>
#include "shader.h"
#include "mesh.h"
#include "camera.h"

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

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "NeuroViz3D - Fase 2", nullptr, nullptr);
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

    Shader shader("shaders/neuron.vert", "shaders/neuron.frag");

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

    std::vector<float> lineVerts;
    for (size_t l = 0; l + 1 < positions.size(); ++l) {
        for (auto& p0 : positions[l]) {
            for (auto& p1 : positions[l + 1]) {
                lineVerts.push_back(p0.x); lineVerts.push_back(p0.y); lineVerts.push_back(p0.z);
                lineVerts.push_back(p1.x); lineVerts.push_back(p1.y); lineVerts.push_back(p1.z);
            }
        }
    }

    SphereMesh sphere(0.22f, 24, 16);
    LineMesh lines(lineVerts);

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.04f, 0.04f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
            (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.getViewMatrix();

        shader.use();

        shader.setMat4("uMVP", projection * view);
        shader.setVec3("uColor", glm::vec3(0.0f, 0.55f, 0.65f));
        lines.draw();

        shader.setVec3("uColor", glm::vec3(0.85f, 0.85f, 0.88f));
        for (auto& layer : positions) {
            for (auto& pos : layer) {
                glm::mat4 model = glm::translate(glm::mat4(1.0f), pos);
                shader.setMat4("uMVP", projection * view * model);
                sphere.draw();
            }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
