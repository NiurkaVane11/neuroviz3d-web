#pragma once
#include <vector>
#include <glad/glad.h>

class SphereMesh {
public:
    SphereMesh(float radius, int sectors, int stacks);
    void draw() const;

private:
    unsigned int VAO, VBO, EBO;
    unsigned int indexCount;
};

// Cada vertice: posicion (3 floats) + color (3 floats)
class LineMesh {
public:
    // dynamic=true permite actualizar los vertices en tiempo de ejecucion via updateVertices()
    explicit LineMesh(const std::vector<float>& posColorData, bool dynamic = false);
    void draw() const;                  // dibuja como GL_LINES (pares independientes) - conexiones de la red
    void draw(unsigned int count) const; // dibuja los primeros N vertices como GL_LINE_STRIP - curvas continuas
    void updateVertices(const std::vector<float>& posColorData); // solo valido si se creo con dynamic=true

private:
    unsigned int VAO, VBO;
    unsigned int vertexCount;
    bool isDynamic = false;
};
