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

class LineMesh {
public:
    explicit LineMesh(const std::vector<float>& pointsXYZ);
    void draw() const;

private:
    unsigned int VAO, VBO;
    unsigned int vertexCount;
};
