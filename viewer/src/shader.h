#pragma once
#include <string>
#include <glad/glad.h>

class Shader {
public:
    unsigned int ID;

    Shader(const char* vertexPath, const char* fragmentPath);
    void use() const;

private:
    std::string readFile(const char* path);
    unsigned int compile(const char* source, GLenum type, const char* label);
};
