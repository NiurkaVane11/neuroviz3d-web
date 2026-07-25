#pragma once
#include <glm/glm.hpp>

class OrbitCamera {
public:
    OrbitCamera(glm::vec3 target, float distance);
    glm::mat4 getViewMatrix() const;
    glm::vec3 getPosition() const;
    void rotate(float dYaw, float dPitch);
    void zoom(float amount);

private:
    glm::vec3 target;
    float distance;
    float yaw;
    float pitch;
};
