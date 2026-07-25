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

class FreeCamera {
public:
    FreeCamera(glm::vec3 position);
    glm::mat4 getViewMatrix() const;
    glm::vec3 getPosition() const;
    void processMouseMovement(float dx, float dy);
    void processKeyboard(int direction, float deltaTime); // 0=forward,1=back,2=left,3=right,4=up,5=down
    void lookAt(glm::vec3 target);
private:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    float yaw;
    float pitch;
    float speed;
    float sensitivity;
    void updateVectors();
};
