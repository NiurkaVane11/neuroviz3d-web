#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

OrbitCamera::OrbitCamera(glm::vec3 target, float distance)
    : target(target), distance(distance), yaw(-90.0f), pitch(20.0f) {}

glm::vec3 OrbitCamera::getPosition() const {
    float yawRad = glm::radians(yaw);
    float pitchRad = glm::radians(pitch);
    glm::vec3 offset;
    offset.x = distance * cosf(pitchRad) * cosf(yawRad);
    offset.y = distance * sinf(pitchRad);
    offset.z = distance * cosf(pitchRad) * sinf(yawRad);
    return target + offset;
}

glm::mat4 OrbitCamera::getViewMatrix() const {
    return glm::lookAt(getPosition(), target, glm::vec3(0.0f, 1.0f, 0.0f));
}

void OrbitCamera::rotate(float dYaw, float dPitch) {
    yaw += dYaw;
    pitch += dPitch;
    pitch = std::max(-89.0f, std::min(89.0f, pitch));
}

void OrbitCamera::zoom(float amount) {
    distance -= amount;
    distance = std::max(2.0f, std::min(50.0f, distance));
}

FreeCamera::FreeCamera(glm::vec3 position)
    : position(position), yaw(-90.0f), pitch(0.0f), speed(6.0f), sensitivity(0.1f) {
    updateVectors();
}
void FreeCamera::updateVectors() {
    glm::vec3 f;
    f.x = cosf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    f.y = sinf(glm::radians(pitch));
    f.z = sinf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    front = glm::normalize(f);
    right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
    up = glm::normalize(glm::cross(right, front));
}
glm::mat4 FreeCamera::getViewMatrix() const {
    return glm::lookAt(position, position + front, up);
}
glm::vec3 FreeCamera::getPosition() const {
    return position;
}
void FreeCamera::processMouseMovement(float dx, float dy) {
    yaw += dx * sensitivity;
    pitch += dy * sensitivity;
    pitch = std::max(-89.0f, std::min(89.0f, pitch));
    updateVectors();
}
void FreeCamera::processKeyboard(int direction, float deltaTime) {
    float velocity = speed * deltaTime;
    switch (direction) {
        case 0: position += front * velocity; break;
        case 1: position -= front * velocity; break;
        case 2: position -= right * velocity; break;
        case 3: position += right * velocity; break;
        case 4: position += glm::vec3(0.0f, 1.0f, 0.0f) * velocity; break;
        case 5: position -= glm::vec3(0.0f, 1.0f, 0.0f) * velocity; break;
    }
}

void FreeCamera::lookAt(glm::vec3 target) {
    glm::vec3 dir = glm::normalize(target - position);
    pitch = glm::degrees(asinf(dir.y));
    yaw = glm::degrees(atan2f(dir.z, dir.x));
    updateVectors();
}
