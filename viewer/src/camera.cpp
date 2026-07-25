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
