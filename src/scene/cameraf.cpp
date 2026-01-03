#include "cameraf.hpp"


__host__ Camera::Camera() : width(10), height(10), samples(10), depth(4),
    eye(Vector3D(0,0,0)), direction(Vector3D(1,0,0)), up(Vector3D(0,0,1)),
    fov(110), 
    window(Vector3D(), Vector3D(), Vector3D(), Vector3D()) {}

__host__ Camera::Camera(int widthPixels, int heightPixels) : width(widthPixels), height(heightPixels), samples(10), depth(4),
    eye(Vector3D(0,0,0)), direction(Vector3D(1,0,0)), up(Vector3D(0,0,1)),
    fov(110), window(Vector3D(), Vector3D(), Vector3D(), Vector3D()) {
        if(widthPixels < 0.0f || heightPixels < 0.0f) {
            throw std::invalid_argument("Camera: Values must be positive");
        }
    }

__host__ Camera::Camera(int widthPixels, int heightPixels, float FOV) : width(widthPixels), height(heightPixels), 
    samples(10), depth(4), eye(Vector3D(0,0,0)), direction(Vector3D(1,0,0)), up(Vector3D(0,0,1)),
    window(Vector3D(), Vector3D(), Vector3D(), Vector3D()) {
        if(widthPixels < 0.0f || heightPixels < 0.0f) {
            throw std::invalid_argument("Camera: Values must be positive");
        } else if(FOV < 0.0f || FOV > 180.0f) {
            throw std::invalid_argument("Camera: Invalid FOV value");
        }
        fov = FOV;
    }


__host__ void Camera::check() {
    initializeWindow();
    if(width < 0.0f) {
        throw std::invalid_argument("Camera check: Width must be positive");
    } else if(height < 0.0f) {
        throw std::invalid_argument("Camera check: Height must be positive");
    } else if(samples < 0.0f) {
        throw std::invalid_argument("Camera check: Sample count must be positive");
    } else if(depth < 0.0f) {
        throw std::invalid_argument("Camera check: Recursion depth must be positive");
    } else if(direction.lengthSquared() < epsilon) {
        throw std::invalid_argument("Camera check: Camera direction undefined");
    } else if(up.lengthSquared() < epsilon) {
        throw std::invalid_argument("Camera check: Camera up direction undefined");
    } else if(fov < 0.0f || fov > 180.0f) {
        throw std::invalid_argument("Camera check: field of view should be in [0, 180] degrees");
    } else if((direction - up).length() < epsilon) {
        throw std::invalid_argument("Camera check: Direction and up vectors are too similar");
    }
}



__host__ void Camera::initializeWindow() {
    direction = unitVec(direction); up = unitVec(up);
    window = initialRays(eye, direction, 1.0f, up, height, width, fov);
}
