#ifndef CAMERA_CUDA_HPP
#define CAMERA_CUDA_HPP

#include <stdexcept>
#include <vector>
#include <memory>

#include "rayf.hpp"

class Camera {
public:

    int width;
    int height;
    int samples;
    int depth;

    Vector3D eye;
    Vector3D direction;
    Vector3D up;

    float windowHeight;


    /**
     * @brief Construct a new Camera object
     * 
     */
    __host__ Camera() : width(10), height(10), samples(10),
        eye(Vector3D(0,0,0)), direction(Vector3D(1,0,0)), up(Vector3D(0,0,1)),
        windowHeight(1.8), 
        window(Vector3D(), Vector3D(), Vector3D(), Vector3D()) {}

    /**
     * @brief Construct a new Camera object
     * 
     * @param widthPixels   image width
     * @param heightPixels  image height
     */
    __host__ Camera(int widthPixels, int heightPixels) : width(widthPixels), height(heightPixels), samples(10),
        eye(Vector3D(0,0,0)), direction(Vector3D(1,0,0)), up(Vector3D(0,0,1)),
        windowHeight(1.8), window(Vector3D(), Vector3D(), Vector3D(), Vector3D()) {
            if(widthPixels < 0.0f || heightPixels < 0.0f) {
                throw std::invalid_argument("Values must be positive");
            }
        }

    /**
     * @brief Construct a new Camera object
     * 
     * @param widthPixels   image width
     * @param heightPixels  image height
     * @param FOV           field of view
     */
    __host__ Camera(int widthPixels, int heightPixels, float FOV) : width(widthPixels), height(heightPixels), 
        samples(10), eye(Vector3D(0,0,0)), direction(Vector3D(1,0,0)), up(Vector3D(0,0,1)),
        window(Vector3D(), Vector3D(), Vector3D(), Vector3D()) {
            if(widthPixels < 0.0f || heightPixels < 0.0f) {
                throw std::invalid_argument("Values must be positive");
            } else if(FOV < 0.0f || FOV > 180.0f) {
                throw std::invalid_argument("Invalid FOV value");
            }
            windowHeight = 2.0f*float(height)/float(width)*std::tan(FOV*M_PI/360.0f);
        }



    /**
     * @brief Set the field of view of the camera
     * 
     * @param FOV angle in degrees
     */
    __host__ void setFOV(float FOV) {
        if(FOV < 180.0f && FOV > 0.0f) {
            windowHeight = 2.0f*float(height)/float(width)*std::tan(FOV*M_PI/360.0f);
        } else {
            std::cout << "Error: Could not set given camera FOV" << std::endl;
        }
    }

    /**
     * @brief Check that all parameters of the camera are good for rendering
     * 
     */
    __host__ void check() {
        if(width < 0.0f) {
            throw std::invalid_argument("Width must be positive");
        } else if(height < 0.0f) {
            throw std::invalid_argument("Height must be positive");
        } else if(samples < 0.0f) {
            throw std::invalid_argument("Sample count must be positive");
        } else if(depth < 0.0f) {
            throw std::invalid_argument("Recursion depth must be positive");
        } else if(direction.lengthSquared() < epsilon) {
            throw std::invalid_argument("Camera direction undefined");
        } else if(up.lengthSquared() < epsilon) {
            throw std::invalid_argument("Camera up direction undefined");
        } else if(windowHeight < 0.0f) {
            throw std::invalid_argument("WindowHeight must be positive");
        } else if((direction - up).length() < epsilon) {
            throw std::invalid_argument("Direction and up vectors are too similar");
        }
    }


    WindowVectors window;

    __host__ void initializeWindow() {
        window = initialRays(eye, direction, 1.0f, up, height, width, windowHeight);
    }

};

#endif