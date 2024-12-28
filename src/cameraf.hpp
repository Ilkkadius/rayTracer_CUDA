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
    __host__ Camera();

    /**
     * @brief Construct a new Camera object
     * 
     * @param widthPixels   image width
     * @param heightPixels  image height
     */
    __host__ Camera(int widthPixels, int heightPixels);

    /**
     * @brief Construct a new Camera object
     * 
     * @param widthPixels   image width
     * @param heightPixels  image height
     * @param FOV           field of view
     */
    __host__ Camera(int widthPixels, int heightPixels, float FOV);

    /**
     * @brief Set the field of view of the camera
     * 
     * @param FOV angle in degrees
     */
    __host__ void setFOV(float FOV);

    /**
     * @brief Check that all parameters of the camera are good for rendering
     * 
     */
    __host__ void check();


    WindowVectors window;

    __host__ void initializeWindow();
};

#endif