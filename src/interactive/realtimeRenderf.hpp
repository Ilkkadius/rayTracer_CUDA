#ifndef REAL_TIME_RAY_TRACINGF_HPP
#define REAL_TIME_RAY_TRACINGF_HPP

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <memory>

#include "geometria.hpp"
#include "rayf.hpp"
#include "tracerf.hpp"
#include "rotationf.hpp"
#include "cameraf.hpp"
#include "kernelSet.hpp"

namespace cameraMove{

    void left(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void right(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void front(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void back(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void up(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void down(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void rotateDirection(const sf::Vector2i& diff, Vector3D& direction, Vector3D& up, int width, int height, double speed = 1.0);

    void roll(const Vector3D& direction, Vector3D& up, double xangle);
}

namespace realtimeRender{
    void centerMouse(sf::Vector2i center, sf::Window& window);

    void detectKey(const sf::Event& event, Vector3D& eye, Vector3D& direction, Vector3D& up, bool& changes, int& speed);

    __host__ void simpleRender(sf::Uint8* pixels, Camera& cam, TargetList** targetPtr, BackgroundColor** background, curandState* randState);

    __host__ void simpleRender(sf::Uint8* pixels, Camera& cam, BVHTree** tree, BackgroundColor** background, curandState* randState);

    __host__ void startCamera(Camera& cam, TargetList** targetPtr, BackgroundColor** background, 
                                curandState* randState, Vector3D& eye, Vector3D& direction, Vector3D& up);

    __host__ void startCamera(Camera& cam, BVHTree** tree, BackgroundColor** background, curandState* randState, 
                                Vector3D& eye, Vector3D& direction, Vector3D& up);
}



#endif