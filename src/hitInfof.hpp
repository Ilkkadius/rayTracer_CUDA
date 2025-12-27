#ifndef SCATTERING_CUDA_HPP
#define SCATTERING_CUDA_HPP

#include "vector3D.hpp"
#include "rayf.hpp"
#include "target.hpp"

/**
 * @brief Records a ray-Target collision. The normal MUST be normalized!
 * 
 */
class HitInfo{
public:
    Vector3D normal;
    Vector3D color;
    float emission;
    float t;
    
    __device__ HitInfo() : normal(Vector3D()), color(Vector3D(1.0f,1.0f,1.0f)), emission(0.0f), t(-1.0f) {}

    __device__ HitInfo(const Vector3D& normal_, const Vector3D& color_, float emission_, float t_) : normal(normal_), color(color_), emission(emission_), t(t_) {}
};

#endif