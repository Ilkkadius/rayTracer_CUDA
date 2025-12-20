#ifndef SCATTERING_CUDA_HPP
#define SCATTERING_CUDA_HPP

#include "vector3D.hpp"
#include "rayf.hpp"
#include "targetf.hpp"


class HitInfo{
public:
    Vector3D point, normal, rayDir;
    float t;
    Target* target;

    __device__ HitInfo();

    __device__ HitInfo(const Vector3D& rayDirection, const Vector3D& hitPoint, 
                        const Vector3D& surfaceNormal, float t_, Target* target_);

};

#endif