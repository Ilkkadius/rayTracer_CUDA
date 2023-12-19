#ifndef SCATTERING_CUDA_HPP
#define SCATTERING_CUDA_HPP

#include "vector3D.hpp"
#include "rayf.hpp"


class HitInfo{
public:
    Vector3D point, normal, rayDir;
    float t;
    Target* target;

    __device__ HitInfo() : rayDir(Vector3D()), point(Vector3D()), normal(Vector3D()), t(-1.0f) {}

    __device__ HitInfo(const Vector3D& rayDirection, const Vector3D& hitPoint, const Vector3D& surfaceNormal, float t_, Target* target_)
    : rayDir(rayDirection), point(hitPoint), t(t_), target(target_) {
        if(Dot(rayDirection, surfaceNormal) > 0.0f) { // Make sure the normal points outwards!
            normal = -surfaceNormal;
        } else {
            normal = surfaceNormal;
        }
    }
    

};

#endif