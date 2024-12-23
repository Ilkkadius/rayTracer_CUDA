#ifndef SCATTERING_CUDA_HPP
#define SCATTERING_CUDA_HPP

#include "vector3D.hpp"
#include "rayf.hpp"


class HitInfo{
public:
    Vector3D point, normal;
    Ray ray;
    float t;

    __device__ HitInfo() : ray(Ray()), point(Vector3D()), normal(Vector3D()), t(-1.0f) {}

    __device__ HitInfo(const Ray& ray_, const Vector3D& point_, const Vector3D& normal_, float t_)
     : ray(ray_), point(point_), t(t_) {
        if(Dot(ray_.dir, normal_) > 0.0f) { // Make sure the normal points outwards!
            normal = -normal_;
        } else {
            normal = normal_;
        }
     }
    

};

#endif