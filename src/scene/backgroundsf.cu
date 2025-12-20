#include "backgroundsf.hpp"


__device__ dayTime::dayTime(const Vector3D& color) : color_(color) {}

__device__ Vector3D dayTime::colorize(const Ray& ray) const {
    float a = 0.5f*(ray.dir.z + 1.0f);
    return (1.0f-a)*Vector3D(1, 1, 1) + a*color_;
}

__device__ nightTime::nightTime(const Vector3D& color) : color_(color) {}

__device__ Vector3D nightTime::colorize(const Ray& ray) const {
    auto a = 0.5*(ray.dir.z + 1.0);
    return (1.0-a)*color_ + a*Vector3D(0, 0, 0);
}
