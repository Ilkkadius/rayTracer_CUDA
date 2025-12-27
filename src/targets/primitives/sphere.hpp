#ifndef SPHERE_CUDA_HPP
#define SPHERE_CUDA_HPP

#include "target.hpp"

class Sphere : public Target {
public:
    Vector3D center;
    float radius;

    __device__ Sphere(const Vector3D& center_, float radius_, const Vector3D& color_, float emissivity_ = 0.0f);

    __device__ bool rayCollision(const Ray& ray, HitInfo* hit) const;
    __device__ int allCollisions(const Ray& ray, HitInfo* hitlist) const;
    __device__ int maxCollisions() const {return 2;}

    __device__ Vector3D centroid() const;
    
    __device__ void translate(const Vector3D& vec);
    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);
    __device__ void affine(const Matrix& A, const Vector3D& b);

    __device__ Vector3D minBox() const;
    __device__ Vector3D maxBox() const;

};

#endif