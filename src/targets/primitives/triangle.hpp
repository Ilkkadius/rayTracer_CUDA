#ifndef TRIANGLE_CUDA_HPP
#define TRIANGLE_CUDA_HPP

#include "target.hpp"

class Triangle : public Target {
public:
    Vector3D v0, v1, v2;

    __device__ Triangle(const Vector3D& vertex0, const Vector3D& vertex1, const Vector3D& vertex2, const Vector3D& color_, float emission_ = 0.0f);

    __device__ bool rayCollision(const Ray& ray, HitInfo* hit) const;
    __device__ int allCollisions(const Ray& ray, HitInfo* hitlist) const {return 0;}
    __device__ int maxCollisions() const {return 0;}

    __device__ Vector3D centroid() const;
    
    __device__ void translate(const Vector3D& vec);
    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);
    __device__ void affine(const Matrix& A, const Vector3D& b);

    __device__ Vector3D minBox() const;
    __device__ Vector3D maxBox() const;

};

#endif