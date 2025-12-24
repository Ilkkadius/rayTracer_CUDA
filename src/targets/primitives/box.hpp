#ifndef BOX_CUDA_HPP
#define BOX_CUDA_HPP

#include "target.hpp"

class Box : public Target {
public:
    Matrix invmat;
    Vector3D center;

    __device__ Box(const Vector3D& minimumCorner, const Vector3D& maximumCorner, const Vector3D& color_, float emissivity_ = 0.0f);

    __device__ bool rayCollision(const Ray& ray, HitInfo* hit) const;
    __device__ int allCollisions(const Ray& ray, HitInfo* hitlist) const;
    __device__ int maxCollisions() const {return 2;}

    __device__ Vector3D centroid() const;
    
    __device__ void translate(const Vector3D& vec);
    __device__ void translate(float x, float y, float z);

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);

    __device__ Vector3D minBox() const;
    __device__ Vector3D maxBox() const;

};

#endif