#ifndef TARGET_CUDA_HPP
#define TARGET_CUDA_HPP

#include "vector3D.hpp"
#include "geometria.hpp"

class Target{
public:

Shape* shape;
Vector3D color;
float emissivity;

    __device__ Target(Shape* shape_, const Vector3D& color_, float emissivity_ = 0.0f);

    __device__ float collision(const Ray& ray) const;

    __device__ Vector3D normal(const Vector3D& point) const;

    __device__ Vector3D centroid() const;

    __device__ void translate(const Vector3D& vec);
    __device__ void translate(float x, float y, float z);

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);

    __device__ bool isRadiant() const;

    __device__ Vector3D emission() const;

    __device__ Vector3D minBox() const;

    __device__ Vector3D maxBox() const;


};


#endif