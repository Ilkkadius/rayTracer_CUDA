#ifndef GEOMETRIA_CUDA_HPP
#define GEOMETRIA_CUDA_HPP

#include <cuda_runtime.h>

#include "vector3D.hpp"
#include "rayf.hpp"
#include "auxiliaryf.hpp"
#include "rotationf.hpp"

class Shape{
public:
    Vector3D minBox, maxBox;

    __device__ virtual Vector3D normal(const Vector3D& point) const = 0;

    __device__ virtual float rayCollision(const Ray& ray) const = 0;

    __device__ virtual Vector3D centroid() const = 0;

    __device__ virtual void translate(const Vector3D& vec) = 0;
    __device__ virtual void translate(float x, float y, float z) = 0;

    __device__ virtual void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) = 0;

protected:

    __device__ Vector3D rotateVec(const Vector3D& vec, double angle, 
                                    const Vector3D& axis, const Vector3D& axisPos) {
        Matrix rot = generateRotation(angle, axis);
        return rot * (vec - axisPos) + axisPos;
    }

};

class Sphere : public Shape{
public:
    Vector3D center;
    float radius;

    __device__ Sphere(const Vector3D& center_, float radius_);

    __device__ Vector3D normal(const Vector3D& point) const;

    __device__ float rayCollision(const Ray& ray) const;

    __device__ Vector3D centroid() const;

    __device__ void translate(const Vector3D& vec);
    __device__ void translate(float x, float y, float z);

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);

private:

    __device__ void buildBox();

};

class Triangle : public Shape{
public:
    Vector3D v0, v1, v2;

    __device__ Triangle(const Vector3D& vertex0, const Vector3D& vertex1, const Vector3D& vertex2);

    __device__ Vector3D normal(const Vector3D& point) const;

    __device__ float rayCollision(const Ray& ray) const;

    __device__ Vector3D centroid() const;

    __device__ void translate(const Vector3D& vec);
    __device__ void translate(float x, float y, float z);

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);

private:

    __device__ void buildBox();

};

#endif