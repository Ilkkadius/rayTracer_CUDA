#ifndef GEOMETRIA_CUDA_HPP
#define GEOMETRIA_CUDA_HPP

#include <cuda_runtime.h>

#include "vector3D.hpp"
#include "rayf.hpp"
#include "auxiliaryf.hpp"
#include "rotationf.hpp"

class Shape{
public:
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

    __device__ Sphere(const Vector3D& center_, float radius_) : center(center_), radius(radius_) {}

    __device__ Vector3D normal(const Vector3D& point) const {
        return unitVec(point - center);
    }

    __device__ float rayCollision(const Ray& ray) const {
        Vector3D v = ray.pos - center;
        float vd = Dot(v, ray.dir);
        float disc = vd * vd - (v.lengthSquared() - radius * radius);
        if(disc < 0.0f) {
            return -1.0f;
        }
        float root = sqrtf(disc);
        float res = -vd - root;
        if(res < 0.0f) {
            res = -vd + root;
            if(res < 0.0f) {
                return -1.0f;
            }
        }
        return res;        
    }

    __device__ Vector3D centroid() const {return center;}

    __device__ void translate(const Vector3D& vec) {
        center += vec;
    }
    __device__ void translate(float x, float y, float z) {
        translate(Vector3D(x,y,z));
    }

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
        center = rotateVec(center, angle, axis, axisPos);
    }



};

class Triangle : public Shape{
public:
    Vector3D v0, v1, v2;

    __device__ Triangle(const Vector3D& vertex0, const Vector3D& vertex1, const Vector3D& vertex2) : v0(vertex0), v1(vertex1), v2(vertex2) {}

    __device__ Vector3D normal(const Vector3D& point) const {
        return unitVec(Cross(v1-v0, v2-v0));
    }

    __device__ float rayCollision(const Ray& ray) const {
        Vector3D edge1, edge2, h, s, q;
        float a, f, u, v;
        edge1 = v1 - v0;
        edge2 = v2 - v0;
        h = Cross(ray.heading(), edge2);
        a = Dot(edge1, h);

        if (a > -epsilon && a < epsilon)
            return -1.0f;

        f = 1.0f / a;
        s = ray.location() - v0;
        u = f * Dot(s,h);

        if (u < 0.0f || u > 1.0f)
            return -1.0f;

        q = Cross(s, edge1);
        v = f * Dot(ray.heading(), q);

        if (v < 0.0f || u + v > 1.0f)
            return -1.0f;

        float t = f * Dot(edge2, q);

        if (t > epsilon) {
            return t;
        }
        return -1.0f;
    }

    __device__ Vector3D centroid() const {
        return (v0 + v1 + v2)/3.0f;
    }

    __device__ void translate(const Vector3D& vec) {
        v0 += vec; v1 += vec; v2 += vec;
    }
    __device__ void translate(float x, float y, float z) {
        translate(Vector3D(x,y,z));
    }

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
        v0 = rotateVec(v0, angle, axis, axisPos);
        v1 = rotateVec(v1, angle, axis, axisPos);
        v2 = rotateVec(v2, angle, axis, axisPos);
    }


};

#endif