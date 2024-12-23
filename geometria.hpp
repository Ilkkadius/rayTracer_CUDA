#ifndef GEOMETRIA_CUDA_HPP
#define GEOMETRIA_CUDA_HPP

#include <cuda_runtime.h>

#include "vector3D.hpp"
#include "rayf.hpp"
#include "auxiliaryf.hpp"

class Shape{
public:
    __device__ virtual Vector3D normal(const Vector3D& point) const = 0;

    __device__ virtual float rayCollision(const Ray& ray) const = 0;

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

};

#endif