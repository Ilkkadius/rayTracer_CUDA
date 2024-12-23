#ifndef ROTATIONS_CUDA_HPP
#define ROTATIONS_CUDA_HPP

#include <cmath>

#include "matrix3D.hpp"

class RotationX : public Matrix{
public:
    __device__ RotationX(float angle) : Matrix(Vector3D(1,0,0),
                                        Vector3D(0, cos(angle), sin(angle)),
                                        Vector3D(0, -sin(angle), cos(angle))) {}
};

class RotationY : public Matrix{
public:
    __device__ RotationY(float angle) : Matrix(Vector3D(cos(angle), 0, -sin(angle)),
                                        Vector3D(0, 1, 0),
                                        Vector3D(sin(angle), 0, cos(angle))) {}
};

class RotationZ : public Matrix{
public:
    __device__ RotationZ(float angle) : Matrix(Vector3D(cos(angle), sin(angle), 0),
                                        Vector3D(-sin(angle), cos(angle), 0),
                                        Vector3D(0, 0, 1)) {}
};

/**
 * @brief Generate a rotation matrix of given angle w.r.t. the axis c 
 * 
 * @param a angle (in radians)
 * @param c axis (x, y or z)
 * @return Matrix 
 */
__device__ Matrix generateRotation(float a, char c) {
    if(c == 'x' || c == 'X') {
        return RotationX(a);
    } else if(c == 'y' || c == 'Y') {
        return RotationY(a);
    } else if(c == 'z' || c == 'Z') {
        return RotationZ(a);
    } else {
        printf("Error: Wrong rotation axis\n");
        return Matrix();
    }
}


/**
 * @brief Generate a rotation matrix from given angle w.r.t. the given axis
 * 
 * @param a     angle for rotation (in radians)
 * @param axis  vector defining the axis of rotation
 * @return Matrix 
 */
__host__ __device__ Matrix generateRotation(float a, const Vector3D& axis) {
    Vector3D k = unitVec(axis);
    Matrix K(Vector3D(0, k.z, -k.y), Vector3D(-k.z, 0, k.x), Vector3D(k.y, -k.x, 0));
    return Matrix() + K * sin(a) + (1 - cos(a)) * K * K;
}



#endif