#include "rotationf.hpp"

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

__host__ __device__ Matrix generateRotation(float a, const Vector3D& axis) {
    Vector3D k = unitVec(axis);
    Matrix K(Vector3D(0, k.z, -k.y), Vector3D(-k.z, 0, k.x), Vector3D(k.y, -k.x, 0));
    return Matrix() + K * sin(a) + (1 - cos(a)) * K * K;
}
