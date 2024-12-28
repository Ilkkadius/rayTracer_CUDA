#include "matrix3D.hpp"


__host__ __device__ Matrix::Matrix() : c1(Vector3D(1,0,0)), c2(Vector3D(0,1,0)), c3(Vector3D(0,0,1)) {}

__host__ __device__ Matrix::Matrix(const Vector3D& A, const Vector3D& B, const Vector3D& C)
            : c1(A), c2(B), c3(C) {}

__host__ __device__ Matrix::Matrix(const Matrix& mat) : c1(mat.c1), c2(mat.c2), c3(mat.c3) {}

__host__ __device__ Matrix& Matrix::operator=(const Matrix& mat) {
    c1 = mat.c1; c2 = mat.c2; c3 = mat.c3;
    return *this;
}

__host__ __device__ Matrix Matrix::operator-() const {return Matrix(-c1, -c2, -c3);}

__host__ __device__ Vector3D Matrix::r1() const {
    return Vector3D(c1.x, c2.x, c3.x);
}

__host__ __device__ Vector3D Matrix::r2() const {
    return Vector3D(c1.y, c2.y, c3.y);
}

__host__ __device__ Vector3D Matrix::r3() const {
    return Vector3D(c1.z, c2.z, c3.z);
}

__host__ __device__ Matrix operator*(const Matrix& A, const Matrix& B) {
    Vector3D c1(Dot(A.r1(), B.c1),Dot(A.r2(), B.c1),Dot(A.r3(), B.c1));
    Vector3D c2(Dot(A.r1(), B.c2),Dot(A.r2(), B.c2),Dot(A.r3(), B.c2));
    Vector3D c3(Dot(A.r1(), B.c3),Dot(A.r2(), B.c3),Dot(A.r3(), B.c3));
    return Matrix(c1, c2, c3);
}

__host__ __device__ Vector3D operator*(const Matrix& mat, const Vector3D& vec) {
    return Vector3D(Dot(mat.r1(), vec), Dot(mat.r2(), vec), Dot(mat.r3(), vec));
}

__host__ __device__ Matrix operator*(float a, const Matrix& mat) {
    return Matrix(a*mat.c1, a*mat.c2, a*mat.c3);
}

__host__ __device__ Matrix operator*(const Matrix& mat, float a) {
    return a*mat;
}

__host__ __device__ Matrix operator+(const Matrix& A, const Matrix& B) {
    return Matrix(A.c1 + B.c1, A.c2 + B.c2, A.c3 + B.c3);
}

__host__ __device__ Matrix operator-(const Matrix& A, const Matrix& B) {
    return A + (-B);
}

