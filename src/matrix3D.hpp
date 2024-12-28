#ifndef MATRICES_CUDA_HPP
#define MATRICES_CUDA_HPP

#include "vector3D.hpp"

class Matrix{
public:
    Vector3D c1, c2, c3;

    __host__ __device__ Matrix();

    __host__ __device__ Matrix(const Vector3D& A, const Vector3D& B, const Vector3D& C);

    __host__ __device__ Matrix(const Matrix& mat);

    __host__ __device__ Matrix& operator=(const Matrix& mat);

    __host__ __device__ Matrix operator-() const;
    /**
     * @brief Return first row of the matrix
     * 
     * @return Vector3D
     */
    __host__ __device__ Vector3D r1() const;

    /**
     * @brief Return second row of the matrix
     * 
     * @return Vector3D
     */
    __host__ __device__ Vector3D r2() const;

    /**
     * @brief Return third row of the matrix
     * 
     * @return Vector3D
     */
    __host__ __device__ Vector3D r3() const;

};

/**
 * @brief Matrix-matrix product
 * 
 * @param A 
 * @param B 
 * @return Matrix 
 */
__host__ __device__ Matrix operator*(const Matrix& A, const Matrix& B);

/**
 * @brief Matrix-vector product
 * 
 * @param mat 
 * @param vec 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator*(const Matrix& mat, const Vector3D& vec);

__host__ __device__ Matrix operator*(float a, const Matrix& mat);

__host__ __device__ Matrix operator*(const Matrix& mat, float a);

__host__ __device__ Matrix operator+(const Matrix& A, const Matrix& B);

__host__ __device__ Matrix operator-(const Matrix& A, const Matrix& B);

#endif