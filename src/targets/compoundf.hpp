#ifndef COMPOUND_CUDA_HPP
#define COMPOUND_CUDA_HPP

#include "targetList.hpp"
#include "auxiliaryf.hpp"

class Compound{
public:
    Target** targets;
    size_t size, capacity;

    __device__ Compound(size_t N);

    __device__ Vector3D centroid() const;

    __device__ void translate(const Vector3D& vec);
    __device__ void translate(float x, float y, float z);

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);

    __device__ void rotate(float angle, const Vector3D& axis);


    __device__ void release();

    /**
     * @brief Copy Compound data to a targetList and release the Compound
     * 
     * @param list 
     * @param shapes 
     * @return __device__ 
     */
    __device__ void copyToList(TargetList* list, Shape** shapes);

    __device__ void mergeCompound(Compound& c);

    __device__ bool add(Target* target);

};

class compoundTest : public Compound{
public:

    __device__ compoundTest();

private:
    __device__ void generator();

};

/**
 * @brief Tetrahedron compound, 4 triangles
 * 
 */
class Tetrahedron : public Compound{
public:
    
    __device__ Tetrahedron(const Vector3D& center, float radius, const Vector3D& color);

private:
    __device__ void makeTetrahedron(const Vector3D& center, float radius, const Vector3D& color);
};

/**
 * @brief Icosahedron compound, 20 triangles
 * 
 */
class Icosahedron : public Compound{
public:
    __device__ Icosahedron(const Vector3D& center, float radius, const Vector3D& color);

private:
    __device__ void makeIcosahedron(const Vector3D& center, float radius, const Vector3D& color);
};

/**
 * @brief Pentagon compound, 3 triangles
 * 
 */
class Pentagon : public Compound{
public:
    __device__ Pentagon(const Vector3D& v1, const Vector3D& v2, const Vector3D& v3, const Vector3D& v4, const Vector3D& v5, const Vector3D& color);

private:
    __device__ void makePentagon(const Vector3D& v1, const Vector3D& v2, const Vector3D& v3, 
                                    const Vector3D& v4, const Vector3D& v5, const Vector3D& color);
};

/**
 * @brief Dodecahedron compound, 36 triangles
 * 
 */
class Dodecahedron : public Compound{
public:
    __device__ Dodecahedron(const Vector3D& center, float radius, const Vector3D& color);

private:
    __device__ void makeDodecahedron(const Vector3D& center, float radius, const Vector3D& color);
};

/**
 * @brief Octahedron compound, 8 triangles
 * 
 */
class Octahedron : public Compound{
public:
    __device__ Octahedron(const Vector3D& center, float radius, const Vector3D& color);

private:
    __device__ void makeOctahedron(const Vector3D& center, float radius, const Vector3D& color);
};

/**
 * @brief Cube compound, 12 triangles
 * 
 */
class Cube : public Compound{
public:
    __device__ Cube(const Vector3D& center, float radius, const Vector3D& color);

private:
    __device__ void buildCube(const Vector3D& center, float radius, const Vector3D& color);
};




#endif