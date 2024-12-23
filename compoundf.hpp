#ifndef COMPOUND_CUDA_HPP
#define COMPOUND_CUDA_HPP

#include "targetf.hpp"
#include "auxiliaryf.hpp"

class Compound{
public:
    Target** targets;
    size_t size, capacity;

    __device__ Compound(size_t N) : capacity(N), size(0) {
        targets = new Target*[N];
    }

    __device__ Vector3D centroid() const {
        Vector3D center;
        for(int i = 0; i < size; i++) {
            center += targets[i]->centroid();
        }
        return center/float(size);
    }

    __device__ void translate(const Vector3D& vec) {
        for(int i = 0; i < size; i++) {
            targets[i]->translate(vec);
        }
    }
    __device__ void translate(float x, float y, float z) {
        for(int i = 0; i < size; i++) {
            targets[i]->translate(x,y,z);
        }
    }

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
        for(int i = 0; i < size; i++) {
            targets[i]->rotate(angle, axis, axisPos);
        }
    }

    __device__ void rotate(float angle, const Vector3D& axis) {
        Vector3D center = centroid();
        for(int i = 0; i < size; i++) {
            targets[i]->rotate(angle, axis, center);
        }
    }

    __device__ void release() {
        for(size_t i = 0; i < capacity; i++) {
            if(targets[i]) {
                delete targets[i]->shape;
                delete targets[i];
            }
        }
        delete[] targets;
    }

    /**
     * @brief Copy Compound data to a targetList and release the Compound
     * 
     * @param list 
     * @param shapes 
     * @return __device__ 
     */
    __device__ void copyToList(targetList* list, Shape** shapes) {
        for(size_t i = 0; i < size; i++) {
            if(list->size < list->capacity) {
                list->targets[list->size] = targets[i];
                shapes[list->size] = targets[i]->shape;
                list->size++;
                targets[i] = NULL;
            }
        }
        release();
    }

    __device__ bool add(Target* target) {
        if(size < capacity) {
            targets[size] = target;
            size++;
            return true;
        } else {
            delete target->shape;
            delete target;
            return false;
        }
    }

};

class compoundTest : public Compound{
public:

    __device__ compoundTest() : Compound(1) {
        generator();
    }

private:
    __device__ void generator() {
        Sphere* S = new Sphere(Vector3D(6, -0.5, 2), 0.5f);
        Target* T = new Target(S);
        add(T);
    }



};

class Icosahedron : public Compound{
public:
    __device__ Icosahedron(const Vector3D& center, float radius)
                : Compound(20) {
                    makeIcosahedron(center, radius);
                }

private:
    __device__ void makeIcosahedron(const Vector3D& center, float radius) {
        float R = radius;

        float permutations[4][2] = {{1.0f, phi}, {-1.0f, phi}, {1.0f, -phi}, {-1.0f, -phi}};
        Vector3D vertices[12];

        for(int i = 0; i < 4; i++) {
            float p0 = permutations[i][0];
            float p1 = permutations[i][1];
            vertices[3*i] = Vector3D(0, p0, p1);
            vertices[3*i+1] = Vector3D(p0, p1, 0);
            vertices[3*i+2] = Vector3D(p1, 0, p0);
        }

        
        
        int facePermutations[20][3] = { 
            {0,3,8},    {0,4,8},    {4,8,11},   {8,10,11},  {3,8,10},
            {1,2,5},    {1,5,6},    {5,6,9},    {5,7,9},    {2,5,7},
            {0,1,2},    {1,4,6},    {6,9,11},   {7,9,10},   {2,3,7},
            {0,1,4},    {4,6,11},   {9,10,11},  {3,7,10},   {0,2,3}
            };
        
        for(int i = 0; i < 20; i++) {
            int v1 = facePermutations[i][0], v2 = facePermutations[i][1], v3 = facePermutations[i][2];
            Triangle* T = new Triangle(R*vertices[v1] + center, R*vertices[v2] + center, R*vertices[v3] + center);
            add(new Target(T));
        }
    }
};


#endif