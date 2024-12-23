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
                targets[i] = NULL;
            }
        }
        if(targets) {
            delete[] targets;
            targets = NULL;
        }
    }

    /**
     * @brief Copy Compound data to a targetList and release the Compound
     * 
     * @param list 
     * @param shapes 
     * @return __device__ 
     */
    __device__ void copyToList(targetList* list, Shape** shapes) {
        for(int i = 0; i < size; i++) {
            if(list->size < list->capacity) {
                list->targets[list->size] = targets[i];
                shapes[list->size] = targets[i]->shape;
                list->size++;
                targets[i] = NULL;
            }
        }
        release();
    }

    __device__ void mergeCompound(Compound& c) {
        for(int i = 0; i < c.size; i++) {
            if(size < capacity) {
                targets[size] = c.targets[i];
                size++;
                c.targets[i] = NULL;
            }
        }
        c.release();
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

class Tetrahedron : public Compound{
public:
    
    __device__ Tetrahedron(const Vector3D& center, float radius)
                : Compound(4) {
                    makeTetrahedron(center, radius);
                }

private:
    __device__ void makeTetrahedron(const Vector3D& center, float radius) {
        float R = radius, s = 0.70711f; // = 1/sqrt(2)

        Vector3D vertices[4] = {
            Vector3D(1, 0, -s), Vector3D(-1,0,-s), Vector3D(0,1,s), Vector3D(0,-1,s)
        };

        int faces[4][3] = { {0,1,2}, {0,2,3}, {0,1,3}, {1,2,3} };

        for(int i = 0; i < 4; i++) {
            int v1 = faces[i][0], v2 = faces[i][1], v3 = faces[i][2];
            Triangle* T = new Triangle(R*vertices[v1] + center, R*vertices[v2] + center, R*vertices[v3] + center);
            add(new Target(T));
        }
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

class Pentagon : public Compound{
public:
    __device__ Pentagon(const Vector3D& v1, const Vector3D& v2, const Vector3D& v3, const Vector3D& v4, const Vector3D& v5)
            : Compound(3) {makePentagon(v1, v2, v3, v4, v5);}

private:
    __device__ void makePentagon(const Vector3D& v1, const Vector3D& v2, const Vector3D& v3, const Vector3D& v4, const Vector3D& v5) {
        Triangle* T1 = new Triangle(v1, v2, v3);
        Triangle* T2 = new Triangle(v1, v3, v4);
        Triangle* T3 = new Triangle(v1, v4, v5);
        add(new Target(T1)); add(new Target(T2)); add(new Target(T3));
    }
};

class Dodecahedron : public Compound{
public:
    __device__ Dodecahedron(const Vector3D& center, float radius)
                : Compound(36) {makeDodecahedron(center, radius);}

private:
    __device__ void makeDodecahedron(const Vector3D& center, float radius) {
        float R = radius;

        float permutations[4][2] = {{1.0f, 1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f}, {-1.0f, -1.0f}};
        Vector3D vertices[20];

        for(int i = 0; i < 4; i++) { // Generate each vertex of the dodecahedron (source: Wikipedia)
            float p0 = permutations[i][0], p1 = permutations[i][1];
            vertices[5*i] = Vector3D(0.0f, p0*phi, p1/phi);
            vertices[5*i+1] = Vector3D(p0/phi, 0.0f, p1*phi);
            vertices[5*i+2] = Vector3D(p0*phi, p1/phi, 0.0f);
            vertices[5*i+3] = Vector3D(1.0f, p0, p1);
            vertices[5*i+4] = Vector3D(-1.0f, p0, p1);
        }
        
        int faces[12][5] = { // These numbers are found (painfully) by hand
            {6,4,7,17,9}, {7,14,16,19,17}, 
            {4,0,10,14,7}, {9,17,19,15,5},
            {10,14,16,11,13}, {15,18,11,16,19}, 
            {1,3,0,4,6}, {1,8,5,9,6},
            {3,0,10,13,2}, {8,12,18,15,5}, 
            {1,3,2,12,8}, {2,13,11,18,12}
        };  // Indices of vertices in the vector "vertices" that form a single pentagon

        for(int i = 0; i < 12; i++) {
            int vv0 = faces[i][0], vv1 = faces[i][1], vv2 = faces[i][2], vv3 = faces[i][3], vv4 = faces[i][4];
            Vector3D v1(R*vertices[vv0] + center), v2(R*vertices[vv1] + center), 
                    v3(R*vertices[vv2] + center), v4(R*vertices[vv3] + center), v5(R*vertices[vv4] + center);
            Pentagon p(v1, v2, v3, v4, v5);
            mergeCompound(p);
        }
    }
};

class Octahedron : public Compound{
public:
    __device__ Octahedron(const Vector3D& center, float radius)
                : Compound(8) {makeOctahedron(center, radius);}

private:
    __device__ void makeOctahedron(const Vector3D& center, float radius) {
        float R = radius;

        Vector3D vertices[6] = {
            Vector3D(1,0,0), Vector3D(0,1,0), Vector3D(0,0,1), Vector3D(-1,0,0), Vector3D(0,-1,0), Vector3D(0,0,-1)
        };

        int faces[8][3] = {
            {0,1,2}, {1,2,3}, {2,3,4}, {0,2,4}, {0,1,5}, {1,3,5}, {3,4,5}, {0,4,5}
        };

        for(int i = 0; i < 8; i++) {
            int v1 = faces[i][0], v2 = faces[i][1], v3 = faces[i][2];
            Triangle* T = new Triangle(R*vertices[v1] + center, R*vertices[v2] + center, R*vertices[v3] + center);
            add(new Target(T));
        }
    }

};

class Cube : public Compound{
public:
    __device__ Cube(const Vector3D& center, float radius)
        : Compound(12) {buildCube(center, radius);}

private:
    __device__ void buildCube(const Vector3D& center, float radius) {
        float R = radius, d = 0.577350269f;

        Vector3D v[8] = { // Vertices
            Vector3D(d,d,d), Vector3D(-d,d,d), Vector3D(-d,-d,d), Vector3D(d,-d,d),
            Vector3D(d,d,-d), Vector3D(-d,d,-d), Vector3D(-d,-d,-d), Vector3D(d,-d,-d)
        };

        int f[6][4] = { // Faces
            {0,1,2,3}, {4,5,6,7}, {0,1,5,4}, {2,3,7,6}, {0,3,7,4}, {1,2,6,5}
        };

        for(int i = 0; i < 6; i++) {
            Vector3D q0 = R*v[f[i][0]] + center, q1 = R*v[f[i][1]] + center, q2 = R*v[f[i][2]] + center, q3 = R*v[f[i][3]] + center;
            Triangle* T1 = new Triangle(q0, q1, q2);
            Triangle* T2 = new Triangle(q2, q3, q0);
            add(new Target(T1)); add(new Target(T2));
        }
    }
};




#endif