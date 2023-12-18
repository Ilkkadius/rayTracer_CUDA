#ifndef COMPOUND_CUDA_HPP
#define COMPOUND_CUDA_HPP

#include "targetf.hpp"

class Compound{
public:
    Target** targets;
    size_t size, capacity;

    __device__ Compound(size_t N) : capacity(N), size(0) {
        targets = new Target*[N];
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
            release();
        }
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
        Sphere* S = new Sphere(Vector3D(4, -0.5, 2), 0.5f);
        Target* T = new Target(S);
        add(T);
    }



};


#endif