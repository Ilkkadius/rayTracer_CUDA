#include "initializers.hpp"

/*
    printf("\033[31mDEBUG: id\033[0m\n");
    int size = (**list).size;
    printf("Number of objects: %d\n", size);
    for(int i = 0; i < size; i++) {
        Target* t1 = (**list).targets[i];
        Target* t2 = targets[i];
        Shape* s = shapes[i];
        printf("Target from list: %p, from targets: %p and shape: %p\n", t1, t2, s);
    }
*/




__device__ void init::createTargets(TargetList** list, int capacity) {
    int N = capacity;
    switch(5) {
        case 1:
            Scene::Platon(*list, N);
            break;
        case 2:
            Scene::testScene(*list, N);
            break;
        case 3:
            Scene::CSG(*list, N);
            break;
        case 4:
            Scene::CSG2(*list, N);
            break;
        case 5:
            Scene::light(*list, N);
            break;
        default:
            Scene::empty(*list, N);
            break;
    }
}

__device__ BackgroundColor* init::createBackground(backgroundType type) {
    switch(type) {
        case backgroundType::NIGHT:
            return new nightTime();
        case backgroundType::DAY:
            return new dayTime();
        default:
            return new darkness();
    }
}

