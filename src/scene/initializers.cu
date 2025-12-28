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




__device__ void init::createTargets(TargetList** list, sceneType type) {
    switch(type) {
        case sceneType::PLATON:
            Scene::Platon(*list);
            break;
        case sceneType::TEST:
            Scene::testScene(*list);
            break;
        case sceneType::CSG:
            Scene::CSG(*list);
            break;
        case sceneType::CSG2:
            Scene::CSG2(*list);
            break;
        case sceneType::LIGHT:
            Scene::light(*list);
            break;
        default:
            Scene::empty(*list);
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

