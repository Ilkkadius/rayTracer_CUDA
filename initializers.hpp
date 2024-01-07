#ifndef INITIALIZERS_CUDA_HPP
#define INITIALIZERS_CUDA_HPP

#include <cuda_runtime.h>

#include "targetf.hpp"
#include "backgroundsf.hpp"
#include "compoundf.hpp"
#include "auxiliaryf.hpp"
#include "scenesf.hpp"
#include "cameraf.hpp"
#include "kernelSet.hpp"

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

/**
 * @brief Methods for the CPU (host) to use before rendering
 * 
 */
namespace Initializer{

    __host__ void Window(WindowVectors* window, Camera& cam) {
        cam.check();
        CHECK(cudaMalloc(&window, sizeof(WindowVectors)));
        CHECK(cudaMemcpy(window, &cam.window, sizeof(WindowVectors), cudaMemcpyHostToDevice));
        std::cout << "Window ready" << std::endl;
    }

    __host__ void Background(BackgroundColor*** background) {
        CHECK(cudaMalloc(background, sizeof(BackgroundColor*)));
        initializeBG<<<1,1>>>(*background);
        CHECK(cudaDeviceSynchronize());
        std::cout << "Background ready" << std::endl;
    }

    __host__ void RandomStates(curandState** randState, int width, int height, dim3 blocks, dim3 threads) {
        CHECK(cudaMalloc(randState, width*height*sizeof(curandState)));
        initializeRand<<<blocks, threads>>>(*randState, width, height);
        CHECK(cudaDeviceSynchronize());
        std::cout << "Random states generated" << std::endl;
    }

    __host__ void BVH(BVHTree*** tree, targetList** list) {
        CHECK(cudaMalloc(tree, sizeof(BVHTree*)));
        CHECK(cudaDeviceSynchronize());
        buildBVH<<<1,1>>>(list, *tree);
        CHECK(cudaDeviceSynchronize());
    }

    __host__ void CompoundsToTargets(Compound** compounds, size_t compoundCount, targetList** list, Shape** shapes) {
        addCompoundsToTargetlist<<<1,1>>>(compounds, compoundCount, list, shapes);
        CHECK(cudaDeviceSynchronize());
    }
}




#endif