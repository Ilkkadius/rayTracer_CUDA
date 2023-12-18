#include <SFML/Graphics.hpp>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "initializers.hpp"
#include "vector3D.hpp"
#include "tracerf.hpp"
#include "rayf.hpp"
#include "backgroundsf.hpp"
#include "auxiliaryf.hpp"
#include "targetf.hpp"
#include "geometria.hpp"

// PPC
static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

//###############################################################

__global__ void render(sf::Uint8 *pixels, 
        int width, int height, 
        int depth, int samples,
        targetList** list,
        BackgroundColor** background, 
        WindowVectors* window, 
        curandState* randState) {
            
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;

    int idx = width * j + i;
    curandState rand = randState[idx];

    if(false && i % 100 == 0 && j % 75 == 0) {
        Vector3D vec = aux::randUnitVec(&rand);
        float l = vec.length();
        if(vec.x < 0 && vec.y < 0 && vec.z < 0) {
            printf("Vector (%f, %f, %f) of length %f\n", vec.x, vec.y, vec.z, l);
        }
    }

    

    Vector3D color = 255 * TracePixelRnd(window, i, j, list, depth, samples, *background, rand);

    

    pixels[4*idx] = color.x;
    pixels[4*idx + 1] = color.y;
    pixels[4*idx + 2] = color.z;
    pixels[4*idx + 3] = 255;
}

__global__ void initializeBG(BackgroundColor** background) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *background = createDayTime();
    }
}

__global__ void initializeTargets(Target** targets, targetList** list, Shape** shapes, int capacity) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        createTargets(targets, list, shapes, capacity);
    }
}

__global__ void initializeRand(curandState* randState, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;
    int idx = i + width * j;
    curand_init(1889, idx, 0, &randState[idx]);
}


__global__ void releaseBG(BackgroundColor** background) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        delete *background;
    }
}

__global__ void releaseTargets(Target** targets, targetList** list, Shape** shapes) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        targetList l = **list;
        for(int i = 0; i < l.size; i++) {
            delete *(targets + i);
            delete *(shapes + i);
        }
        delete *list;
    }
}

// ################################################################

// nvcc main.cu -o main -lsfml-graphics -lsfml-window -lsfml-system

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    // #################################
    // # SET PROGRAM RUN PARAMETERS
    // #################################

    int width = 1920, height = 1080;
    int depth = 5, samples = 100;
    int tx = 8, ty = 8;

    WindowVectors window = initialRays(Vector3D(0,0,0), Vector3D(1,0,0),
    1.0f, Vector3D(1,1,100), height, width, 0.8);

    // #################################
    // # LOAD DATA TO DEVICE
    // #################################

    dim3 blocks(divup(width, tx), divup(height, ty));
    dim3 threads(tx, ty);

    sf::Uint8 *pixels;// = new sf::Uint8[width*height*4];
    CHECK(cudaMallocManaged(&pixels, width*height*4));

    WindowVectors *cudaWindow = NULL;
    CHECK(cudaMalloc(&cudaWindow, sizeof(WindowVectors)));
    CHECK(cudaMemcpy(cudaWindow, &window, sizeof(WindowVectors), cudaMemcpyHostToDevice));
    std::cout << "Window ready" << std::endl;



    BackgroundColor** background_d;
    CHECK(cudaMalloc(&background_d, sizeof(BackgroundColor*)));
    initializeBG<<<1,1>>>(background_d);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Background ready" << std::endl;

    curandState *randState_d;
    CHECK(cudaMalloc(&randState_d, width*height*sizeof(curandState)));
    initializeRand<<<blocks, threads>>>(randState_d, width, height);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Random states generated" << std::endl;

    targetList** list; Target** targets; Shape** shapes; int N = 50;
    CHECK(cudaMalloc(&list, sizeof(targetList*)));
    CHECK(cudaMalloc(&targets, N*sizeof(Target*)));
    CHECK(cudaMalloc(&shapes, N*sizeof(Shape*)));
    initializeTargets<<<1,1>>>(targets, list, shapes, N);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Targets generated" << std::endl;


    std::cout << "Starting GPU rendering..." << std::endl;

    render<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d);
    CHECK(cudaDeviceSynchronize());
    
    std::cout << "\033[32;1mSuccessfully rendered & synchronized!\033[0m" << std::endl;

    //######################################
    // # GENERATE IMAGE, FREE MEMORY
    //######################################

    sf::Texture texture;
    texture.create(width, height);
    texture.update(pixels);

    sf::Image image = texture.copyToImage();
    image.saveToFile("testikuvaGPU.png");


    CHECK(cudaFree(cudaWindow));
    CHECK(cudaFree(pixels));
    CHECK(cudaFree(randState_d));

    releaseBG<<<1,1>>>(background_d);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(background_d));

    releaseTargets<<<1,1>>>(targets, list, shapes);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(targets));
    CHECK(cudaFree(list));
    CHECK(cudaFree(shapes));


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Total program runtime: " << duration << " seconds" << std::endl;

    cudaDeviceReset();

    return 0;
}