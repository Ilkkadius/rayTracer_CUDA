#include <SFML/Graphics.hpp>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>

#include "initializers.hpp"
#include "vector3D.hpp"
#include "tracerf.hpp"
#include "rayf.hpp"
#include "backgroundsf.hpp"
#include "auxiliaryf.hpp"
#include "targetf.hpp"
#include "geometria.hpp"
#include "cameraf.hpp"
#include "logMethods.hpp"
#include "imageBackupf.hpp"
#include "BVHf.hpp"

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

//static inline int roundup(int a, int b) {
//    return divup(a, b) * b;
//}

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

    Vector3D color = 255 * TracePixelRnd(window, i, j, list, depth, samples, *background, rand);
    
    pixels[4*idx] = color.x;
    pixels[4*idx + 1] = color.y;
    pixels[4*idx + 2] = color.z;
    pixels[4*idx + 3] = 255;
}

__global__ void renderHalf(sf::Uint8 *pixels, 
        int width, int height, 
        int depth, int samples,
        targetList** list,
        BackgroundColor** background, 
        WindowVectors* window, 
        curandState* randState, bool left) {
            
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;
    if(left) {
        if(i >= width/2) return;
    } else {
        if(i < width/2) return;
    }

    int idx = width * j + i;
    curandState rand = randState[idx];

    Vector3D color = 255 * TracePixelRnd(window, i, j, list, depth, samples, *background, rand);
    
    pixels[4*idx] = color.x;
    pixels[4*idx + 1] = color.y;
    pixels[4*idx + 2] = color.z;
    pixels[4*idx + 3] = 255;
}

__global__ void renderQuarter(sf::Uint8 *pixels, 
        int width, int height, 
        int depth, int samples,
        targetList** list,
        BackgroundColor** background, 
        WindowVectors* window, 
        curandState* randState, int quarter) {
            
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;
    quarter = quarter % 4;
    switch(quarter) {
        case 0:
            if(i >= width/4) return;
            break;
        case 1:
            if(i < width/4 || i >= width/2) return;
            break;
        case 2:
            if(i < width/2 || i >= 3*width/4) return;
            break;
        case 3:
            if(i < 3*width/4) return;
            break;
    }

    int idx = width * j + i;
    curandState rand = randState[idx];

    Vector3D color = 255 * TracePixelRnd(window, i, j, list, depth, samples, *background, rand);
    
    pixels[4*idx] = color.x;
    pixels[4*idx + 1] = color.y;
    pixels[4*idx + 2] = color.z;
    pixels[4*idx + 3] = 255;
}



__global__ void initializeBG(BackgroundColor** background) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *background = createNightTime();
    }
}

__global__ void initializeTargets(Target** targets, targetList** list, Shape** shapes, int capacity) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        createTargets(targets, list, shapes, capacity);
    }
}

__global__ void initializeBVH(targetList** listptr, Target** targets, Shape** shapes, BVHTree* tree, int capacity) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        createTargets(targets, listptr, shapes, capacity);
        *tree = BVHTree(listptr);
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

__global__ void releaseBVH(Target** targets, targetList** list, Shape** shapes, BVHTree* tree) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        targetList l = **list;
        for(int i = 0; i < l.size; i++) {
            delete *(targets + i);
            delete *(shapes + i);
        }
        delete *list;
        tree->clear();
    }
}

// ################################################################

// nvcc main.cu -o main -lsfml-graphics -lsfml-window -lsfml-system

int main() {
    // #################################
    // # SET PROGRAM RUN PARAMETERS
    // #################################

    Camera cam;

    int width = 1920, height = 1080;
    int depth = 3, samples = 500;
    int tx = 8, ty = 8;
    bool backup = true;
    int partition = 2;

    cam.setFOV(80.0f);

    cam.eye = Vector3D(0, 0, 0);
    cam.direction = Vector3D(1, 0, 0);
    cam.up = cam.direction + Vector3D(0, 0, 100);

    // #################################
    // # LOAD DATA TO DEVICE
    // #################################

    std::cout << "\033[0;93m#################################\033[0m" << std::endl;
    std::cout << "\033[0;93m#        Ray tracer (GPU)       #\033[0m" << std::endl;
    std::cout << "\033[0;93m# Date: " << getDate() << " #\033[0m" << std::endl;
    std::cout << "\033[0;93m#################################\033[0m" << std::endl;

    std::cout << "Resolution: " << width << "x" << height << ", N = " << samples << ", recursion = " << depth << std::endl;
    std::cout << "Backup to file: " << (backup ? "\033[1;32m" : "\033[1;31m") << std::boolalpha << backup << "\033[0m" << std::endl;

    cam.width = width; cam.height = height;
    cam.depth = depth; cam.samples = samples;

    cam.check();
    cam.initializeWindow();

    std::string backupBinPath(getRawDate() + "_" + getImageDimensions(width, height) + "_GPU_backup.bin");
    std::string backupTextPath = "" + getRawDate() + "_" + std::to_string(width) + "x" + std::to_string(height) 
                            + (samples > 0 ? "_" + std::to_string(samples) + "samples" : "") + "_GPU_backup.txt";

    WindowVectors *cudaWindow = NULL;
    CHECK(cudaMalloc(&cudaWindow, sizeof(WindowVectors)));
    CHECK(cudaMemcpy(cudaWindow, &cam.window, sizeof(WindowVectors), cudaMemcpyHostToDevice));
    std::cout << "Window ready" << std::endl;

    dim3 blocks(divup(width, tx), divup(height, ty));
    dim3 threads(tx, ty);

    sf::Uint8 *pixels;// = new sf::Uint8[width*height*4];
    CHECK(cudaMallocManaged(&pixels, width*height*4));



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


    targetList** list; Target** targets; Shape** shapes; int N = 95;
    CHECK(cudaMalloc(&list, sizeof(targetList*)));
    CHECK(cudaMalloc(&targets, N*sizeof(Target*)));
    CHECK(cudaMalloc(&shapes, N*sizeof(Shape*)));
    initializeTargets<<<1,1>>>(targets, list, shapes, N);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Targets generated" << std::endl;



    std::cout << "\033[0;32mGPU rendering started\033[0m" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = 0;

    if(partition == 0) {

        render<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d);

        if(backup) {
            Backup::fullImageToBinary(pixels, width, height);
        }

    } else if(partition == 1) {
        renderHalf<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d, true);
        CHECK(cudaDeviceSynchronize());

        if(backup) {
            Backup::halfImageToBinary(backupBinPath, pixels, width, height, true);
            //Backup::halfImageToText(backupTextPath, pixels, width, height, true);
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        std::cout << "Half rendered, time so far: " << getDuration(duration) << std::endl;

        renderHalf<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                    list, background_d, cudaWindow, 
                                    randState_d, false);
        CHECK(cudaDeviceSynchronize());

        if(backup) {
            Backup::halfImageToBinary(backupBinPath, pixels, width, height, false);
            //Backup::halfImageToText(backupTextPath, pixels, width, height, false);
        }

    } else if(partition == 2) {

        renderQuarter<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d, 0);
        CHECK(cudaDeviceSynchronize());

        if(backup) {
            Backup::quarterImageToBinary(backupBinPath, pixels, width, height, 0);
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "25% rendered, time so far: " << getDuration(duration) << std::endl;

        renderQuarter<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d, 1);
        CHECK(cudaDeviceSynchronize());

        if(backup) {
            Backup::quarterImageToBinary(backupBinPath, pixels, width, height, 1);
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "50% rendered, time so far: " << getDuration(duration) << std::endl;

        renderQuarter<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d, 2);
        CHECK(cudaDeviceSynchronize());

        if(backup) {
            Backup::quarterImageToBinary(backupBinPath, pixels, width, height, 2);
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "75% rendered, time so far: " << getDuration(duration) << std::endl;

        renderQuarter<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d, 3);
        CHECK(cudaDeviceSynchronize());

        if(backup) {
            Backup::quarterImageToBinary(backupBinPath, pixels, width, height, 3);
        }

    }

    //Backup::fullImageToText(backupTextPath, pixels, width, height);
    //Backup::fullImageToBinary(pixels, width, height);
    
    std::cout << "\033[32;1mSuccessfully rendered & synchronized!\033[0m" << std::endl;

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Rendertime: " << getDuration(duration) << std::endl;

    //######################################
    // # GENERATE IMAGE, FREE MEMORY
    //######################################

    sf::Texture texture;
    texture.create(width, height);
    texture.update(pixels);

    sf::Image image = texture.copyToImage();

    std::string filename = getImageFilename(width, height, samples, duration);
    if(!image.saveToFile("figures/" + filename)) {
        if(!image.saveToFile(filename)) {
            std::cout << "\033[31mImage was not saved...\033[0m" << std::endl;
        } else {
            std::cout << "\033[32;1mImage saved!\033[0m" << std::endl;
        }
    } else {
        std::cout << "\033[32;1mImage saved!\033[0m" << std::endl;
    }

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


    

    cudaDeviceReset();

    return 0;
}