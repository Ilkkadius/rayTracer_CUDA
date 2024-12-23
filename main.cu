#include <SFML/Graphics.hpp>
#include <chrono>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "initializers.hpp"
#include "tracerf.hpp"
#include "rayf.hpp"
#include "backgroundsf.hpp"
#include "auxiliaryf.hpp"
#include "targetList.hpp"
#include "geometria.hpp"
#include "cameraf.hpp"
#include "logMethods.hpp"
#include "imageBackupf.hpp"
#include "BVHf.hpp"
#include "fileOperations.hpp"

#include "kernelSet.hpp"

#include "realtimeRenderf.hpp"

// PPC
#ifndef CHECK_FUNC
static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)
#endif

#ifndef divup_FUNC
    static inline int divup(int a, int b) {
        return (a + b - 1)/b;
    }
#endif

//static inline int roundup(int a, int b) {
//    return divup(a, b) * b;
//}


// ################################################################

// nvcc main.cu -o main -lsfml-graphics -lsfml-window -lsfml-system

int main() {
    // #################################
    // # SET PROGRAM RUN PARAMETERS
    // #################################

    Camera cam;

    int width = 1920, height = 1080;
    int depth = 4, samples = 100;
    int tx = 8, ty = 8;
    bool backup = true;
    int partition = 0;
    bool realTime = false;
    bool fileRead = false;

    cam.setFOV(80.0f);

    Vector3D eye(0, 0, 0);
    Vector3D direction(1, 0, 0);
    Vector3D up = cam.direction + Vector3D(0, 0, 100);

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

    cam.eye = eye;
    cam.direction = direction;
    cam.up = up;

    cam.check();

    if(partition < 0 || partition > 3) {
        std::cout << "ERROR: Invalid partition value" << std::endl;
        return 1;
    }

    std::string backupBinPath(getRawDate() + "_" + getImageDimensions(width, height) 
                            + (samples > 0 ? "_N" + std::to_string(samples) : "") + "_GPU_backup.bin");
    std::string backupTextPath = "" + getRawDate() + "_" + std::to_string(width) + "x" + std::to_string(height) 
                            + (samples > 0 ? "_N" + std::to_string(samples) : "") + "_GPU_backup.txt";

    WindowVectors *cudaWindow = NULL;
    CHECK(cudaMalloc(&cudaWindow, sizeof(WindowVectors)));
    CHECK(cudaMemcpy(cudaWindow, &cam.window, sizeof(WindowVectors), cudaMemcpyHostToDevice));
    std::cout << "Window ready" << std::endl;


    dim3 blocks(divup(width, tx), divup(height, ty));
    dim3 threads(tx, ty);


    sf::Uint8 *pixels;
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


    targetList** list; Target** targets; Shape** shapes; int N = 2000;
    CHECK(cudaMalloc(&list, sizeof(targetList*)));
    CHECK(cudaMalloc(&targets, N*sizeof(Target*)));
    CHECK(cudaMalloc(&shapes, N*sizeof(Shape*)));
    initializeTargets<<<1,1>>>(targets, list, shapes, N);
    CHECK(cudaDeviceSynchronize());
    
    if(fileRead) {
        FileOperations::TargetsFromFile("teapot.obj", list, shapes);
        CHECK(cudaDeviceSynchronize());
    }

    Compound** compounds; size_t compoundCount;
    CHECK(cudaMalloc(&compounds, sizeof(Compound*)));
    CHECK(cudaDeviceSynchronize());
    //FileOperations::CompoundsFromFile("teapot.obj", compounds, compoundCount);
    //addCompoundsToTargetlist<<<1,1>>>(compounds, 1, list, shapes);
    
    BVHTree** tree;
    CHECK(cudaMalloc(&tree, sizeof(BVHTree*)));
    CHECK(cudaDeviceSynchronize());
    buildBVH<<<1,1>>>(list, tree);
    CHECK(cudaDeviceSynchronize());
    
    std::cout << "Targets generated" << std::endl;

    if(realTime) {
        realtimeRender::startCamera(cam, tree, background_d, randState_d, eye, direction, up); // TREE
        return 0;
    }

    // ####################################################################

    std::cout << "\033[0;32mGPU rendering started\033[0m" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = 0;

    if(partition == 0) {

        completeRender<<<blocks, threads>>>(pixels, width, height, depth, samples, // TREE
                                tree, background_d, cudaWindow, 
                                randState_d);
        CHECK(cudaDeviceSynchronize());

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

        //renderQuarter<<<blocks, threads>>>(pixels, width, height, depth, samples,
        //                        list, background_d, cudaWindow, 
        //                        randState_d, 0);
        //CHECK(cudaDeviceSynchronize());

        if(backup) {
            //Backup::quarterImageToBinary(backupBinPath, pixels, width, height, 0);
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "25% rendered, time so far: " << getDuration(duration) << std::endl;

        //renderQuarter<<<blocks, threads>>>(pixels, width, height, depth, samples,
        //                        list, background_d, cudaWindow, 
        //                        randState_d, 1);
        //CHECK(cudaDeviceSynchronize());

        if(backup) {
            //Backup::quarterImageToBinary(backupBinPath, pixels, width, height, 1);
        }

        auto prevDuration = duration;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "50% rendered, time so far: " << getDuration(duration) << " (\u0394t = " << getRawDuration(duration - prevDuration) << ")" << std::endl;

        renderQuarter<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d, 2);
        CHECK(cudaDeviceSynchronize());

        if(backup) {
            Backup::quarterImageToBinary(backupBinPath, pixels, width, height, 2);
        }

        prevDuration = duration;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "75% rendered, time so far: " << getDuration(duration) << " (\u0394t = " << getRawDuration(duration - prevDuration) << ")" << std::endl;

        renderQuarter<<<blocks, threads>>>(pixels, width, height, depth, samples,
                                list, background_d, cudaWindow, 
                                randState_d, 3);
        CHECK(cudaDeviceSynchronize());

        if(backup) {
            Backup::quarterImageToBinary(backupBinPath, pixels, width, height, 3);
        }

    }

    //Backup::fullImageToText(backupTextPath, pixels, width, height);
    if(backup)
        Backup::fullImageToBinary(pixels, width, height);
    
    
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