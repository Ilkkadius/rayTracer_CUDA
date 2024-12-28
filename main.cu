#include <SFML/Graphics.hpp>
#include <chrono>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "termcolor.hpp"

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


// ################################################################

// Dynamic:
// nvcc main.cu -o main -lsfml-graphics -lsfml-window -lsfml-system

// Static
// nvcc main.cu -o main -I./dependencies/include -DSFML_STATIC -L./dependencies/lib -lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lopengl32 -lfreetype -lwinmm -lgdi32

// nvcc main.cu -o main -w -I./dependencies/include -L./dependencies/lib -lsfml-graphics -lsfml-window -lsfml-system -lopengl32 -lfreetype -lwinmm -lgdi32

#define SINGLE_KERNEL_RENDER FALSE
#define KERNEL_RUNTIME_MAX_LIMIT 1.5f
#define KERNEL_RUNTIME_MIN_LIMIT 0.5f
#define MAXIMUM_CURANDSTATE_MEMORY 1000000000 // In bytes

int main() {
    // #################################
    // # SET PROGRAM RUN PARAMETERS
    // #################################

    Camera cam;

    int width = 1920, height = 1080;
    int depth = 4, samples = 10000;
    int tx = 8, ty = 8;
    bool backup = true;
    bool realTime = false;
    bool fileRead = false;

    cam.setFOV(80.0f);

    Vector3D eye(0, 0, 0);
    Vector3D direction(1, 0, 0);
    Vector3D up = direction + Vector3D(0, 0, 100);

    // #################################
    // # LOAD DATA TO DEVICE
    // #################################

    std::cout << termcolor::yellow <<
    "#################################\n"
    "#        Ray tracer (GPU)       #\n"
    "# Date: " << getDate() << " #\n"
    "#################################" 
    << termcolor::reset << std::endl;

    std::cout << "Resolution: " << width << "x" << height << ", N = " << samples << ", recursion = " << depth << std::endl;
    std::cout << "Backup to file: ";
    if(backup) {
        std::cout << termcolor::bright_green;
    } else {
        std::cout << termcolor::bright_red;
    }
    std::cout << std::boolalpha << backup << termcolor::reset << std::endl;

    cam.width = width; cam.height = height;
    cam.depth = depth; cam.samples = samples;

    cam.eye = eye;
    cam.direction = direction;
    cam.up = up;

    cam.check();

    int pixelCount = width*height;

    std::string backupBinPath(getRawDate() + "_" + getImageDimensions(width, height) 
                            + (samples > 0 ? "_N" + std::to_string(samples) : "") + "_GPU_backup.bin");
    std::string backupTextPath = "" + getRawDate() + "_" + std::to_string(width) + "x" + std::to_string(height) 
                            + (samples > 0 ? "_N" + std::to_string(samples) : "") + "_GPU_backup.txt";

    WindowVectors *cudaWindow = NULL;
    CHECK(cudaMalloc(&cudaWindow, sizeof(WindowVectors)));
    CHECK(cudaMemcpy(cudaWindow, &cam.window, sizeof(WindowVectors), cudaMemcpyHostToDevice));
    std::cout << "Window ready" << std::endl;

    Vector3D *results;
    CHECK(cudaMallocManaged(&results, width*height*sizeof(Vector3D)));

    BackgroundColor** background_d;
    CHECK(cudaMalloc(&background_d, sizeof(BackgroundColor*)));
    initializeBG<<<1,1>>>(background_d);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Background ready" << std::endl;

    TargetList** list; Target** targets; Shape** shapes; int N = 2000;
    CHECK(cudaMalloc(&list, sizeof(TargetList*)));
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
        dim3 blocks(divup(width, tx), divup(height, ty));
        dim3 threads(tx, ty);

        curandState *randState_d;
        CHECK(cudaMalloc(&randState_d, width*height*sizeof(curandState)));
        initializeRand<<<blocks, threads>>>(randState_d, width, height);
        CHECK(cudaDeviceSynchronize());
        realtimeRender::startCamera(cam, tree, background_d, randState_d, eye, direction, up); // TREE
        return 0;
    }

    #if SINGLE_KERNEL_RENDER == TRUE

    dim3 blocks(divup(width, tx), divup(height, ty));
    dim3 threads(tx, ty);

    curandState *randState_d;
    CHECK(cudaMalloc(&randState_d, width*height*sizeof(curandState)));
    initializeRand<<<blocks, threads>>>(randState_d, width, height);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Random states generated" << std::endl;

    // ####################################################################

    std::cout << "GPU rendering started" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    double duration = 0; 

    completeRender<<<blocks, threads>>>(results, width, height, depth, samples, // TREE
                            tree, background_d, cudaWindow, 
                            randState_d);
    CHECK(cudaDeviceSynchronize());

    #else

    #define FULL_IMAGE FALSE

    #if FULL_IMAGE == TRUE

    dim3 blocks(divup(width, tx), divup(height, ty));
    dim3 threads(tx, ty);

    curandState *randState_d;
    CHECK(cudaMalloc(&randState_d, width*height*sizeof(curandState)));
    initializeRand<<<blocks, threads>>>(randState_d, width, height);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Random states generated" << std::endl;

    // ####################################################################

    std::cout << "GPU rendering started" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    double duration = 0; 

    int division = 10;
    float part = 1.0f/division;

    int i = 0, batchSize = 1;
    while(i < samples) {
        batchSize = std::min(samples - i, batchSize);
        auto t0 = std::chrono::high_resolution_clock::now();
        completeRender<<<blocks, threads>>>(results, width, height, depth, batchSize, // TREE
                            tree, background_d, cudaWindow, 
                            randState_d);
        CHECK(cudaDeviceSynchronize());
        i += batchSize;

        auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t0).count();
        if(tdiff > KERNEL_RUNTIME_MAX_LIMIT) {
            batchSize = std::max(1, int(std::floor(0.99*batchSize)));
        } else if(tdiff < KERNEL_RUNTIME_MIN_LIMIT) {
            batchSize = std::max(int(1.01*batchSize), batchSize+1);
        }

        while(i > part*samples) {
            std::cout << std::setprecision(3) << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count() << " s: "
            << part*100.0f << " % done (batch size: " << batchSize << ")" << std::endl;
            part += 1.0f/division;
        }
    }


    #else
    

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blockNum = divup(samples, threadsPerBlock);
    int maximum_offset_length = MAXIMUM_CURANDSTATE_MEMORY/(threadsPerBlock*blockNum*sizeof(curandState));
    int offsetLen = 1;
    dim3 blocks(blockNum, offsetLen);

    curandState *randState_d;
    //CHECK(cudaMalloc(&randState_d, threadsPerBlock*blockNum*sizeof(curandState)));
    std::cout << "Allocated memory for random states: " << maximum_offset_length*threadsPerBlock*blockNum*sizeof(curandState)/1000000 << " MB" << std::endl;
    CHECK(cudaMalloc(&randState_d, maximum_offset_length*threadsPerBlock*blockNum*sizeof(curandState)));
    initializeRandSamples<<<dim3(blockNum, maximum_offset_length), threadsPerBlock>>>(randState_d);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Random states generated" << std::endl;

    std::cout << "GPU rendering started" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    double duration = 0;    

    int division = 20;
    float part = 1.0f/division;
    int i = 0;
    while(i < width*height) {
        auto t0 = std::chrono::high_resolution_clock::now();
        renderPixels<<<blocks, threadsPerBlock>>>(results, i, width, height, depth, samples, tree, background_d, cudaWindow, randState_d);
        CHECK(cudaDeviceSynchronize());
        i += offsetLen;

        while(i > part*width*height) {
            std::cout << std::setprecision(3) << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count() 
            << " s: " << part*100.0f << " % done (offset: " << offsetLen << ")" << std::endl;
            part += 1.0f/division;
        }

        auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t0).count();
        if(tdiff > KERNEL_RUNTIME_MAX_LIMIT) {
            offsetLen = std::max(1, int(0.99*offsetLen));
            blocks = dim3(blockNum, offsetLen);
        } else if(tdiff < KERNEL_RUNTIME_MIN_LIMIT) {
            offsetLen = std::min(int(maximum_offset_length), int(std::ceil(1.01f*offsetLen)));
            blocks = dim3(blockNum, offsetLen);
        }
    }

    #endif

    #endif

    std::cout << termcolor::bright_green << "Successfully rendered & synchronized!" << termcolor::reset << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Rendertime: " << getDuration(duration) << std::endl;

    sf::Uint8* pixels = new sf::Uint8[width*height*4];

    for(int i = 0; i < width*height; i++) {
        Vector3D p = results[i]/float(samples);
        if(p.max() > 1.0f) {
            p =  p/p.max();
        }
        p = 255.0f*p;
        int idx = i << 2;
        pixels[idx] = std::min(255, (int)std::round(p.x));
        pixels[idx+1] = std::min(255, (int)std::round(p.y));
        pixels[idx+2] = std::min(255, (int)std::round(p.z));
        pixels[idx+3] = 255;
    }


    if(backup) {
        Backup::fullImageToBinary(pixels, width, height);
    }

    //######################################
    // # GENERATE IMAGE, FREE MEMORY
    //######################################

    sf::Texture texture;
    texture.create(width, height);
    texture.update(pixels);

    delete[] pixels;

    std::string filename = getImageFilename(width, height, samples, duration);

    sf::Image image = texture.copyToImage();

    std::string prefix = "figures/";
    if(!image.saveToFile(prefix + filename)) {
        std::cout << "Trying again..." << std::endl;
        prefix = "../" + prefix;
        if(!image.saveToFile(prefix + filename)) {
            std::cout << "Trying again..." << std::endl;
            if(!image.saveToFile(filename)) {
                std::cout << termcolor::red << "Image was not saved..." << termcolor::reset << std::endl;
            } else {
                std::cout << "Successfully saved the image to the current directory" << std::endl;
            }

        } else {
            std::cout << "Successfully saved the image \"" << prefix + filename << "\"" << std::endl;
        }    
    } else {
        std::cout << "Successfully saved the image \"" << prefix + filename << "\"" << std::endl;
    }


    CHECK(cudaFree(results));
    CHECK(cudaFree(cudaWindow));
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