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
#include "cameraf.hpp"
#include "image.hpp"
#include "BVHf.hpp"
#include "meshRead.hpp"
#include "kernelSet.hpp"
#include "realtimeRenderf.hpp"
#include "renderMode.hpp"
#include "csgRead.hpp"


// ################################################################

// Dynamic:
// nvcc main.cu -o main -lsfml-graphics -lsfml-window -lsfml-system

// nvcc main.cu src/*.cu src/*.cpp -o main -Isrc -lsfml-graphics -lsfml-window -lsfml-system -rdc=true -w


// Static
// nvcc main.cu -o main -I./dependencies/include -DSFML_STATIC -L./dependencies/lib -lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lopengl32 -lfreetype -lwinmm -lgdi32

// nvcc main.cu -o main -w -I./dependencies/include -L./dependencies/lib -lsfml-graphics -lsfml-window -lsfml-system -lopengl32 -lfreetype -lwinmm -lgdi32

int main(int argc, char *argv[]) {

    // #################################
    // # SET PROGRAM RUN PARAMETERS
    // #################################


    Camera cam;

    cam.width = 1920; cam.height = 1080;
    cam.depth = 4; cam.samples = 10;
    int tx = 8, ty = 8;
    bool backup = false;
    bool realTime = false;
    bool fileRead = false;
    bool readCSG = true;

    cam.setFOV(80.0f);

    Vector3D eye(0, 0, 0);
    Vector3D direction(1, 0, 0);
    Vector3D up = direction + Vector3D(0, 0, 100);

    RenderMode launchMode = RenderMode::Single_full;

    if(argc >= 2) {
        std::string mode = argv[1];
        aux::uppercase(mode);
        if(mode == "SINGLE" || mode == "FULL") {
            launchMode = RenderMode::Single_full;
        } else if(mode == "PARTFULL" || mode == "PARTIALFULL" || mode == "PARTIALLYFULL") {
            launchMode = RenderMode::Partial_full;
        } else if(mode == "PARTIALPIXEL" || mode == "PIXEL" || mode == "PIXELS") {
            launchMode = RenderMode::Partial_pixel;
        }
    }

    if(argc >= 3) {
        aux::stringToInt((std::string)argv[2], cam.samples);
        if(cam.samples < 1) {
            std::cout << termcolor::bold << termcolor::red << "Samplecount must be at least 1" << termcolor::reset << std::endl;
            exit(1);
        }

    }

    std::string parentDir = aux::parentDirectory(__FILE__);
    std::cout << parentDir << std::endl;

    // #################################
    // # LOAD DATA TO DEVICE
    // #################################

    std::cout << termcolor::yellow <<
    "#################################\n"
    "#        Ray tracer (GPU)       #\n"
    "# Date: " << aux::getDate() << " #\n"
    "#################################" 
    << termcolor::reset << std::endl;

    std::cout << "Resolution: " << cam.width << "x" << cam.height << ", N = " << cam.samples << ", recursion = " << cam.depth << std::endl;
    std::cout << "Backup to file: ";
    if(backup) {
        std::cout << termcolor::bright_green;
    } else {
        std::cout << termcolor::bright_red;
    }
    std::cout << std::boolalpha << backup << termcolor::reset << std::endl;

    cam.eye = eye;
    cam.direction = direction;
    cam.up = up;

    cam.check();

    int width = cam.width, height = cam.height;

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    std::string backupBinPath(aux::getRawDate() + "_" + Image::getImageDimensions(width, height) 
                            + (cam.samples > 0 ? "_N" + std::to_string(cam.samples) : "") + "_GPU_backup.bin");
    std::string backupTextPath = "" + aux::getRawDate() + "_" + std::to_string(width) + "x" + std::to_string(height) 
                            + (cam.samples > 0 ? "_N" + std::to_string(cam.samples) : "") + "_GPU_backup.txt";

    WindowVectors *cudaWindow = NULL;
    CHECK(cudaMalloc(&cudaWindow, sizeof(WindowVectors)));
    CHECK(cudaMemcpy(cudaWindow, &cam.window, sizeof(WindowVectors), cudaMemcpyHostToDevice));
    std::cout << "Window ready" << std::endl;

    Vector3D *results;
    CHECK(cudaMallocManaged(&results, width*height*sizeof(Vector3D)));

    backgroundType bgType = backgroundType::NIGHT;
    BackgroundColor** background_d;
    CHECK(cudaMalloc(&background_d, sizeof(BackgroundColor*)));
    initializeBG<<<1,1>>>(background_d, bgType);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Background ready" << std::endl;

    TargetList** list; Target** targets; int N = int(MAXIMUM_TARGET_COUNT);
    CHECK(cudaMalloc(&list, sizeof(TargetList*)));
    CHECK(cudaMalloc(&targets, N*sizeof(Target*)));
    initializeTargets<<<1,1>>>(targets, list, N);
    CHECK(cudaDeviceSynchronize());


    if(readCSG) csgRead::csgFromFile((parentDir + "csg.txt").c_str(), list);

    if(fileRead) {
        MeshRead::TargetsFromFile("teapot.obj", list);
        CHECK(cudaDeviceSynchronize());
    }

    Compound** compounds;
    CHECK(cudaMalloc(&compounds, sizeof(Compound*)));
    CHECK(cudaDeviceSynchronize());
    
    BVHTree** tree;
    CHECK(cudaMalloc(&tree, sizeof(BVHTree*)));
    CHECK(cudaDeviceSynchronize());
    buildBVH<<<1,1>>>(list, tree);
    CHECK(cudaDeviceSynchronize());
    
    std::cout << "Targets generated" << std::endl;

    if(realTime) {
        realtimeRender::startCamera(cam, tree, background_d);
        return 0;
    }

    timepoint start;
    curandState *randState_d;

    switch(launchMode) {
        case RenderMode::Single_full: // Full image rendered by one kernel
            Mode::FullRender(width, height, tx, ty, start, &randState_d, results, cam.depth, cam.samples, tree, background_d, cudaWindow);
            break;
        case RenderMode::Partial_full: // Set of kernels each rendering the full image, but number of samples divided evenly among the kernels
            Mode::partialFullRender(width, height, tx, ty, start, &randState_d, results, cam.depth, cam.samples, tree, background_d, cudaWindow);
            break;
        case RenderMode::Partial_pixel: // A large set of kernels each rendering one or many pixels of the image
            Mode::partialPixelRender(width, height, tx, ty, start, &randState_d, results, cam.depth, cam.samples, tree, background_d, cudaWindow);
            break;
    }

    std::cout << termcolor::bright_green << "Successfully rendered & synchronized!" << termcolor::reset << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Rendertime: " << aux::getDuration(duration) << std::endl;

    sf::Uint8* pixels = new sf::Uint8[width*height*4];

    for(int i = 0; i < width*height; i++) {
        Vector3D p = results[i]/float(cam.samples);
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
        Image::ToBinary(pixels, cam.width, cam.height);
    }

    //######################################
    // # GENERATE IMAGE, FREE MEMORY
    //######################################

    Image::writePNG(pixels, cam.width, cam.height, cam.samples, duration);
    delete[] pixels;

    CHECK(cudaFree(results));
    CHECK(cudaFree(cudaWindow));
    CHECK(cudaFree(randState_d));

    releaseBG<<<1,1>>>(background_d);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(background_d));

    releaseTargets<<<1,1>>>(targets, list);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(targets));
    CHECK(cudaFree(list));

    cudaDeviceReset();

    return 0;
}