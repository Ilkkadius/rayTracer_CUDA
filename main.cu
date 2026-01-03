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
#include "cameraf.hpp"
#include "image.hpp"
#include "BVHf.hpp"
#include "meshRead.hpp"
#include "realtimeRenderf.hpp"
#include "renderMode.hpp"
#include "fileRead.hpp"
#include "cui.hpp"



// ################################################################

// Dynamic:
// nvcc main.cu -o main -lsfml-graphics -lsfml-window -lsfml-system

// nvcc main.cu src/*.cu src/*.cpp -o main -Isrc -lsfml-graphics -lsfml-window -lsfml-system -rdc=true -w


// Static
// nvcc main.cu -o main -I./dependencies/include -DSFML_STATIC -L./dependencies/lib -lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lopengl32 -lfreetype -lwinmm -lgdi32

// nvcc main.cu -o main -w -I./dependencies/include -L./dependencies/lib -lsfml-graphics -lsfml-window -lsfml-system -lopengl32 -lfreetype -lwinmm -lgdi32

int main(int argc, char *argv[]) {

    std::string parentDir = aux::parentDirectory(__FILE__);

    cui::inputs cmd;

    if(!cui::parseCommandLineInput(argc, argv, cmd)) return 0;
    
    cui::checkScene(cmd);

    // #################################
    // # SET PROGRAM RUN PARAMETERS
    // #################################

    Config conf;
    Camera& cam = conf.cam;

    cam.width = 1920; cam.height = 1080;
    cam.depth = 4; cam.samples = 10;
    int tx = 8, ty = 8;
    
    cam.setFOV(80.0f);

    cudaDeviceSetLimit(cudaLimitStackSize, MAXIMUM_TOTAL_STACK_SIZE);
   
    // #################################
    // # LOAD DATA TO DEVICE
    // #################################

    cui::overrideConfig(cmd, conf);

    TargetList** list; Target** targets; int N = int(MAXIMUM_TARGET_COUNT);
    CHECK(cudaMalloc(&list, sizeof(TargetList*)));
    CHECK(cudaMalloc(&targets, N*sizeof(Target*)));
    initializeTargets<<<1,1>>>(targets, list, N, conf.scene);
    CHECK(cudaDeviceSynchronize());
    
    if(!cmd.file.empty())
        fileRead::parseFile(cmd.file.c_str(), list, conf);
    conf.cam.check();

    if(cmd.realtime) conf.realtime = true;
    if(cmd.samples > 0) conf.cam.samples = cmd.samples;

    int width = cam.width, height = cam.height;

    cui::printStart(conf);

    WindowVectors *cudaWindow = NULL;
    CHECK(cudaMalloc(&cudaWindow, sizeof(WindowVectors)));
    CHECK(cudaMemcpy(cudaWindow, &conf.cam.window, sizeof(WindowVectors), cudaMemcpyHostToDevice));
    std::cout << "Window ready" << std::endl;

    BackgroundColor** background_d;
    CHECK(cudaMalloc(&background_d, sizeof(BackgroundColor*)));
    initializeBG<<<1,1>>>(background_d, conf.background);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Background ready" << std::endl;

    std::string backupBinPath(aux::getRawDate() + "_" + Image::getImageDimensions(width, height) 
                            + (cam.samples > 0 ? "_N" + std::to_string(cam.samples) : "") + "_GPU_backup.bin");
    std::string backupTextPath = "" + aux::getRawDate() + "_" + std::to_string(width) + "x" + std::to_string(height) 
                            + (cam.samples > 0 ? "_N" + std::to_string(cam.samples) : "") + "_GPU_backup.txt";


    Vector3D *results;
    CHECK(cudaMallocManaged(&results, width*height*sizeof(Vector3D)));

    /*
    bool meshRead = false;
    if(meshRead) {
        MeshRead::TargetsFromFile("teapot.obj", list);
        CHECK(cudaDeviceSynchronize());
    }
    */

    Compound** compounds;
    CHECK(cudaMalloc(&compounds, sizeof(Compound*)));
    CHECK(cudaDeviceSynchronize());
    
    BVHTree** tree;
    CHECK(cudaMalloc(&tree, sizeof(BVHTree*)));
    CHECK(cudaDeviceSynchronize());
    buildBVH<<<1,1>>>(list, tree); 
    CHECK(cudaDeviceSynchronize());

    if(conf.realtime) {
        realtimeRender::startCamera(cam, tree, background_d);
        return 0;
    }

    timepoint start;
    curandState *randState_d;

    switch(cmd.launchMode) {
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


    if(conf.backup) {
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