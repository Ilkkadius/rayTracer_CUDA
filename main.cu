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

    int width = 1920, height = 1080;
    int depth = 4, samples = 10;
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

    if(argc == 3) {
        aux::stringToInt((std::string)argv[2], samples);
        if(samples < 1) {
            std::cout << termcolor::bold << termcolor::red << "Samplecount must be at least 1" << termcolor::reset << std::endl;
            exit(1);
        }

    }

    std::string parentDir = "";
	{
		std::string maincpp = std::string(__FILE__); int spot = 0;
		for(int i = 0; i < maincpp.size(); i++)
			if(maincpp[i] == '/' || maincpp[i] == '\\')
				spot = i;
		if(spot > 0) parentDir = maincpp.substr(0, spot) + "/";
	}


    // #################################
    // # LOAD DATA TO DEVICE
    // #################################

    std::cout << termcolor::yellow <<
    "#################################\n"
    "#        Ray tracer (GPU)       #\n"
    "# Date: " << aux::getDate() << " #\n"
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

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    std::string backupBinPath(aux::getRawDate() + "_" + Image::getImageDimensions(width, height) 
                            + (samples > 0 ? "_N" + std::to_string(samples) : "") + "_GPU_backup.bin");
    std::string backupTextPath = "" + aux::getRawDate() + "_" + std::to_string(width) + "x" + std::to_string(height) 
                            + (samples > 0 ? "_N" + std::to_string(samples) : "") + "_GPU_backup.txt";

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
        dim3 blocks(divup(width, tx), divup(height, ty));
        dim3 threads(tx, ty);

        curandState *randState_d;
        CHECK(cudaMalloc(&randState_d, width*height*sizeof(curandState)));
        initializeRand<<<blocks, threads>>>(randState_d, width, height);
        CHECK(cudaDeviceSynchronize());
        realtimeRender::startCamera(cam, tree, background_d, randState_d, eye, direction, up); // TREE
        return 0;
    }

    timepoint start;
    curandState *randState_d;

    switch(launchMode) {
        case RenderMode::Single_full: // Full image rendered by one kernel
            Mode::FullRender(width, height, tx, ty, start, &randState_d, results, depth, samples, tree, background_d, cudaWindow);
            break;
        case RenderMode::Partial_full: // Set of kernels each rendering the full image, but number of samples divided evenly among the kernels
            Mode::partialFullRender(width, height, tx, ty, start, &randState_d, results, depth, samples, tree, background_d, cudaWindow);
            break;
        case RenderMode::Partial_pixel: // A large set of kernels each rendering one or many pixels of the image
            Mode::partialPixelRender(width, height, tx, ty, start, &randState_d, results, depth, samples, tree, background_d, cudaWindow);
            break;
    }

    std::cout << termcolor::bright_green << "Successfully rendered & synchronized!" << termcolor::reset << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Rendertime: " << aux::getDuration(duration) << std::endl;

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
        Image::ToBinary(pixels, width, height);
    }

    //######################################
    // # GENERATE IMAGE, FREE MEMORY
    //######################################

    Image::writePNG(pixels, width, height, samples, duration);
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