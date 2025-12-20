#include "renderMode.hpp"

void Mode::FullRender(int width, int height, int tx, int ty, timepoint& start, curandState* randState_d, Vector3D* results, 
                int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow) {
    dim3 blocks(divup(width, tx), divup(height, ty));
    dim3 threads(tx, ty);

    CHECK(cudaMalloc(&randState_d, width*height*sizeof(curandState)));
    initializeRand<<<blocks, threads>>>(randState_d, width, height);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Random states generated" << std::endl;

    std::cout << "GPU rendering started, single kernel" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    completeRender<<<blocks, threads>>>(results, width, height, depth, samples, // TREE
                            tree, background_d, cudaWindow, 
                            randState_d);
    CHECK(cudaDeviceSynchronize());
            
}


void Mode::partialFullRender(int width, int height, int tx, int ty, timepoint& start, curandState* randState_d, Vector3D* results, 
                int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow) {
    dim3 blocks(divup(width, tx), divup(height, ty));
    dim3 threads(tx, ty);

    CHECK(cudaMalloc(&randState_d, width*height*sizeof(curandState)));
    initializeRand<<<blocks, threads>>>(randState_d, width, height);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Random states generated" << std::endl;

    std::cout << "GPU rendering started, partial, full figure" << std::endl;
    std::cout << "Kernel limits: [" << float(KERNEL_RUNTIME_MIN_LIMIT) << ", " << float(KERNEL_RUNTIME_MAX_LIMIT) << "]" << std::endl;

    start = std::chrono::high_resolution_clock::now();

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
}


void Mode::partialPixelRender(int width, int height, int tx, int ty, timepoint& start, curandState* randState_d, Vector3D* results, 
                int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blockNum = divup(samples, threadsPerBlock);
    int maximum_offset_length = MAXIMUM_CURANDSTATE_MEMORY/(threadsPerBlock*blockNum*sizeof(curandState));
    int offsetLen = 1;
    dim3 blocks(blockNum, offsetLen);

    std::cout << "Allocated memory for random states: " << maximum_offset_length*threadsPerBlock*blockNum*sizeof(curandState)/1000000 << " MB" << std::endl;
    CHECK(cudaMalloc(&randState_d, maximum_offset_length*threadsPerBlock*blockNum*sizeof(curandState)));
    initializeRandSamples<<<dim3(blockNum, maximum_offset_length), threadsPerBlock>>>(randState_d);
    CHECK(cudaDeviceSynchronize());
    std::cout << "Random states generated" << std::endl;

    std::cout << "GPU rendering started, partial, pixels" << std::endl;
    std::cout << "Kernel limits: [" << float(KERNEL_RUNTIME_MIN_LIMIT) << ", " << float(KERNEL_RUNTIME_MAX_LIMIT) << "]" << std::endl;

    start = std::chrono::high_resolution_clock::now();

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
}
