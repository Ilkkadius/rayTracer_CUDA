#include "kernelSet.hpp"


__global__ void completeRender(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    TargetList** list,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;

    int idx = width * j + i;
    curandState rand = randState[idx];

    Vector3D color = 255.0f * TracePixelRnd(window, i, j, list, depth, samples, *background, rand);
    
    idx = idx << 2;
    
    pixels[idx] = color.x;
    pixels[idx + 1] = color.y;
    pixels[idx + 2] = color.z;
    pixels[idx + 3] = 255;
}

__global__ void completeRender(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;

    int idx = width * j + i;
    curandState rand = randState[idx];

    Vector3D color = 255.0f * TracePixelRnd(window, i, j, *tree, depth, samples, *background, rand);

    idx = idx << 2;
    
    pixels[idx] = color.x;
    pixels[idx + 1] = color.y;
    pixels[idx + 2] = color.z;
    pixels[idx + 3] = 255;
}

__global__ void completeRender(Vector3D* pixels, 
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;

    int idx = width * j + i;
    curandState rand = randState[idx];

    pixels[idx] += TracePixelRnd(window, i, j, *tree, depth, samples, *background, rand);
    randState[idx] = rand;
}

__global__ void renderPixel(Vector3D* color, 
    int x, int y,
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState) {
    
    __shared__ Vector3D sharedData[THREADS_PER_BLOCK];

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int threadCount = gridDim.x * blockDim.x;

    curandState rand = randState[tidx];

    sharedData[threadIdx.x] = {0.0f, 0.0f, 0.0f};

    Vector3D threadSum = {0.0f, 0.0f, 0.0f};
    for (int i = tidx; i < samples; i += threadCount) {
        threadSum = threadSum + TracePixelRnd(window, x, y, *tree, depth, *background, rand);
    }
    sharedData[threadIdx.x] = threadSum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] = sharedData[threadIdx.x] + sharedData[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write block-level result to global memory
    if (threadIdx.x == 0) {
        atomicAdd(&color->x, sharedData[0].x);
        atomicAdd(&color->y, sharedData[0].y);
        atomicAdd(&color->z, sharedData[0].z);
    }
}

__global__ void renderPixels(Vector3D* color, 
    int pixelIdx,
    int width, int height,
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState) {
    
    __shared__ Vector3D sharedData[THREADS_PER_BLOCK];

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    pixelIdx += blockIdx.y; // blocks in y-direction define various pixel offsets
    int threadCount = gridDim.x * blockDim.x;

    if(pixelIdx >= width*height) return;

    int randIdx = tidx + blockIdx.y * gridDim.x * blockDim.x; // tidx
    //curandState rand = randState[tidx];
    curandState rand = randState[randIdx];

    sharedData[threadIdx.x] = {0.0f, 0.0f, 0.0f};

    int x = pixelIdx % width, y = pixelIdx / width;

    Vector3D threadSum = {0.0f, 0.0f, 0.0f};
    for (int i = tidx; i < samples; i += threadCount) {
        threadSum = threadSum + TracePixelRnd(window, x, y, *tree, depth, *background, rand);
    }
    sharedData[threadIdx.x] = threadSum;
    randState[randIdx] = rand;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] = sharedData[threadIdx.x] + sharedData[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write block-level result to global memory
    if (threadIdx.x == 0) {
        atomicAdd(&(color[pixelIdx].x), sharedData[0].x);
        atomicAdd(&(color[pixelIdx].y), sharedData[0].y);
        atomicAdd(&(color[pixelIdx].z), sharedData[0].z);
    }
}

__global__ void RealTimeRender(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState, double* darray) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;

    int idx = width * j + i;
    curandState rand = randState[idx];

    Vector3D color = 255 * TracePixelRnd(window, i, j, *tree, depth, samples, *background, rand);

    int pidx = idx << 2;
    
    pixels[pidx] = color.x;
    pixels[pidx + 1] = color.y;
    pixels[pidx + 2] = color.z;
    pixels[pidx + 3] = 255;

    idx += idx << 1;

    darray[idx] = color.x;
    darray[idx + 1] = color.y;
    darray[idx + 2] = color.z;
}

__global__ void RealTimeUpdateRender(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState, double* darray, float frameIdx) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;

    int idx = width * j + i;
    curandState rand = randState[idx];

    Vector3D color = 255 * TracePixelRnd(window, i, j, *tree, depth, samples, *background, rand);

    int didx = 3*idx;

    darray[didx] += color.x;
    darray[didx + 1] += color.y;
    darray[didx + 2] += color.z;

    int pidx = didx + idx;
    float rd = 1/(1.0f + frameIdx);
    
    pixels[pidx] = darray[didx] * rd;
    pixels[pidx + 1] = darray[didx + 1] * rd;
    pixels[pidx + 2] = darray[didx + 1] * rd;
    pixels[pidx + 3] = 255;

    
}


__global__ void renderHalf(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    TargetList** list,
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
    
    idx = idx << 2;
    
    pixels[idx] = color.x;
    pixels[idx + 1] = color.y;
    pixels[idx + 2] = color.z;
    pixels[idx + 3] = 255;
}


__global__ void renderQuarter(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    TargetList** list,
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
    
    idx = idx << 2;
    
    pixels[idx] = color.x;
    pixels[idx + 1] = color.y;
    pixels[idx + 2] = color.z;
    pixels[idx + 3] = 255;
}

__global__ void generateTargets(TargetList** list, Shape** shapes, 
                                Vector3D* vertices, int* fVertices, 
                                Vector3D* fColors, size_t fCount, 
                                Vector3D* defaultColor) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        Vector3D color, v1, v2, v3;
        Compound fileCompound(fCount);
        for(size_t i = 0; i < fCount; i++) {
            int idx = 3*i;
            int v[3] = {fVertices[idx], fVertices[idx + 1], fVertices[idx + 2]};
            color = (fColors[i].x >= 0 && fColors[i].y >= 0 && fColors[i].z >= 0) ? fColors[i] : *defaultColor; // Fix this!

            v1 = vertices[v[0]], v2 = vertices[v[1]], v3 = vertices[v[2]];
            Triangle* T = new Triangle(v1, v2, v3);
            fileCompound.add(new Target(T, color));
        }
        fileCompound.copyToList(*list, shapes);
    }
}

__global__ void generateCompounds(Compound** list, Vector3D* vertices, 
                                    int* fVertices, Vector3D* fColors, 
                                    size_t fCount, Vector3D* defaultColor) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        Vector3D color, v1, v2, v3;
        Compound* fileCompound = new Compound(fCount);
        for(size_t i = 0; i < fCount; i++) {
            int idx = 3*i;
            int v[3] = {fVertices[idx], fVertices[idx + 1], fVertices[idx + 2]};
            color = (fColors[i].x >= 0 && fColors[i].y >= 0 && fColors[i].z >= 0) ? fColors[i] : *defaultColor; // Fix this!

            v1 = vertices[v[0]], v2 = vertices[v[1]], v3 = vertices[v[2]];
            Triangle* T = new Triangle(v1, v2, v3);
            fileCompound->add(new Target(T, color));
        }
        *list = fileCompound;
    }
}

__global__ void addCompoundsToTargetlist(Compound** compounds, size_t compoundCount, TargetList** list, Shape** shapes) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        for(int i = 0; i < compoundCount; i++) {
            compounds[i]->copyToList(*list, shapes);
        }
    }
}

__global__ void initializeBG(BackgroundColor** background) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *background = init::createBackground();
    }
}

__global__ void initializeTargets(Target** targets, TargetList** list, Shape** shapes, int capacity) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        init::createTargets(targets, list, shapes, capacity);
    }
}

__global__ void buildBVH(TargetList** listptr, BVHTree** tree) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *tree = new BVHTree(listptr);
    }
}

__global__ void initializeRand(curandState* randState, int width, int height, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;
    int idx = i + width * j;
    curand_init(seed, idx, 0, &randState[idx]);
}

__global__ void initializeRandSamples(curandState* randState, int seed) {
    //if(blockIdx.y > 0) return;
    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    curand_init(seed, i, 0, &randState[i]);
}


__global__ void releaseBG(BackgroundColor** background) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        delete *background;
    }
}

__global__ void releaseTargets(Target** targets, TargetList** list, Shape** shapes) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        TargetList l = **list;
        for(int i = 0; i < l.size; i++) {
            delete *(targets + i);
            delete *(shapes + i);
        }
        delete *list;
    }
}

__global__ void releaseBVH(Target** targets, TargetList** list, Shape** shapes, BVHTree* tree) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        TargetList l = **list;
        for(int i = 0; i < l.size; i++) {
            delete *(targets + i);
            delete *(shapes + i);
        }
        delete *list;
        tree->clear();
    }
}

    


