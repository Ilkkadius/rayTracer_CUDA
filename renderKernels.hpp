#ifndef CUDA_KERNELS_FOR_RENDERING_HPP
#define CUDA_KERNELS_FOR_RENDERING_HPP

#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

#include "targetf.hpp"
#include "backgroundsf.hpp"
#include "tracerf.hpp"

    __global__ void completeRender(sf::Uint8 *pixels, 
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

    __global__ void cumulativeRender(sf::Uint8 *pixels, 
        int width, int height, 
        int depth, int samples,
        targetList** list,
        BackgroundColor** background, 
        WindowVectors* window, 
        curandState* randState,
        int currentIteration) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if(i >= width || j >= height) return;

        int idx = width * j + i;
        int k = currentIteration;
        curandState rand = randState[idx];

        Vector3D color = 255 * TracePixelRnd(window, i, j, list, depth, samples, *background, rand);
        
        pixels[4*idx] = (color.x + k*pixels[4*idx])/(k + 1.0f);
        pixels[4*idx + 1] = (color.y + k*pixels[4*idx + 1])/(k + 1.0f);
        pixels[4*idx + 2] = (color.z + k*pixels[4*idx + 2])/(k + 1.0f);
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

    



#endif