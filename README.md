# C++ ray tracer utilizing CUDA

Ray tracing is a method for rendering images from 3D data. Each pixel of the image is sampled with rays, which in turn acquire the color data by scattering in the scene from different objects. The large number of rays to be traced makes this method computationally heavy, but as each pixel is essentially independent of each other, the calculations can be executed in parallel.

This ray tracer uses *CUDA* in order to take advantage of GPU in parallel calculations. Therefore, only **NVIDIA**'s GPUs are supported and the program must be compiled with *nvcc*. The creation of image from calculated pixel colors is done using [SFML library](https://www.sfml-dev.org/).

---

## Rendering and saving the image

At the moment changing the settings takes place manually in the **main.cu** file. Adjustable parameters include, e.g. resolution, samples per pixel and number of scatters per ray. Rendered images can be built immediately after the rendering to a *.png*-format, or they may be saved into a binary file. The **buildImage.cu** is then able to create the image from these binary files.


## Camera mode

In addition to "static rendering", also interactive camera mode can be used. Controls are **WASD** and arrow keys for the horizontal movement, space and left shift for vertical movement. **Q** and **E** can be used to adjust the view angle (roll). Mouse cursor allows more free movement of view. Pressing **ESC** releases the cursor from the render window and **ENTER** ends the rendering completely. Pressing **V** outputs the current camera parameters to console and **F** toggles the fast movement mode.


## Other features

Currently only spheres and triangles are supported. Scattering process is Lambertian.

![Platonic solids](/figures/25000_sample_Platon.png)

3D models can be imported to the program - see an example below. Supported file type is **.obj**

![Teacup](/figures/teacup_N10000.png)

To further increase the efficiency, a bounding volume hierarchy (BVH) structure is used as an acceleration structure.

---

## Useful resources

- Aalto university: [Programming Parallel Computers](https://ppc.cs.aalto.fi/)
- R. Allen: [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
- J. Bikker: [How to build a BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)