# CUDA/C++ ray tracer

Ray tracing is a method for rendering images from 3D data. Each pixel of the image is sampled with rays, which in turn acquire the color data by scattering in the scene from different objects. The large number of rays to be traced makes this method computationally heavy, but as each pixel is independent of each other, the calculations can be executed in parallel.

This ray tracer uses *CUDA* in order to take advantage of GPU in parallel calculations. Therefore, only **NVIDIA**'s GPUs are supported and the program must be compiled with *nvcc*. The creation of image from calculated pixel colors is done using [SFML library](https://www.sfml-dev.org/).

---

## Rendering and saving the image

Settings for rendering and the scene can be configured in a text file and passed to the program after *-f* command line argument. Otherwise there are some hard-coded scene templates available. Pass *?* as a CL argument for help. Adjustable parameters include, e.g. resolution, field of view, samples per pixel and number of scatters per ray. An example of such a configuration file is provided, c.f. *example.txt*. Rendered images can be built immediately after the rendering to a *.png*-format and saved under *figures*, or they may be saved into a binary file. The **buildImage.cu** is then able to create the image from these binary files.


## Camera mode

In addition to "static rendering", an interactive camera mode can be used by passing *-rt* commmand line argument. Controls are **WASD** and arrow keys for the horizontal movement, space and left shift for vertical movement. **Q** and **E** can be used to adjust the view angle (roll). Mouse cursor allows more free movement of view. Pressing **ESC** releases the cursor from the render window and **ENTER** closes the program. Pressing **V** outputs the current camera coordinate system to console and **F** toggles the fast movement mode.


## Scenes and geometries

Supported primitives include...

* Triangle
* Sphere
* Box

Scattering process is Lambertian.

![Platonic solids](/figures/25000_sample_Platon.png)

3D triangle mesh can be imported to the program from **.obj** file - see an example below. TODO: Integrate mesh loading with the configuration file

![Teacup](/figures/teacup_N10000.png)

To further increase the efficiency, a bounding volume hierarchy (BVH) structure is used as an acceleration structure.

CSG shapes based on UNION, INTERSECTION and DIFFERENCE and built from 3D primitives are supported. The intersection algorithm should be optimized further. Currently large CSG objects may take a long time to render.

![CSG](/figures/csg_N10000_2min47s.png)

![dice](/figures/dice_N10000_53min51s_figure.png)

---

## Useful resources

- Aalto university: [Programming Parallel Computers](https://ppc.cs.aalto.fi/)
- R. Allen: [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
- J. Bikker: [How to build a BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)