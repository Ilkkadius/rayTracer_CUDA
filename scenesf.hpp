#ifndef SCENES_CUDA_HPP
#define SCENES_CUDA_HPP

#include "cuda_runtime.h"

#include "compoundf.hpp"

namespace Scene{

    __device__ void testScene(targetList** list, Target** targets, Shape** shapes, int capacity) {
        //float r = 500;
        Vector3D red(0.9,0.1,0.1), green(0.1,0.9,0.1), blue(0.1,0.1,0.9), white(1,1,1), black(0,0,0);

        shapes[0] = new Sphere(Vector3D(3,29,30), 15);
        targets[0] = new Target(shapes[0], Vector3D(1,1,1), 20);

        shapes[1] = new Sphere(Vector3D(10,3,1), 1);
        targets[1] = new Target(shapes[1], green);

        shapes[2] = new Sphere(Vector3D(10,-3,1), 1);
        targets[2] = new Target(shapes[2], blue);
        
        *list = new targetList(targets, 3, capacity);
    }

    __device__ void Platon(targetList** list, Target** targets, Shape** shapes, int capacity) {

        shapes[0] = new Sphere(Vector3D(3,29,30), 15);
        targets[0] = new Target(shapes[0], Vector3D(1,1,1), 20);
        *list = new targetList(targets, 1, capacity);

        Icosahedron icosa(Vector3D(9,5.5,-2), 2, Vector3D(0.5*0.9,0.5*0.9,0.9*0.2));
        icosa.copyToList(*list, shapes);

        Dodecahedron dodeca(Vector3D(9, -5.3, -2), 2, Vector3D(0.9*0.05, 0.9*0.2, 0.9*0.9));
        dodeca.rotate(0.2,Vector3D(0,0,1));
        dodeca.copyToList(*list, shapes);

        Tetrahedron tetra(Vector3D(6,4,3.5), 2, Vector3D(0.6,0.2,0.9));
        tetra.translate(1,0,0);
        tetra.rotate(0.15*M_PI, Vector3D(0,0,1)); tetra.rotate(-0.05*M_PI, Vector3D(0,-1,0));
        tetra.copyToList(*list, shapes);
        
        Octahedron octa(Vector3D(8,-1,2.5), 2, Vector3D(0.1,0.7,0.6));
        octa.rotate(0.17, Vector3D(0,0,1));
        octa.copyToList(*list, shapes);

        Cube cube(Vector3D(8,-6,4), 2, Vector3D(0.62, 0.11, 0.19));
        cube.rotate(0.3, Vector3D(0,0,1)); cube.rotate(0.2, Vector3D(0,1,0));
        cube.copyToList(*list, shapes);

    }

};

#endif