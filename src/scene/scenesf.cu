#include "scenesf.hpp"


__device__ void Scene::testScene(TargetList* list, int capacity) {
    //float r = 500;
    Vector3D red(0.9,0.1,0.1), green(0.1,0.9,0.1), blue(0.1,0.1,0.9), white(1,1,1), black(0,0,0);

    list->targets[list->size++] = new Sphere(Vector3D(3,29,30), 15, Vector3D(1,1,1), 20);

    list->targets[list->size++] = new Sphere(Vector3D(10,3,1), 1, green);

    list->targets[list->size++] = new Sphere(Vector3D(10,-3,1), 1, blue);

}

__device__ void Scene::empty(TargetList* list, int capacity) {}

__device__ void Scene::Platon(TargetList* list, int capacity) {

    list->targets[list->size++] = new Sphere(Vector3D(3,29,30), 15, Vector3D(1,1,1), 20);
    
    Icosahedron* icosa = new Icosahedron(Vector3D(9,5.5,-2), 2, Vector3D(0.5*0.9,0.5*0.9,0.9*0.2));
    icosa->copyToList(list);
    delete icosa;
    
    Dodecahedron* dodeca = new Dodecahedron(Vector3D(9, -5.3, -2), 2, Vector3D(0.9*0.05, 0.9*0.2, 0.9*0.9));
    dodeca->rotate(0.2,Vector3D(0,0,1));
    dodeca->copyToList(list);
    delete dodeca;
    
    
    Tetrahedron* tetra = new Tetrahedron(Vector3D(6,4,3.5), 2, Vector3D(0.6,0.2,0.9));
    tetra->translate(1,0,0);
    tetra->rotate(0.15*M_PI, Vector3D(0,0,1)); 
    tetra->rotate(-0.05*M_PI, Vector3D(0,-1,0));
    tetra->copyToList(list);
    delete tetra;
    
    
    Octahedron* octa = new Octahedron(Vector3D(8,-1,2.5), 2, Vector3D(0.1,0.7,0.6));
    octa->rotate(0.17, Vector3D(0,0,1));
    octa->copyToList(list);
    delete octa;
    
    
    Cube* cube = new Cube(Vector3D(8,-6,4), 2, Vector3D(0.62, 0.11, 0.19));
    cube->rotate(0.3, Vector3D(0,0,1)); 
    cube->rotate(0.2, Vector3D(0,1,0));
    cube->copyToList(list);
    delete cube;
    
}

