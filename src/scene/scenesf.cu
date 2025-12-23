#include "scenesf.hpp"

#include "CSG.hpp"

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

    /*
    HitInfo* hitlist = new HitInfo[6];
    for(int i = 0; i < 6; i++) {
        hitlist[i].t = 0.0f;
        hitlist[i].color = Vector3D(0,0,0);
        hitlist[i].normal = Vector3D(0,0,0);
        hitlist[i].emission = 0.0f;
    }
    Ray ray(Vector3D(3,29,30),Vector3D(0,0,0));
    list->targets[list->size-1]->allCollisions(ray, hitlist);
    list->targets[list->size-1]->allCollisions(ray, hitlist+4);

    for(int i = 0; i < 6; i++) {
        printf("%f %f\n", hitlist[i].t, hitlist[i].emission);
    }
    */
    
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

__device__ void Scene::CSG(TargetList* list, int capacity) {
    CSGNode* nodes = new CSGNode[3];
    nodes[0] = {1,0,CSG::DIFFERENCE};
    nodes[1] = {0,0,CSG::NONE};
    nodes[2] = {1,0,CSG::NONE};
    //CSGNode nodes[9] = {{1,0,CSG::UNION},{3,0,CSG::UNION},{5,0,CSG::INTERSECTION},{0,2,CSG::NONE},{0,2,CSG::NONE},{0,2,CSG::NONE},{7,0,CSG::DIFFERENCE},{0,2,CSG::NONE},{0,2,CSG::NONE}};
    Target** csgTargets = new Target*[2];
    csgTargets[0] = new Sphere(Vector3D(5,0,-0.5),1.5,Vector3D(0.1,0.1,0.9)); 
    csgTargets[1] = new Sphere(Vector3D(5,0,0.5),1,Vector3D(0.1,0.1,0.9));
    ConstructiveShape* csg = new ConstructiveShape(csgTargets,2,nodes,3);

    Vector3D minb = csg->minBox(), maxb = csg->maxBox(), dir = maxb - minb;

    list->targets[list->size++] = new Sphere(Vector3D(3,29,30), 15, Vector3D(1,1,1), 20);
    list->targets[list->size++] = new Sphere(Vector3D(3,29,30), 0.5, Vector3D(0.9,0.1,0.1));
    list->targets[list->size++] = csg;

    /*
    list->targets[list->size++] = new Sphere(minb, 0.1, Vector3D(1,0.1,0.1));
    list->targets[list->size++] = new Sphere(maxb, 0.1, Vector3D(0.1,0.1,1));
    
    list->targets[list->size++] = new Sphere(minb + Vector3D(dir.x,0,0), 0.1, Vector3D(1,0.1,0.1));
    list->targets[list->size++] = new Sphere(maxb - Vector3D(dir.x,0,0), 0.1, Vector3D(0.1,0.1,1));
    
    list->targets[list->size++] = new Sphere(minb + Vector3D(0,dir.y,0), 0.1, Vector3D(1,0.1,0.1));
    list->targets[list->size++] = new Sphere(maxb - Vector3D(0,dir.y,0), 0.1, Vector3D(0.1,0.1,1));
    
    list->targets[list->size++] = new Sphere(minb + Vector3D(0,0,dir.z), 0.1, Vector3D(1,0.1,0.1));
    list->targets[list->size++] = new Sphere(maxb - Vector3D(0,0,dir.z), 0.1, Vector3D(0.1,0.1,1));

    list->targets[list->size++] = new Sphere(Vector3D(0,0,0),1.5,Vector3D(0.1,0.1,0.9)); 
    list->targets[list->size++] = new Sphere(Vector3D(0,0,1),1,Vector3D(0.1,0.1,0.9));
    */
    
}

