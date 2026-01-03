#include "scenesf.hpp"

#include "CSG.hpp"

__host__ bool Scene::parseScene(std::string line, sceneType& type) {
    aux::uppercase(line);
    if(line == "EMPTY" || line == "VOID") {
        type = sceneType::EMPTY;
    } else if(line == "LIGHT") {
        type = sceneType::LIGHT;
    } else if(line == "CSG") {
        type = sceneType::CSG;
    } else if(line == "TEST") {
        type = sceneType::TEST;
    } else {
        return false;
    }
    return true;
}


__device__ void Scene::testScene(TargetList* list) {
    //float r = 500;
    Vector3D red(0.9,0.1,0.1), green(0.1,0.9,0.1), blue(0.1,0.1,0.9), white(1,1,1), black(0,0,0);

    list->targets[list->size++] = new Sphere(Vector3D(3,29,30), 15, Vector3D(1,1,1), 20);

    list->targets[list->size++] = new Sphere(Vector3D(10,3,1), 1, green);

    list->targets[list->size++] = new Sphere(Vector3D(10,-3,1), 1, blue);

}

__device__ void Scene::empty(TargetList* list) {}

__device__ void Scene::Platon(TargetList* list) {

    list->targets[list->size++] = new Sphere(Vector3D(3,29,30), 15, Vector3D(1,1,1), 20);
    
    Icosahedron* icosa = new Icosahedron(Vector3D(9,5.5,-2), 2, Vector3D(0.5*0.9,0.5*0.9,0.9*0.2));
    icosa->copyToList(list);
    delete icosa;
    
    Dodecahedron* dodeca = new Dodecahedron(Vector3D(9, -5.3, -2), 2, Vector3D(0.9*0.05, 0.9*0.2, 0.9*0.9));
    dodeca->rotate(0.2,Vector3D(0,0,1));
    dodeca->copyToList(list);
    delete dodeca;
    
    
    Tetrahedron* tetra = new Tetrahedron(Vector3D(6,4,3.5), 2, Vector3D(0.6,0.2,0.9));
    tetra->translate(Vector3D(1,0,0));
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

__device__ void Scene::CSG(TargetList* list) {

    list->targets[list->size++] = new Sphere(Vector3D(3,29,30), 15, Vector3D(1,1,1), 20);

    {
    int nodeCount = 3, targetCount = 2;
    CSGNode* nodes = new CSGNode[nodeCount];
    nodes[0] = {1,0,CSG::DIFFERENCE};
    nodes[1] = {0,0,CSG::NONE};
    nodes[2] = {1,0,CSG::NONE};
    Target** csgTargets = new Target*[targetCount];
    csgTargets[0] = new Sphere(Vector3D(5,0,-0.5),1.5,Vector3D(0.1,0.1,0.9)); 
    csgTargets[1] = new Sphere(Vector3D(4.5,0,0.5),1,Vector3D(0.9,0.1,0.1));
    list->targets[list->size++] = new ConstructiveShape(csgTargets,targetCount,nodes,nodeCount);
    }

    
    {
    int nodeCount = 3, targetCount = 2;
    CSGNode* nodes = new CSGNode[nodeCount];
    nodes[0] = {1,0,CSG::UNION};
    nodes[1] = {0,0,CSG::NONE};
    nodes[2] = {1,0,CSG::NONE};
    Target** csgTargets = new Target*[targetCount];
    csgTargets[0] = new Sphere(Vector3D(5,-4,-0.5),1.5,Vector3D(0.1,0.1,0.9)); 
    csgTargets[1] = new Sphere(Vector3D(5,-3.5,0.5),1,Vector3D(0.9,0.1,0.1));
    list->targets[list->size++] = new ConstructiveShape(csgTargets,targetCount,nodes,nodeCount);
    }
    
    {
    int nodeCount = 3, targetCount = 2;
    CSGNode* nodes = new CSGNode[nodeCount];
    nodes[0] = {1,0,CSG::INTERSECTION};
    nodes[1] = {0,0,CSG::NONE};
    nodes[2] = {1,0,CSG::NONE};
    Target** csgTargets = new Target*[targetCount];
    csgTargets[0] = new Sphere(Vector3D(5,4,-0.4),1.5,Vector3D(0.1,0.1,0.9)); 
    csgTargets[1] = new Sphere(Vector3D(4.5,3.9,0.4),1,Vector3D(0.9,0.1,0.1));
    list->targets[list->size++] = new ConstructiveShape(csgTargets,targetCount,nodes,nodeCount);
    }
    
}


__device__ void Scene::light(TargetList* list) {
    list->targets[list->size++] = new Sphere(Vector3D(3,29,30), 15, Vector3D(1,1,1), 20);

    Vector3D dir = unitVec(Vector3D(-1,1,1)), center = Vector3D(5,0,-0.5);
    {
    int nodeCount = 5, targetCount = 3;
    CSGNode* nodes = new CSGNode[nodeCount];
    nodes[0] = {1,0,CSG::DIFFERENCE};
    nodes[1] = {3,0,CSG::UNION};
    nodes[2] = {0,0,CSG::NONE};
    nodes[3] = {1,0,CSG::NONE};
    nodes[4] = {2,0,CSG::NONE};
    Target** csgTargets = new Target*[targetCount];
    csgTargets[1] = new Sphere(center,1.5,Vector3D(0.1,0.1,0.9)); 
    csgTargets[2] = new Sphere(center + 0.25*dir + Vector3D(0,0,0.85),1,Vector3D(0.9,0.1,0.1));
    csgTargets[0] = new Sphere(center + 1.8*dir,1,Vector3D(0.1,0.9,0.1));
    ConstructiveShape* csg = new ConstructiveShape(csgTargets,targetCount,nodes,nodeCount);
    list->targets[list->size++] =  csg;
    }

    list->targets[list->size++] = new Sphere(center + 1.8*dir, 0.1, Vector3D(1,0.9,0.1), 30);
}
