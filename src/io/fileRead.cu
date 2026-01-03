#include "fileRead.hpp"


__host__ void fileRead::getShapeTransformation(streamHolder sh, std::string& line, uint& linenum, affine& aff) {
    findNext(sh, line, linenum); aux::uppercase(line);
    aff = affine{unitMatrix(), Vector3D(0.0f,0.0f,0.0f)};
    if(line == "TRANSFORM") {
        readTransformation(sh, line, linenum, aff);
        findNext(sh, line, linenum);
        if(line != "{") aux::error(sh.path, linenum, "parsePrimitive: Expected \"{\", found \"" + line + "\".");
    } else if(line != "{") aux::error(sh.path, linenum, "parsePrimitive: Expected \"{\", found \"" + line + "\".");
}

__host__ void fileRead::skipLine(streamHolder sh, std::string& line, uint& linenum) {
    std::getline(sh.file, line);
    linenum++; sh.ss.str(line); sh.ss.clear();
}




__host__ void fileRead::parseConfig(streamHolder sh, std::string& line, uint& linenum, Config& conf) {
    if(!findNext(sh, line, linenum) || line != "{") aux::error(sh.path, linenum, "parseConfig: Expected \"{\", found " + line + ".");

    while(findNext(sh, line, linenum)) {
        aux::uppercase(line);
        if(line == "#") {
            skipLine(sh, line, linenum);
            continue;
        } else if(line == "}") {
            return;
        } else if(line == "RENDERMODE") {
            if(!findNext(sh, line, linenum) || !Mode::parseRenderMode(line, conf.mode)) 
                aux::error(sh.path, linenum, "parseConfig: Could not parse \"" + line + "\" as rendermode.");
        } else if(line == "WIDTH") {
            if(!findNext(sh, line, linenum) || !parseInt(line, conf.cam.width))
                aux::error(sh.path, linenum, "parseConfig: Could not parse \"" + line + "\" as width.");
        } else if(line == "HEIGHT") {
            if(!findNext(sh, line, linenum) || !parseInt(line, conf.cam.height))
                aux::error(sh.path, linenum, "parseConfig: Could not parse \"" + line + "\" as height.");
        } else if(line == "SAMPLES") {
            if(!findNext(sh, line, linenum) || !parseInt(line, conf.cam.samples))
                aux::error(sh.path, linenum, "parseConfig: Could not parse \"" + line + "\" as samples.");
        } else if(line == "DEPTH") {
            if(!findNext(sh, line, linenum) || !parseInt(line, conf.cam.depth))
                aux::error(sh.path, linenum, "parseConfig: Could not parse \"" + line + "\" as depth.");
        } else if(line == "BACKGROUND") {
            if(!findNext(sh, line, linenum) || !BackgroundColor::parseBackground(line, conf.background)) 
                aux::error(sh.path, linenum, "parseConfig: Could not parse \"" + line + "\" as background.");
        } else if(line == "EYE") {
            if(!readVector3D(sh.ss, conf.cam.eye))
                aux::error(sh.path, linenum, "parseConfig: Could not parse camera eye vector. ");
        } else if(line == "DIRECTION") {
            if(!readVector3D(sh.ss, conf.cam.direction))
                aux::error(sh.path, linenum, "parseConfig: Could not parse camera direction vector.");
        } else if(line == "UP") {
            if(!readVector3D(sh.ss, conf.cam.up))
                aux::error(sh.path, linenum, "parseConfig: Could not parse camera up vector.");
        } else if(line == "SCENE") {
            if(!findNext(sh, line, linenum) || !Scene::parseScene(line, conf.scene))
                aux::error(sh.path, linenum, "parseConfig: Could not parse \"" + line + "\" as scene.");
        } else if(line == "BACKUP") {
            if(!findNext(sh, line, linenum)) aux::error("parseConfig: Could not find TRUE/FALSE for backup parameter.");
            aux::uppercase(line); if(line == "TRUE") conf.backup = true;
        } else if(line == "FOV") {
            if(!findNext(sh, line, linenum) || !parseFloat(line, conf.cam.fov))
                aux::error(sh.path, linenum, "parseConfig: Could not parse \"" + line + "\" as FOV.");
        } else {
            aux::error(sh.path, linenum, "parseConfig: Could not parse keyword.");
        }
    }
    aux::error(sh.path, linenum, "parseConfig: Config definition end not found, forgotten \"}\"?");
}

__global__ void fileRead::generateTargets(TargetList** list, targetData* data, affine* affines, int targetCount) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {

        TargetList* l = *list;
        
        for(int i = 0; i < targetCount; i++) {
            targetType type = data[i].type;
            affine aff = affines[i];
            float* p = data[i].params;
            switch(type) {
                case targetType::SPHERE:
                {
                    Sphere* obj = new Sphere(Vector3D(p[0],p[1],p[2]),p[3],Vector3D(p[4],p[5],p[6]),p[7]);
                    if(!obj) printf("generateTargets: Sphere allocation failed, i=%d.\n", i);
                    obj->affine(aff.A, aff.b);
                    l->targets[l->size++] = obj;
                    break;
                }
                case targetType::BOX:
                {
                    Box* obj = new Box(p[0],p[1],p[2],Vector3D(p[3],p[4],p[5]),p[6]);
                    if(!obj) printf("generateTargets: Box allocation failed, i=%d.\n", i);
                    obj->affine(aff.A, aff.b);
                    l->targets[l->size++] = obj;
                    break;
                }
                case targetType::TRIANGLE:
                {
                    Triangle* obj = new Triangle(Vector3D(p[0],p[1],p[2]),Vector3D(p[3],p[4],p[5]),Vector3D(p[6],p[7],p[8]),Vector3D(p[9],p[10],p[11]), p[12]);
                    if(!obj) printf("generateTargets: Triangle allocation failed, i=%d.\n", i);
                    obj->affine(aff.A, aff.b);
                    l->targets[l->size++] = obj;
                    break;
                }
            }
        }
        
    }
}

__host__ void fileRead::parsePrimitives(streamHolder sh, std::string& line, uint& linenum, TargetList** list) {
    if(!findNext(sh, line, linenum) || line != "{") aux::error(sh.path, linenum, "parsePrimitives: Expected \"{\", found " + line + ".");
    std::vector<targetData> data;
    std::vector<affine> affines;
    while(findNext(sh, line, linenum)) {
        aux::uppercase(line);
        if(line == "#") {
            skipLine(sh, line, linenum);
            continue;
        } else if(line == "}") {
            targetData* targets_d; affine* affines_d; int tcount = data.size();
            CHECK(cudaMalloc(&targets_d, tcount*sizeof(targetData)));
            CHECK(cudaMalloc(&affines_d, tcount*sizeof(affine)));

            CHECK(cudaMemcpy(targets_d, data.data(), tcount*sizeof(targetData), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(affines_d, affines.data(), tcount*sizeof(affine), cudaMemcpyHostToDevice));

            generateTargets<<<1,1>>>(list, targets_d, affines_d, tcount);
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaFree(targets_d));
            CHECK(cudaFree(affines_d));
            return;
        } else if(line == "SPHERE") {
            affine aff;
            getShapeTransformation(sh, line, linenum, aff);
            Vector3D center, color;
            float r, e;
            if(!readVector3D(sh.ss, center)) aux::error(sh.path, linenum, "Could not parse Sphere center.");
            if(!(sh.ss >> r)) aux::error(sh.path, linenum, "Could not parse Sphere radius.");
            if(!readVector3D(sh.ss, color)) aux::error(sh.path, linenum, "Could not parse Sphere color.");
            if(!(sh.ss >> e)) aux::error(sh.path, linenum, "Could not parse Sphere emission.");
            if(!findNext(sh, line, linenum) || line != "}") aux::error(sh.path, linenum, "Sphere definition end not found, missing \"}\"?");
            data.push_back({targetType::SPHERE, center.x, center.y, center.z, r, color.x, color.y, color.z, e});
            affines.push_back(aff);
        } else if(line == "BOX") {
            affine aff;
            getShapeTransformation(sh, line, linenum, aff);
            Vector3D color;
            float a, b, c, e;
            if(!(sh.ss >> a)) aux::error(sh.path, linenum, "Could not parse Box x-width.");
            if(!(sh.ss >> b)) aux::error(sh.path, linenum, "Could not parse Box y-width.");
            if(!(sh.ss >> c)) aux::error(sh.path, linenum, "Could not parse Box z-width.");
            if(!readVector3D(sh.ss, color)) aux::error(sh.path, linenum, "Could not parse Box color.");
            if(!(sh.ss >> e)) aux::error(sh.path, linenum, "Could not parse Box emission.");
            if(!findNext(sh, line, linenum) || line != "}") aux::error(sh.path, linenum, "Box definition end not found, missing \"}\"?");
            data.push_back({targetType::BOX, a, b, c, color.x, color.y, color.z, e});
            affines.push_back(aff);
        } else if(line == "TRIANGLE") {
            affine aff;
            getShapeTransformation(sh, line, linenum, aff);
            Vector3D color, v0, v1, v2; float e;
            if(!readVector3D(sh.ss, v0)) aux::error(sh.path, linenum, "Could not parse Triangle vertex0.");
            if(!readVector3D(sh.ss, v1)) aux::error(sh.path, linenum, "Could not parse Triangle vertex1.");
            if(!readVector3D(sh.ss, v2)) aux::error(sh.path, linenum, "Could not parse Triangle vertex2.");
            if(!readVector3D(sh.ss, color)) aux::error(sh.path, linenum, "Could not parse Triangle color.");
            if(!(sh.ss >> e)) aux::error(sh.path, linenum, "Could not parse Triangle emission.");
            if(!findNext(sh, line, linenum) || line != "}") aux::error(sh.path, linenum, "Triangle definition end not found, missing \"}\"?");
            data.push_back({targetType::TRIANGLE, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, color.x, color.y, color.z, e});
            affines.push_back(aff);
        } else {
            aux::error(sh.path, linenum, "parsePrimitive: Could not parse primitive type.");
        }
    }
    aux::error(sh.path, linenum, "parsePrimitive: Primitive definition end not found, forgotten \"}\"?");

}

__host__ void fileRead::parseScene(streamHolder sh, std::string& line, uint& linenum, TargetList** list) {
    if(!findNext(sh, line, linenum) || line != "{") aux::error(sh.path, linenum, "parseScene: Expected \"{\", found " + line + ".");
    
    while(findNext(sh, line, linenum)) {
        aux::uppercase(line);
        if(line == "#") {
            skipLine(sh, line, linenum);
            continue;
        } else if(line == "}") {
            return;
        } else if(line == "CSG") {
            csgRead::parseCSG(sh, line, linenum, list);
        } else if(line == "PRIMITIVE") {
            parsePrimitives(sh, line, linenum, list);
        } else {
            aux::error(sh.path, linenum, "parseScene: Could not parse keyword.");
        }
    }
    aux::error(sh.path, linenum, "parseScene: Scene definition end not found, forgotten \"}\"?");
}

__host__ void fileRead::parseFile(const char* path, TargetList** list, Config& conf) {
    std::ifstream file(path);
    std::string line;
    std::stringstream ss;
    uint linenum = 0;

    streamHolder sh{path, file, ss};

    while(findNext(sh, line, linenum)) {
        aux::uppercase(line);
        if(line == "CONFIG") {
            parseConfig(sh, line, linenum, conf);
        } else if(line == "SCENE") {
            parseScene(sh, line, linenum, list);
        } else if(line == "#") {
            skipLine(sh, line, linenum);
            continue;
        } else {
            aux::error(sh.path, linenum, "parseFile: Unknown keyword detected, expected Config/Scene.");
        }
    }
}