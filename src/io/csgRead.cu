#include "csgRead.hpp"

// The following non-empty character must be '{'
__host__ void csgRead::findStart(streamHolder sh, std::string& line, uint& linenum, affine& aff) {
    findNext(sh, line, linenum); aux::uppercase(line);
    if(line == "TRANSFORM") {
        readTransformation(sh, line, linenum, aff);
        findNext(sh, line, linenum);
        if(line != "{") aux::error(sh.path, linenum, "CSG find start: Expected \"{\", found \"" + line + "\".");
    } else if(line != "{") aux::error(sh.path, linenum, "CSG find start: Expected \"{\", found \"" + line + "\".");
}

// The following non-empty character must be '}'
__host__ void csgRead::findEnd(streamHolder sh, std::string& line, uint& linenum) {
    findNext(sh, line, linenum);
    aux::uppercase(line);
    if(line != "}") aux::error(sh.path, linenum, "CSG find end: Expected \"}\", found \"" + line + "\".");
}

__host__ void csgRead::skipTransformation(streamHolder sh, std::string& line, uint& linenum) {
    findNext(sh, line, linenum); aux::uppercase(line);
    if(line == "TRANSFORM") {
        while(findNext(sh, line, linenum) && line != "}") {}
        findNext(sh, line, linenum);
        if(line != "{") aux::error(sh.path, linenum, "Skip transformation: Expected \"{\", found \"" + line + "\".");
    } else if(line != "{") aux::error(sh.path, linenum, "Skip transformation: Expected \"{\", found \"" + line + "\".");
}



__host__ bool csgRead::readTargets(const char* path, std::vector<targetData>& data, uint linenum) {
    int counter = 1;
    std::ifstream file(path);
    std::string line; std::stringstream ss;
    streamHolder sh{path, file, ss};
    for(int i = 0; i < linenum; i++) std::getline(file,line);
    bool ready = false;
    while(std::getline(file, line)) {
        linenum++;
        ss.str(line); ss.clear();
        if(ss >> line) {
            aux::uppercase(line);
            if(line == "SPHERE") {
                Vector3D center, color;
                float r, e;
                skipTransformation(sh, line, linenum);
                if(!readVector3D(ss, center)) aux::error(path, linenum, "Could not parse Sphere center.");
                if(!(ss >> r)) aux::error(path, linenum, "Could not parse Sphere radius.");
                if(!readVector3D(ss, color)) aux::error(path, linenum, "Could not parse Sphere color.");
                if(!(ss >> e)) aux::error(path, linenum, "Could not parse Sphere emission.");
                data.push_back({targetType::SPHERE, center.x, center.y, center.z, r, color.x, color.y, color.z, e});
                findEnd(sh, line, linenum);
            } else if(line == "BOX") {
                Vector3D color;
                float a, b, c, e;
                skipTransformation(sh, line, linenum);
                if(!(ss >> a)) aux::error(path, linenum, "Could not parse Box x-width.");
                if(!(ss >> b)) aux::error(path, linenum, "Could not parse Box y-width.");
                if(!(ss >> c)) aux::error(path, linenum, "Could not parse Box z-width.");
                if(!readVector3D(ss, color)) aux::error(path, linenum, "Could not parse Box color.");
                if(!(ss >> e)) aux::error(path, linenum, "Could not parse Box emission.");
                data.push_back({targetType::BOX, a, b, c, color.x, color.y, color.z, e});
                findEnd(sh, line, linenum);
            } else if(line == "{") {
                counter++;
            } else if(line == "}") {
                counter--;
            }
        }
        while(ss >> line) {
            if(line == "{") {
                counter++;
            } else if(line == "}") {
                counter--;
            }
        }
        if(counter <= 0) {
            ready = true; break;
        }
    }
    return ready;
}

__host__ void csgRead::processLeft(streamHolder sh, int ptr, std::string& line, std::vector<uint>& firstList, std::vector<CSG>& operList, std::vector<affine>& affines, int& nodeCounter, int& targetCounter, uint& linenum, affine aff) {
    while(std::getline(sh.file, line)) {
        linenum++; sh.ss.str(line); sh.ss.clear();
        while(sh.ss >> line) {
            aux::uppercase(line);
            if(line[0] == '#') break;
            else if(line == "UNION" || line == "INTERSECTION" || line == "DIFFERENCE") {

                if(line == "UNION") operList[ptr] = CSG::UNION;
                else if(line == "INTERSECTION") operList[ptr] = CSG::INTERSECTION;
                else operList[ptr] = CSG::DIFFERENCE;

                findStart(sh, line, linenum, aff);
                
                int first = nodeCounter; nodeCounter += 2;
                firstList[ptr] = first;
                csgRead::processLeft(sh, first, line, firstList, operList, affines, nodeCounter, targetCounter, linenum, aff);
                csgRead::processRight(sh, first+1, line, firstList, operList, affines, nodeCounter, targetCounter, linenum, aff);

                findEnd(sh, line, linenum);

                return;

            } else if(line == "SPHERE" || line == "BOX") {

                findStart(sh, line, linenum, aff);

                operList[ptr] = CSG::NONE;
                firstList[ptr] = targetCounter;
                affines[targetCounter++] = aff;

                while(findNext(sh, line, linenum) && line != "}") {}

                return;

            }else {
                aux::error(sh.path, linenum, "Could not parse \""+line+"\" as UNION/INTERSECTION/DIFFERENCE instance.");
            }
        }
    }
}

__host__ void csgRead::processRight(streamHolder sh, int ptr, std::string& line, std::vector<uint>& firstList, std::vector<CSG>& operList, std::vector<affine>& affines, int& nodeCounter, int& targetCounter, uint& linenum, affine aff) {
    while(std::getline(sh.file, line)) {
        linenum++; sh.ss.str(line); sh.ss.clear();
        while(sh.ss >> line) {
            aux::uppercase(line);
            if(line[0] == '#') break;
            else if(line == "UNION" || line == "INTERSECTION" || line == "DIFFERENCE") {

                if(line == "UNION") operList[ptr] = CSG::UNION;
                else if(line == "INTERSECTION") operList[ptr] = CSG::INTERSECTION;
                else operList[ptr] = CSG::DIFFERENCE;

                findStart(sh, line, linenum, aff);
                
                int first = nodeCounter; nodeCounter += 2;
                firstList[ptr] = first;
                csgRead::processLeft(sh, first, line, firstList, operList, affines, nodeCounter, targetCounter, linenum, aff);
                csgRead::processRight(sh, first + 1, line, firstList, operList, affines, nodeCounter, targetCounter, linenum, aff);

                findEnd(sh, line, linenum);

                return;

            } else if(line == "SPHERE" || line == "BOX") {

                findStart(sh, line, linenum, aff);

                operList[ptr] = CSG::NONE;
                firstList[ptr] = targetCounter;
                affines[targetCounter++] = aff;

                while(findNext(sh, line, linenum) && line != "}") {}

                return;

            }else {
                aux::error(sh.path, linenum, "Could not parse \""+line+"\" as UNION/INTERSECTION/DIFFERENCE instance.");
            }
        }
    }
}

__global__ void csgRead::generateCSG(TargetList** list, uint* firstList, CSG* operList, int nodeCount, targetData* data, affine* affines, int targetCount) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {

        CSGNode* nodes = new CSGNode[nodeCount];
        if(!nodes) printf("generateCSG: node allocation failed.\n");

        TargetList* l = *list;

        for(int i = 0; i < nodeCount; i++) {
            nodes[i] = {firstList[i], 0, operList[i]};
        }

        Target** targets = new Target*[targetCount];
        if(!targets) printf("generateCSG: target allocation failed.\n");
        
        for(int i = 0; i < targetCount; i++) {
            targetType type = data[i].type;
            affine aff = affines[i];
            float* p = data[i].params;
            switch(type) {
                case targetType::SPHERE:
                {
                    Sphere* obj = new Sphere(Vector3D(p[0],p[1],p[2]),p[3],Vector3D(p[4],p[5],p[6]),p[7]);
                    if(!obj) printf("generateCSG: Sphere allocation failed, i=%d.\n", i);
                    obj->affine(aff.A, aff.b);
                    targets[i] = obj;
                    break;
                }
                case targetType::BOX:
                {
                    Box* obj = new Box(p[0],p[1],p[2],Vector3D(p[3],p[4],p[5]),p[6]);
                    if(!obj) printf("generateCSG: Box allocation failed, i=%d.\n", i);
                    obj->affine(aff.A, aff.b);
                    targets[i] = obj;
                    break;
                }
            }
        }
        ConstructiveShape* obj = new ConstructiveShape(targets, targetCount, nodes, nodeCount);
        if(!obj) printf("generateCSG: CSG shape allocation failed.\n");
        l->targets[l->size++] = obj;
    }
}

__host__ void csgRead::parseCSG(streamHolder sh, std::string& line, uint& linenum,  TargetList** list) {
    if(!findNext(sh, line, linenum) || line != "{") aux::error(sh.path, linenum, "CSG: Expected \"{\", found " + line + ".");
    while(std::getline(sh.file, line)) {
        linenum++;
        sh.ss.str(line); sh.ss.clear(); // Removes previous error flags
        while(sh.ss >> line) {
            aux::uppercase(line);
            if(line[0] == '#') {
                break;
            } else if(line == "}") {
                return;
            } else if(line == "UNION" || line == "INTERSECTION" || line == "DIFFERENCE") {

                std::vector<uint> firstList;
                std::vector<CSG> operList;
                std::vector<targetData> targets;
                std::vector<affine> affines;
                affine aff{unitMatrix(), Vector3D(0.0f,0.0f,0.0f)};

                if(!readTargets(sh.path, targets, linenum)) aux::error(sh.path, linenum, "Could not parse CSG object. Missing curly braces: \"{\", \"}\" ?");
                if(targets.size() < 1) aux::error(sh.path, linenum, "CSG: No targets found.");
                firstList.resize(2*targets.size()); operList.resize(2*targets.size());
                affines.resize(targets.size());


                firstList[0] = 1;
                if(line == "UNION") operList[0] = CSG::UNION;
                else if(line == "INTERSECTION") operList[0] = CSG::INTERSECTION;
                else operList[0] = CSG::DIFFERENCE;

                int nodeCounter = 3; int targetCounter = 0;

                findStart(sh, line, linenum, aff);

                processLeft(sh, 1, line, firstList, operList, affines, nodeCounter, targetCounter, linenum, aff);
                processRight(sh, 2, line, firstList, operList, affines, nodeCounter, targetCounter, linenum, aff);

                findEnd(sh, line, linenum);

                firstList.resize(nodeCounter); operList.resize(nodeCounter);

                uint* firstList_d; CSG* operList_d; targetData* targets_d; affine* affines_d;
                CHECK(cudaMalloc(&firstList_d, nodeCounter*sizeof(uint)));
                CHECK(cudaMalloc(&operList_d, nodeCounter*sizeof(CSG)));
                CHECK(cudaMalloc(&targets_d, targetCounter*sizeof(targetData)));
                CHECK(cudaMalloc(&affines_d, targetCounter*sizeof(affine)));

                CHECK(cudaMemcpy(firstList_d, firstList.data(), nodeCounter*sizeof(uint), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(operList_d, operList.data(), nodeCounter*sizeof(CSG), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(targets_d, targets.data(), targetCounter*sizeof(targetData), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(affines_d, affines.data(), targetCounter*sizeof(affine), cudaMemcpyHostToDevice));

                generateCSG<<<1,1>>>(list, firstList_d, operList_d, nodeCounter, targets_d, affines_d, targetCounter);
                CHECK(cudaDeviceSynchronize());

                CHECK(cudaFree(firstList_d));
                CHECK(cudaFree(operList_d));
                CHECK(cudaFree(targets_d));
                CHECK(cudaFree(affines_d));
            }
        }
    }
    aux::error(sh.path, linenum, "CSG: Definition end not found, forgotten \"}\"?");
}