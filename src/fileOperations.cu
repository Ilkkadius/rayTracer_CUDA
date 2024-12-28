#include "fileOperations.hpp"

__host__ bool FileOperations::ReadFunctions::readVertex(std::string& str, dynVec<Vector3D>& vertices) {
    std::string v; double X, Y, Z;
    std::stringstream(str) >> v >> X >> Y >> Z;
    if(v != "v") {
        return false;
    }
    vertices.push_back(Vector3D(X,Y,Z));
    return true;
}

__host__ bool FileOperations::ReadFunctions::readUV(std::string& str, dynVec<double*>& uv) {
    std::string vt; double U, V;
    std::stringstream(str) >> vt >> U >> V;
    if(vt != "vt") {
        return false;
    }
    double* d = new double[2]; d[0] = U; d[1] = V;
    uv.push_back(d);
    return true;
}

__host__ bool FileOperations::ReadFunctions::readNormal(std::string& str, dynVec<Vector3D>& normals) {
    std::string vn; double X, Y, Z;
    std::stringstream(str) >> vn >> X >> Y >> Z;
    if(vn != "vn") {
        return false;
    }
    normals.push_back(Vector3D(X,Y,Z));
    return true;
}

__host__ int FileOperations::ReadFunctions::FaceElemIdx(const std::string& faceElem, int elemIdx) {
    std::stringstream ss(faceElem);
    int index; std::string buffer;
    for(int i = 0; i < elemIdx + 1; i++)
        std::getline(ss, buffer, '/');
    if(std::stringstream(buffer) >> index)
        return index;
    return -1;
}

__host__ bool FileOperations::ReadFunctions::FaceSepColor(std::string& line, std::string& faceData, Vector3D& faceColor) {
    if(std::getline(std::stringstream(line), faceData, '#')) {
        if(line.length() > faceData.length()) {
            double r, g, b;
            faceColor = (std::stringstream(line.substr(faceData.length() + 1)) >> r >> g >> b) ? Vector3D(r, g, b) : faceColor;
            std::cout << faceColor << std::endl;
        }
        return true;
    }
    return false;
}

__host__ bool FileOperations::ReadFunctions::readTriangle(std::stringstream& ss, std::string& firstVertex, 
                                                            std::string& secondVertex, dynVec<int*>& fVertices, dynVec<int*>& fNormals) {
    std::string v;
    if(!(ss >> v)) return false;

    int v0 = FaceElemIdx(firstVertex, 0);
    int v1 = FaceElemIdx(secondVertex, 0);
    int v2 = FaceElemIdx(v, 0);
    int* vs = new int[3]; vs[0] = v0 - 1; vs[1] = v1 - 1; vs[2] = v2 - 1; // CORRECT INDEXING
    fVertices.push_back(vs);

    int n0 = FaceElemIdx(firstVertex, 2);
    int n1 = FaceElemIdx(secondVertex, 2);
    int n2 = FaceElemIdx(v, 2);
    int* ns = new int[3]; ns[0] = n0 - 1; ns[1] = n1 - 1; ns[2] = n2 - 1; // CORRECT INDEXING
    (n0 > 0 && n1 > 0 && n2 > 0) ? fNormals.push_back(ns) : fNormals.push_back(NULL);

    secondVertex = v;
    
    return true;
}


__host__ bool FileOperations::ReadFunctions::readFace(std::string& line, dynVec<int*>& fVertices, 
                                                        dynVec<Vector3D>& fColors, dynVec<int*>& fNormals) {
    std::string faceData;
    Vector3D faceColor(-1.0,-1.0,-1.0);

    if(!FaceSepColor(line, faceData, faceColor)) // Separate face data and get the (possible) color of the face
        return false;
        
    std::string f, first, second;
    std::stringstream ss(faceData);
    
    if( !(ss >> f >> first >> second) && (f != "f") ) return false;
    while(readTriangle(ss, first, second, fVertices, fNormals))
        fColors.push_back(faceColor);
    return true;
}

__host__ bool FileOperations::ReadFunctions::ReadGroup(std::ifstream& is, dynVec<Vector3D>& vertices, dynVec<double*>& uv, 
                                                        dynVec<Vector3D>& normals, dynVec<int*>& fVertices, dynVec<int*>& fNormals, 
                                                        dynVec<Vector3D>& fColors) {
    std::string line;
    bool flag = false;
    if(is.is_open()) {
        while(std::getline(is, line)) {
            if(line[0] == 'v') {
                std::string str = line.substr(0,2);
                if(str == "v ") {
                    ReadFunctions::readVertex(line, vertices);
                } else if(str == "vt") {
                    //ReadFunctions::readUV(line, uv);
                } else if(str == "vn") {
                    ReadFunctions::readNormal(line, normals);
                }
            } else if(line[0] == 'f') {
                flag = true;
                ReadFunctions::readFace(line, fVertices, fColors, fNormals);
            } else {
                if(flag) { // If a vertex group has been read completely, i.e. f lines have ended, break from the loop
                    break;
                }
            }
        }
        if(!is.good()) {
            is.close();
            return false;
        }
        return true;
    }
    return false;
}
        
__host__ bool FileOperations::TargetsFromFile(const char* path, TargetList** list, Shape** shapes, const Vector3D& defaultColor) {
    dynVec<Vector3D> vertices(1<<7);                        // Store all vertices "v"
    dynVec<Vector3D> fColors(1<<7);                         // Store colors "#"
    dynVec<double*> uv(1<<7);                               // Store uv data "vt"
    dynVec<Vector3D> normals(1<<7);                         // Store all normals "vn"
    dynVec<int*> fVertices(1<<7), fNormals(1<<7);           // Save the indices given by face elements f/vt/fn

    std::ifstream is(path);
    if(!is.is_open()) {
        std::cout << "Error in opening the file to read targets" << std::endl;
        return false;
    }
    
    while(ReadFunctions::ReadGroup(is, vertices, uv, normals, fVertices, fNormals, fColors)) {}
    
    Vector3D* verts, * fCols; int* fVerts;
    CHECK(cudaMallocManaged(&verts, vertices.size() * sizeof(Vector3D)));
    CHECK(cudaMallocManaged(&fCols, fColors.size() * sizeof(Vector3D)));
    CHECK(cudaMallocManaged(&fVerts, 3 * fVertices.size() * sizeof(int*)));
    CHECK(cudaDeviceSynchronize());
    
    for(int i = 0; i < vertices.size(); i++) {
        verts[i] = vertices[i];
    }
    for(int i = 0; i < fVertices.size(); i++) {
        int idx = 3*i; int* fv = fVertices[i];
        fVerts[idx] = fv[0];
        fVerts[idx + 1] = fv[1];
        fVerts[idx + 2] = fv[2];
    }

    for(int i = 0; i < fColors.size(); i++) {
        fCols[i] = fColors[i];
    }

    Vector3D* defCol;
    CHECK(cudaMallocManaged(&defCol, sizeof(Vector3D)));
    *defCol = defaultColor;

    generateTargets<<<1, 1>>>(list, shapes, verts, fVerts, fCols, fVertices.size(), defCol);
    CHECK(cudaDeviceSynchronize());

    if(is.is_open()) {
        is.close();
    }

    for(int i = 0; i < uv.size(); i++) {
        delete[] uv[i];
    }
    for(int i = 0; i < fVertices.size(); i++) {
        delete[] fVertices[i];
    }
    for(int i = 0; i < fNormals.size(); i++) {
        delete[] fNormals[i];
    }

    CHECK(cudaFree(verts));
    CHECK(cudaFree(fCols));
    CHECK(cudaFree(fVerts));

    return true;
}

__host__ bool FileOperations::CompoundsFromFile(const char* path, Compound** list, size_t& listSize, const Vector3D& defaultColor) {
    dynVec<Vector3D> vertices(1<<7);                        // Store all vertices "v"
    dynVec<Vector3D> fColors(1<<7);                         // Store colors "#"
    dynVec<double*> uv(1<<7);                               // Store uv data "vt"
    dynVec<Vector3D> normals(1<<7);                         // Store all normals "vn"
    dynVec<int*> fVertices(1<<7), fNormals(1<<7);           // Save the indices given by face elements f/vt/fn

    std::ifstream is(path);
    if(!is.is_open()) {
        std::cout << "Error in opening the file to read targets" << std::endl;
        return false;
    }
    
    while(ReadFunctions::ReadGroup(is, vertices, uv, normals, fVertices, fNormals, fColors)) {}
    
    Vector3D* verts, * fCols; int* fVerts;
    CHECK(cudaMallocManaged(&verts, vertices.size() * sizeof(Vector3D)));
    CHECK(cudaMallocManaged(&fCols, fColors.size() * sizeof(Vector3D)));
    CHECK(cudaMallocManaged(&fVerts, 3 * fVertices.size() * sizeof(int*)));
    CHECK(cudaDeviceSynchronize());
    
    for(int i = 0; i < vertices.size(); i++) {
        verts[i] = vertices[i];
    }
    for(int i = 0; i < fVertices.size(); i++) {
        int idx = 3*i; int* fv = fVertices[i];
        fVerts[idx] = fv[0];
        fVerts[idx + 1] = fv[1];
        fVerts[idx + 2] = fv[2];
    }

    for(int i = 0; i < fColors.size(); i++) {
        fCols[i] = fColors[i];
    }

    Vector3D* defCol;
    CHECK(cudaMallocManaged(&defCol, sizeof(Vector3D)));
    *defCol = defaultColor;

    generateCompounds<<<1, 1>>>(list, verts, fVerts, fCols, fVertices.size(), defCol); listSize = 1;
    CHECK(cudaDeviceSynchronize());

    if(is.is_open()) {
        is.close();
    }

    for(int i = 0; i < uv.size(); i++) {
        delete[] uv[i];
    }
    for(int i = 0; i < fVertices.size(); i++) {
        delete[] fVertices[i];
    }
    for(int i = 0; i < fNormals.size(); i++) {
        delete[] fNormals[i];
    }

    CHECK(cudaFree(verts));
    CHECK(cudaFree(fCols));
    CHECK(cudaFree(fVerts));

    return true;
}

