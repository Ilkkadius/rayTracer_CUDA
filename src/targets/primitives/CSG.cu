#include "CSG.hpp"

__device__ ConstructiveShape::ConstructiveShape(Target** targetList, uint numTargets, CSGNode* nodeList, uint numNodes) : targets(targetList), targetCount(numTargets), nodes(nodeList), nodeCount(numNodes) {
    buildPostorder();
}

__device__ void ConstructiveShape::handleHits(const CSGNode& node, int lastHitIdx, HitInfo* hitlist, HitInfo* aux) const {
    int lmax = nodes[node.first].maxCollisionCount, rmax = nodes[node.first+1].maxCollisionCount;
    HitInfo* rhits = hitlist + lastHitIdx - rmax;
    HitInfo* lhits = rhits - lmax;
    int nl = lmax, nr = rmax;
    for(int i = 0; i < lmax; i++) {
        if(lhits[i].emission < 0.0f) {
            nl = i+1;
            break;
        }
    }
    for(int i = 0; i < rmax; i++) {
        if(rhits[i].emission < 0.0f) {
            nr = i+1;
            break;
        }
        if(node.oper == CSG::DIFFERENCE) rhits[i].normal = -1.0*rhits[i].normal;
    }

    int sortedPtr = 0, il = 0, ir = 0; bool inl = false, inr = false, lhit;
    while(il < nl && ir < nr) {
        HitInfo hl = lhits[il], hr = rhits[ir];
        lhit = hl.t < hr.t;

        switch(node.oper) {
            case CSG::UNION:
                if((lhit && !inr) || (!lhit && !inl)) aux[sortedPtr++] = lhit ? hl : hr;
                break;
            case CSG::DIFFERENCE:
                if((lhit && !inr) || (!lhit && inl)) aux[sortedPtr++] = lhit ? hl : hr;
                break;
            case CSG::INTERSECTION:
                if((lhit && inr) || (!lhit && inl)) aux[sortedPtr++] = lhit ? hl : hr;
                break;
        }
        if(lhit) {
            inl = !inl; il++;
        } else {
            inr = !inr; ir++;
        }
    }
    if(node.oper != CSG::INTERSECTION && ((il < nl) || (ir < nr))) {
        if((il < nl) && node.oper == CSG::DIFFERENCE) {
            while(il < nl) aux[sortedPtr++] = lhits[il++];
        } else if(node.oper == CSG::UNION) {
            while(il < nl) aux[sortedPtr++] = lhits[il++];
            while(ir < nr) aux[sortedPtr++] = rhits[ir++];
        }
    }
    while(sortedPtr < lmax+rmax) aux[sortedPtr++] = HitInfo(Vector3D(),Vector3D(),-1.0f,-1.0f);
    for(int i = 0; i < lmax+rmax; i++) {
        lhits[i] = aux[i];
    }

}

__device__ bool ConstructiveShape::rayCollision(const Ray& ray, HitInfo* hit) const {

    HitInfo* hitlist = new HitInfo[2*maxCollisions()];
    HitInfo* aux = hitlist + maxCollisions();
    int hitPtr = 0;

    for(int i = 0; i < nodeCount; i++) {
        CSGNode node = nodes[nodePostorder[i]];
        if(node.oper == CSG::NONE) {
            int h = targets[node.first]->allCollisions(ray, hitlist + hitPtr);
            hitPtr += node.maxCollisionCount;
        } else {
            handleHits(node, hitPtr, hitlist, aux);
        }
    }

    for(int i = 0; i < maxCollisions(); i++) {
        if(hitlist[i].t > 0.0f) {
            *hit = hitlist[i];
            delete[] hitlist;
            return true;
        }
    }
    delete[] hitlist;
    return false;
}
__device__ int ConstructiveShape::allCollisions(const Ray& ray, HitInfo* hitlist) const {return 0;}

__device__ Vector3D ConstructiveShape::centroid() const {
    return 0.5f * (minBox() + maxBox()); 
}

__device__ void ConstructiveShape::translate(const Vector3D& vec) {

}
__device__ void ConstructiveShape::translate(float x, float y, float z) {

}

__device__ void ConstructiveShape::rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {

}

__device__ Vector3D ConstructiveShape::emission() const {return Vector3D();}

__device__ Vector3D ConstructiveShape::minBox() const {
    if(targetCount < 1) return Vector3D(0,0,0);
    Vector3D minbox = targets[nodes[nodePostorder[0]].first]->minBox();
    for(int i = 1; i < nodeCount; i++) {
        CSGNode n = nodes[nodePostorder[i]];
        if(n.oper == CSG::NONE) minbox = minVector(minbox, targets[n.first]->minBox());
    }
    return minbox;
}
__device__ Vector3D ConstructiveShape::maxBox() const {
    if(targetCount < 1) return Vector3D(0,0,0);
    Vector3D maxbox = targets[nodes[nodePostorder[0]].first]->maxBox();
    for(int i = 1; i < nodeCount; i++) {
        CSGNode n = nodes[nodePostorder[i]];
        if(n.oper == CSG::NONE) maxbox = maxVector(maxbox, targets[n.first]->maxBox());
    }
    return maxbox;
}
