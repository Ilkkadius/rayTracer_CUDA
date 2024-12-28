#include "scatteringf.hpp"

__device__ HitInfo::HitInfo() : rayDir(Vector3D()), point(Vector3D()), normal(Vector3D()), t(-1.0f) {}

__device__ HitInfo::HitInfo(const Vector3D& rayDirection, const Vector3D& hitPoint, const Vector3D& surfaceNormal, float t_, Target* target_)
: rayDir(rayDirection), point(hitPoint), t(t_), target(target_) {
    if(Dot(rayDirection, surfaceNormal) > 0.0f) { // Make sure the normal points outwards!
        normal = -surfaceNormal;
    } else {
        normal = surfaceNormal;
    }
}
    
