#include "box.hpp"

__device__ Box::Box(const Vector3D& minimumCorner, const Vector3D& maximumCorner, const Vector3D& color_, float emissivity_) {
    color = color_; emissivity = emissivity_;
    Vector3D min = minVector(minimumCorner, maximumCorner), max = maxVector(minimumCorner, maximumCorner);
    center = 0.5f*(min + max);
    max -= center;
    invmat = diagMatrix(max).inverse();
}

__device__ bool Box::rayCollision(const Ray& ray, HitInfo* hit) const {
    HitInfo hits[2];
    int n = allCollisions(ray, hits);
    for(int i = 0; i < n; i++) {
        if(hits[i].emission < 0.0f) break;
        if(hits[i].t > epsilon) {
            *hit = hits[i]; return true;
        }
    }
    return false;
}
__device__ int Box::allCollisions(const Ray& ray, HitInfo* hitlist) const {
    Vector3D to = invmat * (ray.pos - center), td = invmat * ray.dir;
    //Ray temp(invmat * ray.dir, invmat * (ray.pos - center)); // <- Normalizes direction automatically and that gives wrong results

    float t[6];
    for(int i = 0; i < 3; i++) {
        float d = 1.0f / td[i];
        float o = to[i];
        if(d > 0.0f) {
            t[2*i] = -(1 + o)*d;
            t[2*i + 1] = (1 - o)*d;
        } else {
            t[2*i] = (1 - o)*d;
            t[2*i + 1] = -(1 + o)*d;
        }
    }
    int axismin = 0, axismax = 0;
    float tmin = t[0], tmax = t[1];
    if(t[2] > tmin) {
        axismin = 1; tmin = t[2];
    }
    if(t[3] < tmax) {
        axismax = 1; tmax = t[3];
    }
    if(t[4] > tmin) {
        axismin = 2; tmin = t[4];
    }
    if(t[5] < tmax) {
        axismax = 2; tmax = t[5];
    }

    if(tmin <= tmax) {
        Vector3D n = Vector3D(0.0f,0.0f,0.0f); n[axismin] = (td[axismin] < 0.0f) ? 1.0f : -1.0f;
        hitlist[0].normal = unitVec(invmat.T()*n);
        hitlist[0].color = color;
        hitlist[0].emission = emissivity;
        hitlist[0].t = tmin;

        n = Vector3D(0.0f,0.0f,0.0f); n[axismax] = (td[axismax] > 0.0f) ? 1.0f : -1.0f;
        hitlist[1].normal = unitVec(invmat.T()*n);
        hitlist[1].color = color;
        hitlist[1].emission = emissivity;
        hitlist[1].t = tmax;
        return 2;
    }
    hitlist[0].emission = -1.0f;
    hitlist[1].emission = -1.0f;
    return 0;
}

__device__ Vector3D Box::centroid() const {return center;}

__device__ void Box::translate(const Vector3D& vec) {center += vec;}
__device__ void Box::translate(float x, float y, float z) {translate(Vector3D(x,y,z));}

__device__ void Box::rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
    center = rotateVec(center, angle, axis, axisPos); invmat = (generateRotation(angle, axis) * invmat.inverse()).inverse();
}

__device__ Vector3D Box::minBox() const {
    Vector3D corner = invmat.inverse() * Vector3D(1.0f,1.0f,1.0f); float r = corner.length();
    return center - Vector3D(r,r,r);
}
__device__ Vector3D Box::maxBox() const {
    Vector3D corner = invmat.inverse() * Vector3D(1.0f,1.0f,1.0f); float r = corner.length();
    return center + Vector3D(r,r,r);
}