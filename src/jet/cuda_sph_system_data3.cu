// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/bcc_lattice_point_generator.h>
#include <jet/cuda_sph_system_data3.h>
#include <jet/sph_kernels3.h>

#include <thrust/for_each.h>

using namespace jet;
using namespace experimental;

using thrust::get;

namespace {

struct UpdateDensity {
    float h2;
    float h3;
    float m;
    float* d;

    JET_CUDA_HOST_DEVICE UpdateDensity(float h, float m_, float* d_)
        : h2(h * h), h3(h * h * h), m(m_), d(d_) {}

    inline JET_CUDA_DEVICE void operator()(size_t i, float4 o, size_t,
                                           float4 p) {
        float dist = length(o - p);
        d[i] += m * kernel(dist);
    }

    inline JET_CUDA_DEVICE float kernel(float distance) {
        if (distance * distance >= h2) {
            return 0.0f;
        } else {
            double x = 1.0f - distance * distance / h2;
            return 315.0f / (64.0f * kPiF * h3) * x * x * x;
        }
    }
};

}  // namespace

CudaSphSystemData3::CudaSphSystemData3() : CudaSphSystemData3(0) {}

CudaSphSystemData3::CudaSphSystemData3(size_t numberOfParticles)
    : CudaParticleSystemData3(numberOfParticles) {
    _densityIdx = addFloatData();
    _pressureIdx = addFloatData();

    setTargetSpacing(_targetSpacing);
}

CudaSphSystemData3::CudaSphSystemData3(const CudaSphSystemData3& other) {
    set(other);
}

CudaSphSystemData3::~CudaSphSystemData3() {}

const CudaArrayView1<float> CudaSphSystemData3::densities() const {
    return floatDataAt(_densityIdx);
}

CudaArrayView1<float> CudaSphSystemData3::densities() {
    return floatDataAt(_densityIdx);
}

const CudaArrayView1<float> CudaSphSystemData3::pressures() const {
    return floatDataAt(_pressureIdx);
}

CudaArrayView1<float> CudaSphSystemData3::pressures() {
    return floatDataAt(_pressureIdx);
}

void CudaSphSystemData3::updateDensities() {
    neighborSearcher()->forEachNearbyPoint(
        positions(), _kernelRadius,
        UpdateDensity(_kernelRadius, _mass, densities().data()));
}

float CudaSphSystemData3::targetDensity() const { return _targetDensity; }

void CudaSphSystemData3::setTargetDensity(float targetDensity) {
    _targetDensity = targetDensity;

    computeMass();
}

float CudaSphSystemData3::targetSpacing() const { return _targetSpacing; }

void CudaSphSystemData3::setTargetSpacing(float spacing) {
    _targetSpacing = spacing;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

float CudaSphSystemData3::relativeKernelRadius() const {
    return _kernelRadiusOverTargetSpacing;
}

void CudaSphSystemData3::setRelativeKernelRadius(float relativeRadius) {
    _kernelRadiusOverTargetSpacing = relativeRadius;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

float CudaSphSystemData3::kernelRadius() const { return _kernelRadius; }

void CudaSphSystemData3::setKernelRadius(float kernelRadius) {
    _kernelRadius = kernelRadius;
    _targetSpacing = kernelRadius / _kernelRadiusOverTargetSpacing;

    computeMass();
}

float CudaSphSystemData3::mass() const { return _mass; }

void CudaSphSystemData3::set(const CudaSphSystemData3& other) {
    CudaParticleSystemData3::set(other);

    _targetDensity = other._targetDensity;
    _targetSpacing = other._targetSpacing;
    _kernelRadiusOverTargetSpacing = other._kernelRadiusOverTargetSpacing;
    _kernelRadius = other._kernelRadius;
    _densityIdx = other._densityIdx;
    _pressureIdx = other._pressureIdx;
}

CudaSphSystemData3& CudaSphSystemData3::operator=(
    const CudaSphSystemData3& other) {
    set(other);
    return (*this);
}

void CudaSphSystemData3::computeMass() {
    Array1<Vector3D> points;
    BccLatticePointGenerator pointsGenerator;
    BoundingBox3D sampleBound(
        Vector3D(-1.5 * _kernelRadius, -1.5 * _kernelRadius,
                 -1.5 * _kernelRadius),
        Vector3D(1.5 * _kernelRadius, 1.5 * _kernelRadius,
                 1.5 * _kernelRadius));

    pointsGenerator.generate(sampleBound, _targetSpacing, &points);

    double maxNumberDensity = 0.0;
    SphStdKernel3 kernel(_kernelRadius);

    for (size_t i = 0; i < points.size(); ++i) {
        const Vector3D& point = points[i];
        double sum = 0.0;

        for (size_t j = 0; j < points.size(); ++j) {
            const Vector3D& neighborPoint = points[j];
            sum += kernel(neighborPoint.distanceTo(point));
        }

        maxNumberDensity = std::max(maxNumberDensity, sum);
    }

    JET_ASSERT(maxNumberDensity > 0);

    _mass = static_cast<float>(_targetDensity / maxNumberDensity);
}
