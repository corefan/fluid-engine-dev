// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/cuda_sph_solver3.h>
#include <jet/cuda_utils.h>
#include <jet/timer.h>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>

#include <algorithm>

using namespace jet;
using namespace experimental;
using thrust::get;
using thrust::make_tuple;
using thrust::make_zip_iterator;

static double kTimeStepLimitBySpeedFactor = 0.4;
static double kTimeStepLimitByForceFactor = 0.25;

namespace {
#if 0
struct ResolveCollision {
    __host__ __device__ ResolveCollision() {}

    __device__ void operator()(float4* newPosition, float4* newVelocity) {
        if (newPosition->y < 0.0f) {
            newPosition->y = 0.0f;
            if (newVelocity->y < 0.0f) {
                newVelocity->y *= -1.0;
            }
        }
    }
};

struct AccumulateExternalForces {
    float mass;
    float4 gravity;

    AccumulateExternalForces(float m, float4 g) : mass(m), gravity(g) {}

    __device__ float4 operator()(const float4& initialForce) {
        // Gravity
        float4 force = mass * gravity + initialForce;
        return force;
    }
};

struct TimeIntegration {
    float mass;
    float timeStepInSeconds;

    TimeIntegration(float m, float dt) : mass(m), timeStepInSeconds(dt) {}

    __device__ void operator()(const float4& p0, const float4& v0,
                               const float4& f, float4* p1, float4* v1) const {
        *v1 = v0 + timeStepInSeconds * f / mass;
        *p1 = p0 + timeStepInSeconds * (*v1);
    }
};

struct AdvanceTimeStepKernel {
    AccumulateExternalForces accExtForces;
    TimeIntegration ti;
    ResolveCollision rc;

    AdvanceTimeStepKernel(float m, float dt, float4 gravity)
        : accExtForces(m, gravity), ti(m, dt) {}

    template <typename Tuple>
    __device__ void operator()(const Tuple& t) {
        // posCurr: 0
        // velCurr: 1
        float4 p0 = get<0>(t);
        float4 v0 = get<1>(t);
        float4 p1;
        float4 v1;

        float4 f = accExtForces(make_float4(0, 0, 0, 0));
        ti(p0, v0, f, &p1, &v1);
        rc(&p1, &v1);

        get<0>(t) = p1;
        get<1>(t) = v1;
    }
};
#endif
}  // namespace

CudaSphSolver3::CudaSphSolver3()
    : CudaSphSolver3(static_cast<float>(kWaterDensity), 0.1f, 1.8f) {}

CudaSphSolver3::CudaSphSolver3(float targetDensity, float targetSpacing,
                               float relativeKernelRadius)
    : _targetDensity(targetDensity),
      _targetSpacing(targetSpacing),
      _relativeKernelRadius(relativeKernelRadius) {
    _particleSystemData = std::make_shared<CudaParticleSystemData3>();
    setIsUsingFixedSubTimeSteps(false);
}

CudaSphSolver3::~CudaSphSolver3() {}

float CudaSphSolver3::dragCoefficient() const { return _dragCoefficient; }

void CudaSphSolver3::setDragCoefficient(float newDragCoefficient) {
    _dragCoefficient = std::max(newDragCoefficient, 0.0f);
}

float CudaSphSolver3::restitutionCoefficient() const {
    return _restitutionCoefficient;
}

void CudaSphSolver3::setRestitutionCoefficient(
    float newRestitutionCoefficient) {
    _restitutionCoefficient = clamp(newRestitutionCoefficient, 0.0f, 1.0f);
}

const Vector3F& CudaSphSolver3::gravity() const { return _gravity; }

void CudaSphSolver3::setGravity(const Vector3F& newGravity) {
    _gravity = newGravity;
}

const CudaParticleSystemData3Ptr& CudaSphSolver3::particleSystemData() const {
    return _particleSystemData;
}

void CudaSphSolver3::onInitialize() {
    // When initializing the solver, update the collider and emitter state as
    // well since they also affects the initial condition of the simulation.
    Timer timer;
    updateCollider(0.0f);
    JET_INFO << "Update collider took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    updateEmitter(0.0f);
    JET_INFO << "Update emitter took " << timer.durationInSeconds()
             << " seconds";
}

void CudaSphSolver3::onAdvanceTimeStep(double timeStepInSeconds) {
    beginAdvanceTimeStep(timeStepInSeconds);

    // auto posCurr = _particleSystemData->positions();
    // auto velCurr = _particleSystemData->velocities();
    // auto dt = static_cast<float>(timeStepInSeconds);
    // auto g = make_float4(_gravity.x, _gravity.y, _gravity.z, 0.0f);

    // AdvanceTimeStepKernel kernel(_mass, dt, g);

    // thrust::for_each(
    //     make_zip_iterator(make_tuple(posCurr.begin(), velCurr.begin())),
    //     make_zip_iterator(make_tuple(posCurr.end(), velCurr.end())), kernel);

    endAdvanceTimeStep(timeStepInSeconds);
}

void CudaSphSolver3::onBeginAdvanceTimeStep(double timeStepInSeconds) {
    // Update collider and emitter
    Timer timer;
    updateCollider(timeStepInSeconds);
    JET_INFO << "Update collider took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    updateEmitter(timeStepInSeconds);
    JET_INFO << "Update emitter took " << timer.durationInSeconds()
             << " seconds";
}

void CudaSphSolver3::onEndAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void CudaSphSolver3::beginAdvanceTimeStep(double timeStepInSeconds) {
    onBeginAdvanceTimeStep(timeStepInSeconds);
}

void CudaSphSolver3::endAdvanceTimeStep(double timeStepInSeconds) {
    onEndAdvanceTimeStep(timeStepInSeconds);
}

void CudaSphSolver3::updateCollider(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void CudaSphSolver3::updateEmitter(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

CudaSphSolver3::Builder CudaSphSolver3::builder() { return Builder(); }

//

CudaSphSolver3::Builder& CudaSphSolver3::Builder::withTargetDensity(
    float targetDensity) {
    _targetDensity = targetDensity;
    return (*this);
}

CudaSphSolver3::Builder& CudaSphSolver3::Builder::withTargetSpacing(
    float targetSpacing) {
    _targetSpacing = targetSpacing;
    return (*this);
}

CudaSphSolver3::Builder& CudaSphSolver3::Builder::withRelativeKernelRadius(
    float relativeKernelRadius) {
    _relativeKernelRadius = relativeKernelRadius;
    return (*this);
}

CudaSphSolver3 CudaSphSolver3::Builder::build() const {
    return CudaSphSolver3(_targetDensity, _targetSpacing, _relativeKernelRadius);
}

CudaSphSolver3Ptr CudaSphSolver3::Builder::makeShared() const {
    return std::shared_ptr<CudaSphSolver3>(
        new CudaSphSolver3(_targetDensity, _targetSpacing, _relativeKernelRadius),
        [](CudaSphSolver3* obj) { delete obj; });
}
