// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "sdk_sample_example.h"

#include <jet.viz/points_renderable3.h>

#include <thrust/device_vector.h>

using namespace jet;
using namespace viz;

namespace {

struct PosToVertex {
    template <typename Pos>
    __device__ VertexPosition3Color4 operator()(const Pos& pos) {
        VertexPosition3Color4 vertex;
        vertex.x = pos.x;
        vertex.y = pos.y;
        vertex.z = pos.z;
        vertex.r = 1.0f;
        vertex.g = 1.0f;
        vertex.b = 1.0f;
        vertex.a = 1.0f;
        return vertex;
    }
};

uint numParticles = 16383;

}  // namespace

SdkSampleExample::SdkSampleExample(const jet::Frame& frame)
    : ParticleSimExample(frame) {}

void SdkSampleExample::onSetup(GlfwWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>(Vector3D{0.5, 0.5, 3.0},
                                      Vector3D{0.0, 0.0, -1.0},
                                      Vector3D{0.0, 1.0, 0.0}, 0.01, 10.0),
        Vector3D{0.5, 0.5, 0.5}));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{0, 0, 0, 1});

    // Setup solver
    _solver = std::make_shared<ParticleSystem>(numParticles,
                                               make_uint3(64, 64, 64));
    _solver->spawnParticles();

    // Setup renderable
    thrust::device_ptr<float4> deviceData((float4*)_solver->positions());
    thrust::device_vector<VertexPosition3Color4> vertices(16383);
    thrust::transform(deviceData, deviceData + numParticles, vertices.begin(),
                      PosToVertex());

    _renderable = std::make_shared<PointsRenderable3>(renderer);
    _renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                           window->windowSize().x);
    _renderable->setPositionsAndColors(nullptr, vertices.size());
    _renderable->vertexBuffer()->updateWithCuda(
        (const float*)thrust::raw_pointer_cast(vertices.data()));
    renderer->addRenderable(_renderable);
}

void SdkSampleExample::onUpdate(const Frame& frame) {
    _solver->update(0.5f);

    thrust::device_ptr<float4> deviceData((float4*)_solver->positions());
    thrust::device_ptr<VertexPosition3Color4> vertices(
        (VertexPosition3Color4*)_renderable->vertexBuffer()
            ->cudaMapResources());
    thrust::transform(deviceData, deviceData + numParticles, vertices,
                      PosToVertex());
    _renderable->vertexBuffer()->cudaUnmapResources();
}
