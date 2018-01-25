#pragma once

#include "particle_sim_example.h"

#include "particleSystem.h"

class SdkSampleExample final : public ParticleSimExample {
 public:
    SdkSampleExample(const jet::Frame& frame);

 private:
    std::shared_ptr<ParticleSystem> _solver;
    jet::viz::PointsRenderable3Ptr _renderable;

    void onSetup(jet::viz::GlfwWindow* window) override;

    void onUpdate(const jet::Frame& frame) override;
};
