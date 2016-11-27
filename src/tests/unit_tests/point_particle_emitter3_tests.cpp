// Copyright (c) 2016 Doyub Kim

#include <jet/math_utils.h>
#include <jet/point_particle_emitter3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PointParticleEmitter3, Constructors) {
    PointParticleEmitter3 emitter(
        {1.0, 2.0, 3.0},
        Vector3D(0.5, 1.0, -2.0).normalized(),
        3.0,
        15.0,
        4,
        18);

    EXPECT_EQ(4, emitter.maxNumberOfNewParticlesPerSecond());
    EXPECT_EQ(18, emitter.maxNumberOfParticles());
}

TEST(PointParticleEmitter3, Emit) {
    Vector3D dir = Vector3D(0.5, 1.0, -2.0).normalized();

    PointParticleEmitter3 emitter(
        {1.0, 2.0, 3.0},
        dir,
        3.0,
        15.0,
        4,
        18);

    auto particles = std::make_shared<ParticleSystemData3>();

    Frame frame(1, 1.0);
    emitter.emit(frame, particles);
    EXPECT_EQ(4, particles->numberOfParticles());

    frame.advance();
    emitter.emit(frame, particles);
    EXPECT_EQ(8, particles->numberOfParticles());

    frame.advance();
    emitter.emit(frame, particles);
    EXPECT_EQ(12, particles->numberOfParticles());

    frame.advance();
    emitter.emit(frame, particles);
    EXPECT_EQ(16, particles->numberOfParticles());

    frame.advance();
    emitter.emit(frame, particles);
    EXPECT_EQ(18, particles->numberOfParticles());

    auto pos = particles->positions();
    auto vel = particles->velocities();

    for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
        EXPECT_DOUBLE_EQ(1.0, pos[i].x);
        EXPECT_DOUBLE_EQ(2.0, pos[i].y);
        EXPECT_DOUBLE_EQ(3.0, pos[i].z);

        EXPECT_LE(
            std::cos(degreesToRadians(15.0)),
            vel[i].normalized().dot(dir));
        EXPECT_DOUBLE_EQ(3.0, vel[i].length());
    }
}
