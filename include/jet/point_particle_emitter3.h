// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_
#define INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_

#include <jet/particle_emitter3.h>
#include <limits>
#include <random>

namespace jet {

//!
//! \brief 3-D point particle emitter.
//!
//! This class emits particles from a single point in given direction, speed,
//! and spreading angle.
//!
class PointParticleEmitter3 final : public ParticleEmitter3 {
 public:
    //!
    //! Constructs an emitter that spawns particles from given origin,
    //! direction, speed, spread angle, max number of new particles per second,
    //! max total number of particles to be emitted, and random seed.
    //!
    //! \param[in]  origin                      The origin.
    //! \param[in]  direction                   The direction.
    //! \param[in]  speed                       The speed.
    //! \param[in]  spreadAngleInDegrees        The spread angle in degrees.
    //! \param[in]  maxNumOfNewParticlesPerSec  The max number of new particles
    //!                                         per second.
    //! \param[in]  maxNumOfParticles           The max number of particles to
    //!                                         be emitted.
    //! \param[in]  seed                        The random seed.
    //!
    PointParticleEmitter3(
        const Vector3D& origin,
        const Vector3D& direction,
        double speed,
        double spreadAngleInDegrees,
        size_t maxNumOfNewParticlesPerSec = 1,
        size_t maxNumOfParticles = std::numeric_limits<size_t>::max(),
        uint32_t seed = 0);

    //!
    //! \brief      Emits particles to the particle system data.
    //!
    //! \param[in]  frame     Current animation frame.
    //! \param[in]  particles The particle system data.
    //!
    void emit(
        const Frame& frame,
        const ParticleSystemData3Ptr& particles) override;

    //! Returns max number of new particles per second.
    size_t maxNumberOfNewParticlesPerSecond() const;

    //! Sets max number of new particles per second.
    void setMaxNumberOfNewParticlesPerSecond(size_t rate);

    //! Returns max number of particles to be emitted.
    size_t maxNumberOfParticles() const;

    //! Sets max number of particles to be emitted.
    void setMaxNumberOfParticles(size_t maxNumberOfParticles);

 private:
    std::mt19937 _rng;

    double _firstFrameTimeInSeconds = 0.0;
    size_t _numberOfEmittedParticles = 0;

    size_t _maxNumberOfNewParticlesPerSecond = 1;
    size_t _maxNumberOfParticles = std::numeric_limits<size_t>::max();

    Vector3D _origin;
    Vector3D _direction;
    double _speed;
    double _spreadAngleInRadians;

    void emit(
        Array1<Vector3D>* newPositions,
        Array1<Vector3D>* newVelocities,
        size_t maxNewNumberOfParticles);

    double random();
};

typedef std::shared_ptr<PointParticleEmitter3> PointParticleEmitter3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_