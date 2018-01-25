#pragma once

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

class ParticleSystem {
 public:
    ParticleSystem(uint numParticles, uint3 gridSize);
    ~ParticleSystem();

    enum ParticleArray {
        POSITION,
        VELOCITY,
    };

    void spawnParticles();
    void update(float deltaTime);

    void setArray(ParticleArray array, const float *data, int start, int count);

    size_t numParticles() const { return _numParticles; }

    void *positions() const { return (void *)_positions; }
    void *colors() const { return (void *)_colors; }

    void setIterations(int i) { _solverIterations = i; }

    void setDamping(float x) { _params.globalDamping = x; }
    void setGravity(float x) { _params.gravity = make_float3(0.0f, x, 0.0f); }

    void setCollideSpring(float x) { _params.spring = x; }
    void setCollideDamping(float x) { _params.damping = x; }
    void setCollideShear(float x) { _params.shear = x; }
    void setCollideAttraction(float x) { _params.attraction = x; }

    void setColliderPos(float3 x) { _params.colliderPos = x; }

    float getParticleRadius() { return _params.particleRadius; }
    float3 getColliderPos() { return _params.colliderPos; }
    float getColliderRadius() { return _params.colliderRadius; }
    uint3 getGridSize() { return _params.gridSize; }
    float3 getWorldOrigin() { return _params.worldOrigin; }
    float3 getCellSize() { return _params.cellSize; }

    void addSphere(int index, float *pos, float *vel, int r, float spacing);

 private:
    uint _numParticles;

    // CPU data
    float *_hPos;  // particle positions
    float *_hVel;  // particle velocities

    // GPU data
    float *_positions;  // these are the CUDA deviceMem Pos
    float *_velocities;
    float *_colors;  // these are the CUDA deviceMem Color

    float *_dSortedPos;
    float *_dSortedVel;

    // grid data for sorting method
    uint *_dGridParticleHash;   // grid hash value for each particle
    uint *_dGridParticleIndex;  // particle index for each particle
    uint *_dCellStart;          // index of start of each cell in sorted list
    uint *_dCellEnd;            // index of end of cell

    uint _gridSortBits;

    uint _posVbo;    // vertex buffer object for particle positions
    uint _colorVBO;  // vertex buffer object for colors

    struct cudaGraphicsResource
        *_cuda_posvbo_resource;  // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource
        *_cuda_colorvbo_resource;  // handles OpenGL-CUDA exchange

    // params
    SimParams _params;
    uint3 _gridSize;
    uint _numGridCells;

    uint _solverIterations;

    void init(int numParticles);
    void clear();

    void initGrid(uint *size, float spacing, float jitter, uint numParticles);
};
