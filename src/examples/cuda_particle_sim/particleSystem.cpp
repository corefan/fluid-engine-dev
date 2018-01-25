#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize)
    : _numParticles(numParticles),
      _hPos(0),
      _hVel(0),
      _velocities(0),
      _gridSize(gridSize),
      _solverIterations(1) {
    _numGridCells = _gridSize.x * _gridSize.y * _gridSize.z;

    _gridSortBits = 18;  // increase this for larger grids

    // set simulation parameters
    _params.gridSize = _gridSize;
    _params.numCells = _numGridCells;
    _params.numBodies = _numParticles;

    _params.particleRadius = 1.0f / 64.0f;
    _params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    _params.colliderRadius = 0.2f;

    _params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    //    _params.cellSize = make_float3(worldSize.x / _gridSize.x,
    //    worldSize.y / _gridSize.y, worldSize.z / _gridSize.z);
    float cellSize =
        _params.particleRadius * 2.0f;  // cell size equal to particle diameter
    _params.cellSize = make_float3(cellSize, cellSize, cellSize);

    _params.spring = 0.5f;
    _params.damping = 0.02f;
    _params.shear = 0.1f;
    _params.attraction = 0.0f;
    _params.boundaryDamping = -0.5f;

    _params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    _params.globalDamping = 1.0f;

    init(numParticles);
}

ParticleSystem::~ParticleSystem() {
    clear();
    _numParticles = 0;
}

inline float lerp(float a, float b, float t) { return a + t * (b - a); }

// create a color ramp
void colorRamp(float t, float *r) {
    const int ncolors = 7;
    float c[ncolors][3] = {
        {
            1.0,
            0.0,
            0.0,
        },
        {
            1.0,
            0.5,
            0.0,
        },
        {
            1.0,
            1.0,
            0.0,
        },
        {
            0.0,
            1.0,
            0.0,
        },
        {
            0.0,
            1.0,
            1.0,
        },
        {
            0.0,
            0.0,
            1.0,
        },
        {
            1.0,
            0.0,
            1.0,
        },
    };
    t = t * (ncolors - 1);
    int i = (int)t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i + 1][0], u);
    r[1] = lerp(c[i][1], c[i + 1][1], u);
    r[2] = lerp(c[i][2], c[i + 1][2], u);
}

void ParticleSystem::init(int numParticles) {
    _numParticles = numParticles;

    // allocate host storage
    _hPos = new float[_numParticles * 4];
    _hVel = new float[_numParticles * 4];
    memset(_hPos, 0, _numParticles * 4 * sizeof(float));
    memset(_hVel, 0, _numParticles * 4 * sizeof(float));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * _numParticles;

    checkCudaErrors(cudaMalloc((void **)&_positions, memSize));

    allocateArray((void **)&_velocities, memSize);

    allocateArray((void **)&_dSortedPos, memSize);
    allocateArray((void **)&_dSortedVel, memSize);

    allocateArray((void **)&_dGridParticleHash, _numParticles * sizeof(uint));
    allocateArray((void **)&_dGridParticleIndex, _numParticles * sizeof(uint));

    allocateArray((void **)&_dCellStart, _numGridCells * sizeof(uint));
    allocateArray((void **)&_dCellEnd, _numGridCells * sizeof(uint));

    checkCudaErrors(
        cudaMalloc((void **)&_colors, sizeof(float) * numParticles * 4));

    setParameters(&_params);
}

void ParticleSystem::clear() {
    delete[] _hPos;
    delete[] _hVel;

    checkCudaErrors(cudaFree(_positions));
    checkCudaErrors(cudaFree(_colors));
    freeArray(_velocities);
    freeArray(_dSortedPos);
    freeArray(_dSortedVel);

    freeArray(_dGridParticleHash);
    freeArray(_dGridParticleIndex);
    freeArray(_dCellStart);
    freeArray(_dCellEnd);
}

// step the simulation
void ParticleSystem::update(float deltaTime) {
    // update constants
    setParameters(&_params);

    // integrate
    integrateSystem(_positions, _velocities, deltaTime, _numParticles);

    // calculate grid hash
    calcHash(_dGridParticleHash, _dGridParticleIndex, _positions, _numParticles);

    // sort particles based on hash
    sortParticles(_dGridParticleHash, _dGridParticleIndex, _numParticles);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        _dCellStart, _dCellEnd, _dSortedPos, _dSortedVel, _dGridParticleHash,
        _dGridParticleIndex, _positions, _velocities, _numParticles, _numGridCells);

    // process collisions
    collide(_velocities, _dSortedPos, _dSortedVel, _dGridParticleIndex,
            _dCellStart, _dCellEnd, _numParticles, _numGridCells);
}

void ParticleSystem::setArray(ParticleArray array, const float *data, int start,
                              int count) {
    switch (array) {
        default:
        case POSITION: {
            copyArrayToDevice(_positions, data, start * 4 * sizeof(float),
                              count * 4 * sizeof(float));
        } break;

        case VELOCITY:
            copyArrayToDevice(_velocities, data, start * 4 * sizeof(float),
                              count * 4 * sizeof(float));
            break;
    }
}

inline float frand() { return rand() / (float)RAND_MAX; }

void ParticleSystem::initGrid(uint *size, float spacing, float jitter,
                              uint numParticles) {
    srand(1973);

    for (uint z = 0; z < size[2]; z++) {
        for (uint y = 0; y < size[1]; y++) {
            for (uint x = 0; x < size[0]; x++) {
                uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

                if (i < numParticles) {
                    _hPos[i * 4] = (spacing * x) + _params.particleRadius -
                                   1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    _hPos[i * 4 + 1] = (spacing * y) + _params.particleRadius -
                                       1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    _hPos[i * 4 + 2] = (spacing * z) + _params.particleRadius -
                                       1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    _hPos[i * 4 + 3] = 1.0f;

                    _hVel[i * 4] = 0.0f;
                    _hVel[i * 4 + 1] = 0.0f;
                    _hVel[i * 4 + 2] = 0.0f;
                    _hVel[i * 4 + 3] = 0.0f;
                }
            }
        }
    }
}

void ParticleSystem::spawnParticles() {
    float jitter = _params.particleRadius * 0.01f;
    uint s = (int)ceilf(powf((float)_numParticles, 1.0f / 3.0f));
    uint gridSize[3];
    gridSize[0] = gridSize[1] = gridSize[2] = s;
    initGrid(gridSize, _params.particleRadius * 2.0f, jitter, _numParticles);

    setArray(POSITION, _hPos, 0, _numParticles);
    setArray(VELOCITY, _hVel, 0, _numParticles);
}

void ParticleSystem::addSphere(int start, float *pos, float *vel, int r,
                               float spacing) {
    uint index = start;

    for (int z = -r; z <= r; z++) {
        for (int y = -r; y <= r; y++) {
            for (int x = -r; x <= r; x++) {
                float dx = x * spacing;
                float dy = y * spacing;
                float dz = z * spacing;
                float l = sqrtf(dx * dx + dy * dy + dz * dz);
                float jitter = _params.particleRadius * 0.01f;

                if ((l <= _params.particleRadius * 2.0f * r) &&
                    (index < _numParticles)) {
                    _hPos[index * 4] =
                        pos[0] + dx + (frand() * 2.0f - 1.0f) * jitter;
                    _hPos[index * 4 + 1] =
                        pos[1] + dy + (frand() * 2.0f - 1.0f) * jitter;
                    _hPos[index * 4 + 2] =
                        pos[2] + dz + (frand() * 2.0f - 1.0f) * jitter;
                    _hPos[index * 4 + 3] = pos[3];

                    _hVel[index * 4] = vel[0];
                    _hVel[index * 4 + 1] = vel[1];
                    _hVel[index * 4 + 2] = vel[2];
                    _hVel[index * 4 + 3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, _hPos, start, index);
    setArray(VELOCITY, _hVel, start, index);
}
