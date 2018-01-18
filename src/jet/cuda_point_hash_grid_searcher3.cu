// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_point_hash_grid_searcher3.h>

#include <thrust/for_each.h>

using namespace jet;
using namespace experimental;

namespace {

struct InitializeTables {
    template <typename Tuple>
    inline JET_CUDA_DEVICE void operator()(Tuple t) {
        thrust::get<0>(t) = kMaxSize;
        thrust::get<1>(t) = kMaxSize;
    }
};

struct InitializeIndexPointAndKeys {
    CudaPointHashGridSearcher3::HashUtils hashUtils;

    inline JET_CUDA_HOST_DEVICE InitializeIndexPointAndKeys(float gridSpacing,
                                                            int3 resolution)
        : hashUtils(gridSpacing, resolution) {}

    template <typename Tuple>
    inline JET_CUDA_DEVICE void operator()(Tuple t) {
        // 0: i [in]
        // 1: sortedIndices[out]
        // 2: points[in]
        // 3: points[out]
        // 4: keys[out]
        size_t i = thrust::get<0>(t);
        thrust::get<1>(t) = i;
        float4 p = thrust::get<2>(t);
        thrust::get<3>(t) = p;
        size_t key = hashUtils.getHashKeyFromPosition(p);
        thrust::get<4>(t) = key;
    }
};

struct BuildTables {
    size_t* keys;
    size_t* startIndexTable;
    size_t* endIndexTable;

    inline JET_CUDA_HOST_DEVICE BuildTables(size_t* k, size_t* sit, size_t* eit)
        : keys(k), startIndexTable(sit), endIndexTable(eit) {}

    template <typename Index>
    inline JET_CUDA_DEVICE void operator()(Index i) {
        size_t k = keys[i];
        size_t kLeft = keys[i - 1];
        if (k > kLeft) {
            startIndexTable[k] = i;
            endIndexTable[kLeft] = i;
        }
    }
};

}  // namespace

CudaPointHashGridSearcher3::CudaPointHashGridSearcher3(const Size3& resolution,
                                                       float gridSpacing)
    : CudaPointHashGridSearcher3(resolution.x, resolution.y, resolution.z,
                                 gridSpacing) {}

CudaPointHashGridSearcher3::CudaPointHashGridSearcher3(size_t resolutionX,
                                                       size_t resolutionY,
                                                       size_t resolutionZ,
                                                       float gridSpacing)
    : _gridSpacing(gridSpacing) {
    _resolution.x = std::max(static_cast<int>(resolutionX), 1);
    _resolution.y = std::max(static_cast<int>(resolutionY), 1);
    _resolution.z = std::max(static_cast<int>(resolutionZ), 1);

    _startIndexTable.resize(_resolution.x * _resolution.y * _resolution.z,
                            kMaxSize);
    _endIndexTable.resize(_resolution.x * _resolution.y * _resolution.z,
                          kMaxSize);
}

CudaPointHashGridSearcher3::CudaPointHashGridSearcher3(
    const CudaPointHashGridSearcher3& other) {
    set(other);
}

void CudaPointHashGridSearcher3::build(const CudaArrayView1<float4>& points) {
    _points.clear();
    _keys.clear();
    _startIndexTable.clear();
    _endIndexTable.clear();
    _sortedIndices.clear();

    // Allocate memory chuncks
    size_t numberOfPoints = points.size();
    _startIndexTable.resize(_resolution.x * _resolution.y * _resolution.z);
    _endIndexTable.resize(_resolution.x * _resolution.y * _resolution.z);
    _keys.resize(numberOfPoints);
    _sortedIndices.resize(numberOfPoints);
    _points.resize(numberOfPoints);

    if (numberOfPoints == 0) {
        return;
    }

    // Initialize tables
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                         _startIndexTable.begin(), _endIndexTable.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         _startIndexTable.end(), _endIndexTable.end())),
                     InitializeTables());

    // Initialize indices array and generate hash key for each point
    auto countingBegin = thrust::counting_iterator<size_t>(0);
    auto countingEnd = countingBegin + numberOfPoints;
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                         countingBegin, _sortedIndices.begin(), points.begin(),
                         _points.begin(), _keys.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         countingEnd, _sortedIndices.end(), points.end(),
                         _points.end(), _keys.end())),
                     InitializeIndexPointAndKeys(_gridSpacing, _resolution));

    // Sort indices/points/key based on hash key
    thrust::sort_by_key(_keys.begin(), _keys.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            _sortedIndices.begin(), _points.begin())));

    // Now _points and _keys are sorted by points' hash key values.
    // Let's fill in start/end index table with _keys.

    // Assume that _keys array looks like:
    // [5|8|8|10|10|10]
    // Then _startIndexTable and _endIndexTable should be like:
    // [.....|0|...|1|..|3|..]
    // [.....|1|...|3|..|6|..]
    //       ^5    ^8   ^10
    // So that _endIndexTable[i] - _startIndexTable[i] is the number points
    // in i-th table bucket.

    _startIndexTable[_keys[0]] = 0;
    _endIndexTable[_keys[numberOfPoints - 1]] = numberOfPoints;

    thrust::for_each(countingBegin + 1, countingEnd,
                     BuildTables(_keys.data(), _startIndexTable.data(),
                                 _endIndexTable.data()));
}

float CudaPointHashGridSearcher3::gridSpacing() const { return _gridSpacing; }

Size3 CudaPointHashGridSearcher3::resolution() const {
    return Size3{static_cast<size_t>(_resolution.x),
                 static_cast<size_t>(_resolution.y),
                 static_cast<size_t>(_resolution.z)};
}

const CudaArrayView1<float4> CudaPointHashGridSearcher3::sortedPoints() const {
    return _points.view();
}

const CudaArrayView1<size_t> CudaPointHashGridSearcher3::keys() const {
    return _keys.view();
}

const CudaArrayView1<size_t> CudaPointHashGridSearcher3::startIndexTable()
    const {
    return _startIndexTable.view();
}

const CudaArrayView1<size_t> CudaPointHashGridSearcher3::endIndexTable() const {
    return _endIndexTable.view();
}

const CudaArrayView1<size_t> CudaPointHashGridSearcher3::sortedIndices() const {
    return _sortedIndices.view();
}

CudaPointHashGridSearcher3& CudaPointHashGridSearcher3::operator=(
    const CudaPointHashGridSearcher3& other) {
    set(other);
    return (*this);
}

void CudaPointHashGridSearcher3::set(const CudaPointHashGridSearcher3& other) {
    _gridSpacing = other._gridSpacing;
    _resolution = other._resolution;
    _points = other._points;
    _keys = other._keys;
    _startIndexTable = other._startIndexTable;
    _endIndexTable = other._endIndexTable;
    _sortedIndices = other._sortedIndices;
}
