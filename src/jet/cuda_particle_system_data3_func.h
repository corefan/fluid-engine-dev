// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_point_hash_grid_searcher3.h>
#include <jet/cuda_utils.h>
#include <jet/macros.h>

namespace jet {

namespace experimental {

template <typename NeighborCallback, typename NeighborCounterCallback>
class ForEachNeighborFunc {
 public:
    inline JET_CUDA_HOST ForEachNeighborFunc(
        const CudaPointHashGridSearcher3& searcher, float radius,
        const float4* origins, NeighborCallback insideCb,
        NeighborCounterCallback cntCb)
        : _neighborCallback(insideCb),
          _neighborNeighborCounterCallback(cntCb),
          _hashUtils(searcher.gridSpacing(), toInt3(searcher.resolution())),
          _radius(radius),
          _startIndexTable(searcher.startIndexTable().data()),
          _endIndexTable(searcher.endIndexTable().data()),
          _sortedIndices(searcher.sortedIndices().data()),
          _points(searcher.sortedPoints().data()),
          _origins(origins) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index i) {
        const float4 origin = _origins[i];

        size_t nearbyKeys[8];
        _hashUtils.getNearbyKeys(origin, nearbyKeys);

        const float queryRadiusSquared = _radius * _radius;

        size_t cnt = 0;
        for (int c = 0; c < 8; c++) {
            size_t nearbyKey = nearbyKeys[c];
            size_t start = _startIndexTable[nearbyKey];
            size_t end = _endIndexTable[nearbyKey];

            // Empty bucket -- continue to next bucket
            if (start == kMaxSize) {
                continue;
            }

            for (size_t jj = start; jj < end; ++jj) {
                float4 r = _points[jj] - origin;
                size_t j = _sortedIndices[jj];
                float distanceSquared = lengthSquared(r);
                if (i != j && distanceSquared <= queryRadiusSquared) {
                    _neighborCallback(i, j, cnt);
                    ++cnt;
                }
            }
        }

        _neighborNeighborCounterCallback(i, cnt);
    }

 private:
    NeighborCallback _neighborCallback;
    NeighborCounterCallback _neighborNeighborCounterCallback;
    CudaPointHashGridSearcher3::HashUtils _hashUtils;
    float _radius;
    const size_t* _startIndexTable;
    const size_t* _endIndexTable;
    const size_t* _sortedIndices;
    const float4* _points;
    const float4* _origins;
};

class NoOpFunc {
 public:
    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index, Index) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index, Index, Index) {}
};

class BuildNeighborListsFunc {
 public:
    inline JET_CUDA_HOST_DEVICE BuildNeighborListsFunc(
        const size_t* neighborStarts, const size_t* neighborEnds,
        size_t* neighborLists)
        : _neighborStarts(neighborStarts),
          _neighborEnds(neighborEnds),
          _neighborLists(neighborLists) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index i, Index j, Index cnt) {
        _neighborLists[_neighborStarts[i] + cnt] = j;
    }

 private:
    const size_t* _neighborStarts;
    const size_t* _neighborEnds;
    size_t* _neighborLists;
};

class CountNearbyPointsFunc {
 public:
    inline JET_CUDA_HOST_DEVICE CountNearbyPointsFunc(size_t* cnt)
        : _counts(cnt) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index idx, Index cnt) {
        _counts[idx] = cnt;
    }

 private:
    size_t* _counts;
};

}  // namespace experimental

}  // namespace jet
