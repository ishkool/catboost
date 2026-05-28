#pragma once

#include <stddef.h>
#include <type_traits>

namespace NFlatHash {

class TLinearProbing {
public:
    template <class SizeFitter, class F>
    static auto FindBucket(SizeFitter sf, std::size_t idx, std::size_t sz, F f) {
        idx = sf.EvalIndex(idx, sz);
        while (!f(idx)) {
            idx = sf.EvalIndex(++idx, sz);
        }
        return idx;
    }
};

class TQuadraticProbing {
public:
    template <class SizeFitter, class F>
    static auto FindBucket(SizeFitter sf, std::size_t idx, std::size_t sz, F f) {
        idx = sf.EvalIndex(idx, sz);
        std::size_t k = 0;
        while (!f(idx)) {
            idx = sf.EvalIndex(idx + 2 * ++k - 1, sz);
        }
        return idx;
    }
};

class TDenseProbing {
public:
    template <class SizeFitter, class F>
    static auto FindBucket(SizeFitter sf, std::size_t idx, std::size_t sz, F f) {
        idx = sf.EvalIndex(idx, sz);
        std::size_t k = 0;
        while (!f(idx)) {
            idx = sf.EvalIndex(idx + ++k, sz);
        }
        return idx;
    }
};

}  // NFlatHash

