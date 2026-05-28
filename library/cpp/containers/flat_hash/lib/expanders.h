#pragma once

#include <stddef.h>
#include <utility>

namespace NFlatHash {

struct TSimpleExpander {
    static constexpr bool NeedGrow(std::size_t size, std::size_t buckets) noexcept {
        return size >= buckets / 2;
    }

    static constexpr bool WillNeedGrow(std::size_t size, std::size_t buckets) noexcept {
        return NeedGrow(size + 1, buckets);
    }

    static constexpr std::size_t EvalNewSize(std::size_t buckets) noexcept {
        return buckets * 2;
    }

    static constexpr std::size_t SuitableSize(std::size_t size) noexcept {
        return size * 2 + 1;
    }
};

}  // namespace NFlatHash

