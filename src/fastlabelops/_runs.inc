#pragma once

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define FASTLABELOPS_X86_AVX2_DISPATCH 1
#else
#define FASTLABELOPS_X86_AVX2_DISPATCH 0
#endif

namespace runs_impl {

// Scan runs in input order and call emit(label, start, end). The AVX2 path
// compares each 8-lane block with the same block shifted by one element; the
// resulting bit mask marks every run boundary at once. Non-AVX2 builds and
// non-uint32 inputs use the scalar implementation below.
template <typename Label, typename Emit>
inline void for_each_run_scalar(const Label* data, std::size_t n, Emit&& emit) {
    for (std::size_t start = 0; start < n;) {
        const Label label = data[start];
        std::size_t end = start + 1;
        while (end < n && data[end] == label) ++end;
        emit(label, start, end);
        start = end;
    }
}

template <typename LabelA, typename LabelB, typename Emit>
inline void for_each_pair_run_scalar(
    const LabelA* a,
    const LabelB* b,
    std::size_t n,
    Emit&& emit
) {
    for (std::size_t start = 0; start < n;) {
        const LabelA av = a[start];
        const LabelB bv = b[start];
        std::size_t end = start + 1;
        while (end < n && a[end] == av && b[end] == bv) ++end;
        emit(av, bv, start, end);
        start = end;
    }
}

#if FASTLABELOPS_X86_AVX2_DISPATCH
inline bool avx2_available() noexcept {
    // Runtime dispatch keeps wheels portable instead of requiring -mavx2.
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
}

template <typename Emit>
__attribute__((target("avx2")))
void for_each_run_u32_avx2(const uint32_t* data, std::size_t n, Emit&& emit) {
    if (n == 0) return;

    std::size_t run_start = 0;
    std::size_t i = 1;
    for (; i + 8 <= n; i += 8) {
        const __m256i current =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        const __m256i previous =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i - 1));
        const __m256i equal = _mm256_cmpeq_epi32(current, previous);
        unsigned boundaries =
            ~static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(equal))) & 0xffu;

        while (boundaries != 0) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctz(boundaries));
            const std::size_t run_end = i + bit;
            emit(data[run_start], run_start, run_end);
            run_start = run_end;
            boundaries &= boundaries - 1;
        }
    }

    for (; i < n; ++i) {
        if (data[i] != data[i - 1]) {
            emit(data[run_start], run_start, i);
            run_start = i;
        }
    }
    emit(data[run_start], run_start, n);
}

template <typename Emit>
__attribute__((target("avx2")))
void for_each_pair_run_u32_avx2(
    const uint32_t* a,
    const uint32_t* b,
    std::size_t n,
    Emit&& emit
) {
    if (n == 0) return;

    std::size_t run_start = 0;
    std::size_t i = 1;
    for (; i + 8 <= n; i += 8) {
        const __m256i a_current =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i a_previous =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i - 1));
        const __m256i b_current =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i b_previous =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i - 1));
        const __m256i both_equal = _mm256_and_si256(
            _mm256_cmpeq_epi32(a_current, a_previous),
            _mm256_cmpeq_epi32(b_current, b_previous)
        );
        unsigned boundaries =
            ~static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(both_equal))) & 0xffu;

        while (boundaries != 0) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctz(boundaries));
            const std::size_t run_end = i + bit;
            emit(a[run_start], b[run_start], run_start, run_end);
            run_start = run_end;
            boundaries &= boundaries - 1;
        }
    }

    for (; i < n; ++i) {
        if (a[i] != a[i - 1] || b[i] != b[i - 1]) {
            emit(a[run_start], b[run_start], run_start, i);
            run_start = i;
        }
    }
    emit(a[run_start], b[run_start], run_start, n);
}

template <typename Emit>
__attribute__((target("avx2")))
void for_each_foreground_pair_run_u32_avx2(
    const uint32_t* a,
    const uint32_t* b,
    std::size_t n,
    Emit&& emit
) {
    if (n == 0) return;

    const __m256i zero = _mm256_setzero_si256();
    const __m256i all_bits = _mm256_set1_epi32(-1);
    std::size_t run_start = 0;
    bool run_active = a[0] != 0 && b[0] != 0;
    std::size_t i = 1;

    for (; i + 8 <= n; i += 8) {
        const __m256i a_current =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i a_previous =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i - 1));
        const __m256i b_current =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i b_previous =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i - 1));

        const __m256i current_active = _mm256_andnot_si256(
            _mm256_or_si256(
                _mm256_cmpeq_epi32(a_current, zero),
                _mm256_cmpeq_epi32(b_current, zero)
            ),
            all_bits
        );
        const __m256i previous_active = _mm256_andnot_si256(
            _mm256_or_si256(
                _mm256_cmpeq_epi32(a_previous, zero),
                _mm256_cmpeq_epi32(b_previous, zero)
            ),
            all_bits
        );
        const __m256i same_active_pair = _mm256_and_si256(
            _mm256_and_si256(current_active, previous_active),
            _mm256_and_si256(
                _mm256_cmpeq_epi32(a_current, a_previous),
                _mm256_cmpeq_epi32(b_current, b_previous)
            )
        );
        const __m256i both_inactive = _mm256_andnot_si256(
            _mm256_or_si256(current_active, previous_active),
            all_bits
        );
        const __m256i same_state = _mm256_or_si256(same_active_pair, both_inactive);
        unsigned boundaries =
            ~static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(same_state))) & 0xffu;

        while (boundaries != 0) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctz(boundaries));
            const std::size_t run_end = i + bit;
            if (run_active) emit(a[run_start], b[run_start], run_start, run_end);
            run_start = run_end;
            run_active = a[run_start] != 0 && b[run_start] != 0;
            boundaries &= boundaries - 1;
        }
    }

    for (; i < n; ++i) {
        const bool active = a[i] != 0 && b[i] != 0;
        const bool previous_active = a[i - 1] != 0 && b[i - 1] != 0;
        const bool same = (!active && !previous_active) ||
            (active && previous_active && a[i] == a[i - 1] && b[i] == b[i - 1]);
        if (!same) {
            if (run_active) emit(a[run_start], b[run_start], run_start, i);
            run_start = i;
            run_active = active;
        }
    }
    if (run_active) emit(a[run_start], b[run_start], run_start, n);
}
#endif

template <typename Label, typename Emit>
inline void for_each_run(const Label* data, std::size_t n, Emit&& emit) {
#if FASTLABELOPS_X86_AVX2_DISPATCH
    if constexpr (std::is_same_v<Label, uint32_t>) {
        if (avx2_available()) {
            for_each_run_u32_avx2(data, n, std::forward<Emit>(emit));
            return;
        }
    }
#endif
    for_each_run_scalar(data, n, std::forward<Emit>(emit));
}

constexpr std::size_t kDensitySamples = 64;

template <typename Changed>
inline bool sampled_transitions_are_dense(std::size_t n, Changed&& changed) noexcept {
    if (n < 2) return false;
    const std::size_t samples = std::min(n - 1, kDensitySamples);
    const std::size_t step = std::max<std::size_t>((n - 1) / samples, 1);
    std::size_t boundaries = 0;
    for (std::size_t sample = 0, i = 1; sample < samples; ++sample, i += step) {
        boundaries += changed(i);
    }
    return boundaries * 4 >= samples * 3;
}

template <typename Label>
inline bool runs_are_dense(const Label* data, std::size_t n) noexcept {
    return sampled_transitions_are_dense(n, [&](std::size_t i) {
        return data[i] != data[i - 1];
    });
}

inline bool pair_runs_are_dense_u32(
    const uint32_t* a,
    const uint32_t* b,
    std::size_t n
) noexcept {
    return sampled_transitions_are_dense(n, [&](std::size_t i) {
        return a[i] != a[i - 1] || b[i] != b[i - 1];
    });
}

template <typename LabelA, typename LabelB, typename Emit>
inline void for_each_pair_run(
    const LabelA* a,
    const LabelB* b,
    std::size_t n,
    Emit&& emit
) {
#if FASTLABELOPS_X86_AVX2_DISPATCH
    if constexpr (std::is_same_v<LabelA, uint32_t> && std::is_same_v<LabelB, uint32_t>) {
        if (avx2_available() && !pair_runs_are_dense_u32(a, b, n)) {
            for_each_pair_run_u32_avx2(a, b, n, std::forward<Emit>(emit));
            return;
        }
    }
#endif
    for_each_pair_run_scalar(a, b, n, std::forward<Emit>(emit));
}

inline bool joint_foreground_is_sparse_u32(
    const uint32_t* a,
    const uint32_t* b,
    std::size_t n
) noexcept {
    if (n == 0) return false;
    const std::size_t samples = std::min(n, kDensitySamples);
    const std::size_t step = std::max<std::size_t>(n / samples, 1);
    std::size_t active = 0;
    for (std::size_t sample = 0, i = 0; sample < samples; ++sample, i += step) {
        active += a[i] != 0 && b[i] != 0;
    }
    return active * 2 < samples;
}

template <typename LabelA, typename LabelB, typename Emit>
inline void for_each_foreground_pair_run(
    const LabelA* a,
    const LabelB* b,
    std::size_t n,
    Emit&& emit
) {
#if FASTLABELOPS_X86_AVX2_DISPATCH
    if constexpr (std::is_same_v<LabelA, uint32_t> && std::is_same_v<LabelB, uint32_t>) {
        if (avx2_available() && joint_foreground_is_sparse_u32(a, b, n)) {
            for_each_foreground_pair_run_u32_avx2(a, b, n, std::forward<Emit>(emit));
            return;
        }
    }
#endif
    for_each_pair_run(
        a,
        b,
        n,
        [&](LabelA av, LabelB bv, std::size_t start, std::size_t end) {
            if (av != 0 && bv != 0) emit(av, bv, start, end);
        }
    );
}

}  // namespace runs_impl

#undef FASTLABELOPS_X86_AVX2_DISPATCH
