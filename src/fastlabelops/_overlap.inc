namespace overlap_impl {

constexpr uint64_t kSecondLabelHashSalt = 0xd6e8feb86659fd93ULL;
constexpr unsigned kPairHashRotation = 29;

// SplitMix64 mixing function.
inline uint64_t mix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline uint64_t rotl64(uint64_t x, unsigned r) noexcept {
    return (x << r) | (x >> (64u - r));
}

template <typename LabelA, typename LabelB>
struct Entry {
    LabelA a = LabelA{0};
    LabelB b = LabelB{0};
    std::size_t output_index_plus_one = 0;
};

template <typename LabelA, typename LabelB>
class PairMap {
public:
    explicit PairMap(std::size_t input_size) {
        const std::size_t expected =
            std::min(input_size, label_map_impl::kInitialReserveLimit);
        const std::size_t cap =
            label_map_impl::bucket_count_for_expected_entries(expected);
        entries_.resize(cap);
        configure_capacity(cap);
    }

    bool get_or_insert(LabelA a, LabelB b, std::size_t candidate, std::size_t& index) {
        std::size_t i = index_for(a, b);
        for (;;) {
            Entry<LabelA, LabelB>& entry = entries_[i];
            if (entry.output_index_plus_one == 0) {
                if (size_ >= grow_at_) {
                    grow();
                    return get_or_insert(a, b, candidate, index);
                }
                entry.a = a;
                entry.b = b;
                entry.output_index_plus_one = candidate + 1;
                index = candidate;
                ++size_;
                return true;
            }
            if (entry.a == a && entry.b == b) {
                index = entry.output_index_plus_one - 1;
                return false;
            }
            i = (i + 1) & mask_;
        }
    }

private:
    std::size_t index_for(LabelA a, LabelB b) const noexcept {
        if constexpr (sizeof(LabelA) == sizeof(uint32_t) && sizeof(LabelB) == sizeof(uint32_t)) {
            const uint64_t key =
                (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
            return static_cast<std::size_t>(mix64(key) & mask_);
        }
        const uint64_t ha = mix64(static_cast<uint64_t>(a));
        const uint64_t hb = mix64(static_cast<uint64_t>(b) ^ kSecondLabelHashSalt);
        return static_cast<std::size_t>((ha ^ rotl64(hb, kPairHashRotation)) & mask_);
    }

    void insert_no_grow(const Entry<LabelA, LabelB>& old) noexcept {
        std::size_t i = index_for(old.a, old.b);
        while (entries_[i].output_index_plus_one != 0) i = (i + 1) & mask_;
        entries_[i] = old;
        ++size_;
    }

    void configure_capacity(std::size_t cap) noexcept {
        mask_ = cap - 1;
        grow_at_ = label_map_impl::growth_threshold(cap);
    }

    void grow() {
        if (entries_.size() > entries_.max_size() / 2) throw std::bad_alloc();
        std::vector<Entry<LabelA, LabelB>> old = std::move(entries_);
        const std::size_t new_cap = old.size() * 2;
        entries_.clear();
        entries_.resize(new_cap);
        configure_capacity(new_cap);
        size_ = 0;
        for (const auto& entry : old) {
            if (entry.output_index_plus_one != 0) insert_no_grow(entry);
        }
    }

    std::vector<Entry<LabelA, LabelB>> entries_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
    std::size_t grow_at_ = 0;
};

// Most instance labels overlap only a few labels in the other mask. Keep the
// first two B-label mappings directly in the A-label row; uncommon higher-
// degree rows spill into the existing global PairMap.
template <typename LabelB>
struct TinyRow {
    LabelB labels[2] = {LabelB{0}, LabelB{0}};
    std::size_t indices[2] = {0, 0};
    std::size_t size = 0;
};

template <typename LabelA, typename LabelB>
void count_pairs(
    const LabelA* a,
    const LabelB* b,
    std::size_t n,
    bool include_background,
    std::vector<LabelA>& out_a,
    std::vector<LabelB>& out_b,
    std::vector<uint64_t>& out_counts
) {
    const std::size_t reserve_n = std::min(n, label_map_impl::kInitialReserveLimit);
    label_map_impl::LabelMap<LabelA, std::size_t> row_index(reserve_n);
    std::vector<TinyRow<LabelB>> rows;
    rows.reserve(reserve_n);
    TinyRow<LabelB> background_row;
    PairMap<LabelA, LabelB> overflow(64);

    out_a.reserve(reserve_n);
    out_b.reserve(reserve_n);
    out_counts.reserve(reserve_n);

    for (std::size_t i = 0; i < n;) {
        const LabelA av = a[i];
        const LabelB bv = b[i];
        std::size_t run_end = i + 1;
        while (run_end < n && a[run_end] == av && b[run_end] == bv) ++run_end;
        const uint64_t run_length = static_cast<uint64_t>(run_end - i);

        if (include_background || (av != LabelA{0} && bv != LabelB{0})) {
            TinyRow<LabelB>* row = nullptr;
            if (av == LabelA{0}) {
                row = &background_row;
            } else {
                auto [row_ptr, inserted] = row_index.get_or_insert(av, rows.size());
                if (inserted) rows.emplace_back();
                row = &rows[*row_ptr];
            }

            bool found = false;
            for (std::size_t j = 0; j < row->size; ++j) {
                if (row->labels[j] == bv) {
                    out_counts[row->indices[j]] += run_length;
                    found = true;
                    break;
                }
            }

            if (!found) {
                if (row->size < 2) {
                    const std::size_t index = out_counts.size();
                    row->labels[row->size] = bv;
                    row->indices[row->size] = index;
                    ++row->size;
                    out_a.push_back(av);
                    out_b.push_back(bv);
                    out_counts.push_back(run_length);
                } else {
                    const std::size_t candidate = out_counts.size();
                    std::size_t index = 0;
                    if (overflow.get_or_insert(av, bv, candidate, index)) {
                        out_a.push_back(av);
                        out_b.push_back(bv);
                        out_counts.push_back(run_length);
                    } else {
                        out_counts[index] += run_length;
                    }
                }
            }
        }
        i = run_end;
    }
}

template <typename LabelA, typename LabelB>
PyObject* build_result(
    const std::vector<LabelA>& out_a,
    const std::vector<LabelB>& out_b,
    const std::vector<uint64_t>& out_counts,
    int typenum_a,
    int typenum_b
) {
    const npy_intp size = static_cast<npy_intp>(out_counts.size());
    npy_intp dims[1] = {size};
    PyObject* a_obj = PyArray_SimpleNew(1, dims, typenum_a);
    if (a_obj == nullptr) return nullptr;
    PyObject* b_obj = PyArray_SimpleNew(1, dims, typenum_b);
    if (b_obj == nullptr) { Py_DECREF(a_obj); return nullptr; }
    PyObject* counts_obj = PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (counts_obj == nullptr) { Py_DECREF(a_obj); Py_DECREF(b_obj); return nullptr; }

    if (size > 0) {
        std::memcpy(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(a_obj)),
            out_a.data(),
            out_a.size() * sizeof(LabelA)
        );
        std::memcpy(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(b_obj)),
            out_b.data(),
            out_b.size() * sizeof(LabelB)
        );
        std::memcpy(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(counts_obj)),
            out_counts.data(),
            out_counts.size() * sizeof(uint64_t)
        );
    }

    PyObject* result = PyTuple_New(3);
    if (result == nullptr) {
        Py_DECREF(a_obj);
        Py_DECREF(b_obj);
        Py_DECREF(counts_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, a_obj);
    PyTuple_SET_ITEM(result, 1, b_obj);
    PyTuple_SET_ITEM(result, 2, counts_obj);
    return result;
}

template <typename LabelA, typename LabelB>
PyObject* dispatch_typed(
    PyArrayObject* a,
    PyArrayObject* b,
    bool include_background,
    int typenum_a,
    int typenum_b
) {
    const auto* data_a = reinterpret_cast<const LabelA*>(PyArray_DATA(a));
    const auto* data_b = reinterpret_cast<const LabelB*>(PyArray_DATA(b));
    const std::size_t n = static_cast<std::size_t>(PyArray_SIZE(a));
    std::vector<LabelA> out_a;
    std::vector<LabelB> out_b;
    std::vector<uint64_t> counts;

    const bool ok = bindings::run_with_gil_released(
        [&] { count_pairs(data_a, data_b, n, include_background, out_a, out_b, counts); },
        "overlap kernel"
    );
    if (!ok) return nullptr;
    return build_result(out_a, out_b, counts, typenum_a, typenum_b);
}

bool same_shape(PyArrayObject* a, PyArrayObject* b) noexcept {
    const int ndim = PyArray_NDIM(a);
    if (ndim != PyArray_NDIM(b)) return false;
    for (int i = 0; i < ndim; ++i) {
        if (PyArray_DIM(a, i) != PyArray_DIM(b, i)) return false;
    }
    return true;
}

PyObject* py_overlap_counts(PyObject*, PyObject* args) {
    PyObject* obj_a = nullptr;
    PyObject* obj_b = nullptr;
    int include_background = 0;
    if (!PyArg_ParseTuple(args, "OOp:overlap_counts", &obj_a, &obj_b, &include_background)) {
        return nullptr;
    }

    auto* a = bindings::require_array(obj_a, "labels_a");
    if (a == nullptr) return nullptr;
    auto* b = bindings::require_array(obj_b, "labels_b");
    if (b == nullptr) return nullptr;

    if (!bindings::require_contiguous(a, "labels_a", false)) return nullptr;
    if (!bindings::require_contiguous(b, "labels_b", false)) return nullptr;
    if (!same_shape(a, b)) {
        PyErr_SetString(PyExc_ValueError, "labels_a and labels_b must have the same shape");
        return nullptr;
    }

    const int ta = PyArray_TYPE(a);
    const int tb = PyArray_TYPE(b);
    if ((ta != NPY_UINT32 && ta != NPY_UINT64) ||
        (tb != NPY_UINT32 && tb != NPY_UINT64)) {
        PyErr_SetString(PyExc_TypeError, "labels_a and labels_b dtypes must be uint32 or uint64");
        return nullptr;
    }

    const bool include_bg = include_background != 0;
    if (ta == NPY_UINT32 && tb == NPY_UINT32) {
        return dispatch_typed<uint32_t, uint32_t>(a, b, include_bg, ta, tb);
    }
    if (ta == NPY_UINT32 && tb == NPY_UINT64) {
        return dispatch_typed<uint32_t, uint64_t>(a, b, include_bg, ta, tb);
    }
    if (ta == NPY_UINT64 && tb == NPY_UINT32) {
        return dispatch_typed<uint64_t, uint32_t>(a, b, include_bg, ta, tb);
    }
    return dispatch_typed<uint64_t, uint64_t>(a, b, include_bg, ta, tb);
}

}  // namespace overlap_impl
