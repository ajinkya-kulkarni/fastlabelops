namespace count_impl {

// Experimental direct-address path for compact uint32 IDs. The sample avoids
// allocating the dense counter array when the input is obviously sparse.
constexpr uint32_t kDirectMaxLabel = 1u << 20;
constexpr std::size_t kDirectProbeSamples = 4096;

template <typename Label>
void accumulate_counts(
    const Label* data,
    std::size_t n,
    label_map_impl::LabelMap<Label, uint64_t>& counts,
    std::vector<Label>* observed_labels = nullptr,
    uint64_t* background_count = nullptr
) {
    runs_impl::for_each_run(data, n, [&](Label label, std::size_t start, std::size_t end) {
        const uint64_t run_length = static_cast<uint64_t>(end - start);
        if (label == Label{0}) {
            if (background_count != nullptr) *background_count += run_length;
            return;
        }
        auto [count, inserted] = counts.get_or_insert(label, uint64_t{0});
        if (inserted && observed_labels != nullptr) observed_labels->push_back(label);
        *count += run_length;
    });
}

template <typename Label>
PyObject* build_result(
    const std::vector<Label>& labels,
    const label_map_impl::LabelMap<Label, uint64_t>& counts,
    uint64_t background_count,
    bool include_background,
    int typenum
) {
    const bool emit_background = include_background && background_count > 0;
    const npy_intp size = static_cast<npy_intp>(labels.size() + (emit_background ? 1 : 0));
    npy_intp dims[1] = {size};
    PyObject* labels_obj = PyArray_SimpleNew(1, dims, typenum);
    if (labels_obj == nullptr) return nullptr;
    PyObject* counts_obj = PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (counts_obj == nullptr) {
        Py_DECREF(labels_obj);
        return nullptr;
    }

    auto* out_labels = reinterpret_cast<Label*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(labels_obj))
    );
    auto* out_counts = reinterpret_cast<uint64_t*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(counts_obj))
    );

    std::size_t out_i = 0;
    if (emit_background) {
        out_labels[out_i] = Label{0};
        out_counts[out_i] = background_count;
        ++out_i;
    }
    for (Label label : labels) {
        out_labels[out_i] = label;
        out_counts[out_i] = *counts.find(label);
        ++out_i;
    }

    PyObject* result = PyTuple_New(2);
    if (result == nullptr) {
        Py_DECREF(labels_obj);
        Py_DECREF(counts_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, labels_obj);
    PyTuple_SET_ITEM(result, 1, counts_obj);
    return result;
}

PyObject* build_direct_result(
    const std::vector<uint64_t>& counts,
    std::size_t unique_count,
    uint64_t background_count,
    bool include_background
) {
    const bool emit_background = include_background && background_count > 0;
    const npy_intp size = static_cast<npy_intp>(unique_count + (emit_background ? 1 : 0));
    npy_intp dims[1] = {size};
    PyObject* labels_obj = PyArray_SimpleNew(1, dims, NPY_UINT32);
    if (labels_obj == nullptr) return nullptr;
    PyObject* counts_obj = PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (counts_obj == nullptr) {
        Py_DECREF(labels_obj);
        return nullptr;
    }

    auto* out_labels = reinterpret_cast<uint32_t*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(labels_obj))
    );
    auto* out_counts = reinterpret_cast<uint64_t*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(counts_obj))
    );

    std::size_t out_i = 0;
    if (emit_background) {
        out_labels[out_i] = 0;
        out_counts[out_i] = background_count;
        ++out_i;
    }
    for (std::size_t label = 1; label < counts.size(); ++label) {
        if (counts[label] == 0) continue;
        out_labels[out_i] = static_cast<uint32_t>(label);
        out_counts[out_i] = counts[label];
        ++out_i;
    }

    PyObject* result = PyTuple_New(2);
    if (result == nullptr) {
        Py_DECREF(labels_obj);
        Py_DECREF(counts_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, labels_obj);
    PyTuple_SET_ITEM(result, 1, counts_obj);
    return result;
}

bool sample_for_direct(
    const uint32_t* data,
    std::size_t n,
    uint32_t& sampled_max
) noexcept {
    sampled_max = 0;
    if (n == 0) return true;
    const std::size_t samples = std::min(n, kDirectProbeSamples);
    for (std::size_t s = 0; s < samples; ++s) {
        const std::size_t i = (s * n) / samples;
        sampled_max = std::max(sampled_max, data[i]);
        if (sampled_max > kDirectMaxLabel) return false;
    }
    return true;
}

std::size_t direct_capacity_for(uint32_t label) noexcept {
    std::size_t cap = 16;
    const std::size_t needed = static_cast<std::size_t>(label) + 1;
    while (cap < needed && cap < static_cast<std::size_t>(kDirectMaxLabel) + 1) {
        cap <<= 1;
    }
    return std::min(cap, static_cast<std::size_t>(kDirectMaxLabel) + 1);
}

// Count directly by label ID when IDs stay below kDirectMaxLabel. If sampling
// misses a large outlier, return false and rerun safely through the hash path.
bool try_direct_counts(
    const uint32_t* data,
    std::size_t n,
    bool include_background,
    std::vector<uint64_t>& counts,
    std::size_t& unique_count,
    uint64_t& background_count
) {
    uint32_t sampled_max = 0;
    if (!sample_for_direct(data, n, sampled_max)) return false;

    counts.assign(direct_capacity_for(sampled_max), 0);
    unique_count = 0;
    background_count = 0;
    uint32_t max_seen = 0;

    for (std::size_t i = 0; i < n;) {
        const uint32_t label = data[i];
        std::size_t run_end = i + 1;
        while (run_end < n && data[run_end] == label) ++run_end;
        const uint64_t run_length = static_cast<uint64_t>(run_end - i);

        if (label == 0) {
            if (include_background) background_count += run_length;
        } else {
            if (label > kDirectMaxLabel) return false;
            if (label >= counts.size()) counts.resize(direct_capacity_for(label), 0);
            uint64_t& count = counts[label];
            if (count == 0) ++unique_count;
            count += run_length;
            max_seen = std::max(max_seen, label);
        }
        i = run_end;
    }

    counts.resize(static_cast<std::size_t>(max_seen) + 1);
    return true;
}

template <typename Label>
PyObject* dispatch_label_counts_hash(PyArrayObject* arr, bool include_background, int typenum) {
    const auto* data = reinterpret_cast<const Label*>(PyArray_DATA(arr));
    const std::size_t n = static_cast<std::size_t>(PyArray_SIZE(arr));
    try {
        const std::size_t expected_entries =
            std::min(n, label_map_impl::kInitialReserveLimit);
        label_map_impl::LabelMap<Label, uint64_t> counts(expected_entries);
        std::vector<Label> labels;
        uint64_t background_count = 0;

        const bool ok = bindings::run_with_gil_released(
            [&] {
                labels.reserve(expected_entries);
                accumulate_counts(
                    data,
                    n,
                    counts,
                    &labels,
                    include_background ? &background_count : nullptr
                );
                if (!std::is_sorted(labels.begin(), labels.end())) {
                    std::sort(labels.begin(), labels.end());
                }
            },
            "label-counts kernel"
        );
        if (!ok) return nullptr;

        return build_result(labels, counts, background_count, include_background, typenum);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected C++ error in label-counts setup");
        return nullptr;
    }
}

PyObject* dispatch_label_counts_uint32(PyArrayObject* arr, bool include_background) {
    const auto* data = reinterpret_cast<const uint32_t*>(PyArray_DATA(arr));
    const std::size_t n = static_cast<std::size_t>(PyArray_SIZE(arr));
    try {
        std::vector<uint64_t> direct_counts;
        std::size_t unique_count = 0;
        uint64_t background_count = 0;
        bool used_direct = false;
        const bool ok = bindings::run_with_gil_released(
            [&] {
                used_direct = try_direct_counts(
                    data, n, include_background, direct_counts, unique_count, background_count
                );
            },
            "direct label-counts kernel"
        );
        if (!ok) return nullptr;
        if (used_direct) {
            return build_direct_result(
                direct_counts, unique_count, background_count, include_background
            );
        }
        return dispatch_label_counts_hash<uint32_t>(arr, include_background, NPY_UINT32);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected C++ error in direct label-counts setup");
        return nullptr;
    }
}

PyObject* py_label_counts(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    int include_background = 0;
    if (!PyArg_ParseTuple(args, "Op:label_counts", &obj, &include_background)) return nullptr;

    auto* arr = bindings::require_array(obj, "labels");
    if (arr == nullptr) return nullptr;
    if (!bindings::require_contiguous(arr, "labels", false)) return nullptr;

    const int typenum = PyArray_TYPE(arr);
    if (typenum == NPY_UINT32) {
        return dispatch_label_counts_uint32(arr, include_background != 0);
    }
    if (typenum == NPY_UINT64) {
        return dispatch_label_counts_hash<uint64_t>(arr, include_background != 0, typenum);
    }
    PyErr_SetString(PyExc_TypeError, "labels dtype must be uint32 or uint64");
    return nullptr;
}

}  // namespace count_impl
