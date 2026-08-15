namespace count_impl {

template <typename Label>
void accumulate_counts(
    const Label* data,
    std::size_t n,
    label_map_impl::LabelMap<Label, uint64_t>& counts,
    std::vector<Label>* observed_labels = nullptr,
    uint64_t* background_count = nullptr
) {
    for (std::size_t i = 0; i < n;) {
        const Label label = data[i];
        std::size_t run_end = i + 1;
        while (run_end < n && data[run_end] == label) ++run_end;
        const uint64_t run_length = static_cast<uint64_t>(run_end - i);

        if (label == Label{0}) {
            if (background_count != nullptr) {
                *background_count += run_length;
            }
        } else {
            auto [count, inserted] = counts.get_or_insert(label, uint64_t{0});
            if (inserted && observed_labels != nullptr) {
                observed_labels->push_back(label);
            }
            *count += run_length;
        }
        i = run_end;
    }
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

template <typename Label>
PyObject* dispatch_label_counts(PyArrayObject* arr, bool include_background, int typenum) {
    const auto* data = reinterpret_cast<const Label*>(PyArray_DATA(arr));
    const std::size_t n = static_cast<std::size_t>(PyArray_SIZE(arr));
    try {
        const std::size_t initial_capacity =
            std::min(n, label_map_impl::kDefaultInitialCapacity);
        label_map_impl::LabelMap<Label, uint64_t> counts(initial_capacity);
        std::vector<Label> labels;
        uint64_t background_count = 0;

        const bool ok = bindings::run_with_gil_released(
            [&] {
                labels.reserve(initial_capacity);
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

PyObject* py_label_counts(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    int include_background = 0;
    if (!PyArg_ParseTuple(args, "Op:label_counts", &obj, &include_background)) return nullptr;

    auto* arr = bindings::require_array(obj, "labels");
    if (arr == nullptr) return nullptr;
    if (!bindings::require_contiguous(arr, "labels", false)) return nullptr;

    const int typenum = PyArray_TYPE(arr);
    if (typenum == NPY_UINT32) {
        return dispatch_label_counts<uint32_t>(arr, include_background != 0, typenum);
    }
    if (typenum == NPY_UINT64) {
        return dispatch_label_counts<uint64_t>(arr, include_background != 0, typenum);
    }
    PyErr_SetString(PyExc_TypeError, "labels dtype must be uint32 or uint64");
    return nullptr;
}

}  // namespace count_impl
