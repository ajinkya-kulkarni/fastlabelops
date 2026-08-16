namespace remove_small_impl {

template <typename Label>
void remove_small_kernel(Label* data, std::size_t n, uint64_t max_size) {
    if (n == 0 || max_size == 0) return;
    if (max_size >= static_cast<uint64_t>(n)) {
        std::fill(data, data + n, Label{0});
        return;
    }

    label_map_impl::LabelMap<Label, uint64_t> counts(
        std::min(n, label_map_impl::kInitialReserveLimit)
    );
    count_impl::accumulate_counts(data, n, counts);

    bool remove_any = false;
    bool keep_any = false;
    counts.for_each([&](Label, uint64_t count) {
        remove_any = remove_any || count <= max_size;
        keep_any = keep_any || count > max_size;
    });
    if (!remove_any) return;
    if (!keep_any) {
        std::fill(data, data + n, Label{0});
        return;
    }

    runs_impl::for_each_run(
        data,
        n,
        [&](Label label, std::size_t start, std::size_t end) {
            if (label == Label{0}) return;
            const uint64_t* count = counts.find(label);
            if (count != nullptr && *count <= max_size) {
                std::fill(data + start, data + end, Label{0});
            }
        }
    );
}

PyObject* py_remove_small_objects_inplace(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    unsigned long long max_size = 0;
    if (!PyArg_ParseTuple(args, "OK:remove_small_objects_inplace", &obj, &max_size)) return nullptr;

    auto* arr = bindings::require_array(obj, "labels");
    if (arr == nullptr) return nullptr;
    if (!bindings::require_contiguous(arr, "labels", true)) return nullptr;

    const int typenum = PyArray_TYPE(arr);
    if (typenum != NPY_UINT32 && typenum != NPY_UINT64) {
        PyErr_SetString(PyExc_TypeError, "labels dtype must be uint32 or uint64");
        return nullptr;
    }

    const std::size_t n = static_cast<std::size_t>(PyArray_SIZE(arr));
    bool ok = false;
    if (typenum == NPY_UINT32) {
        auto* data = reinterpret_cast<uint32_t*>(PyArray_DATA(arr));
        ok = bindings::run_with_gil_released(
            [&] { remove_small_kernel(data, n, max_size); },
            "remove-small-objects kernel"
        );
    } else {
        auto* data = reinterpret_cast<uint64_t*>(PyArray_DATA(arr));
        ok = bindings::run_with_gil_released(
            [&] { remove_small_kernel(data, n, max_size); },
            "remove-small-objects kernel"
        );
    }

    if (!ok) return nullptr;
    Py_RETURN_NONE;
}

}  // namespace remove_small_impl
