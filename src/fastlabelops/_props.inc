namespace props_impl {

template <typename Label>
struct RegionStats {
    Label label = Label{0};
    uint64_t area = 0;
    int64_t min_row = 0;
    int64_t min_col = 0;
    int64_t max_row = 0;
    int64_t max_col = 0;
    long double sum_row = 0;
    long double sum_col = 0;
};

template <typename Label>
void scan_labels(
    const Label* data,
    npy_intp height,
    npy_intp width,
    std::size_t size,
    std::vector<RegionStats<Label>>& regions
) {
    const std::size_t expected_entries =
        std::min(size, label_map_impl::kInitialReserveLimit);
    label_map_impl::LabelMap<Label, std::size_t> index(expected_entries);
    regions.reserve(expected_entries);

    for (npy_intp row = 0; row < height; ++row) {
        const Label* row_data = data + row * width;
        for (npy_intp col = 0; col < width;) {
            const Label label = row_data[col];
            npy_intp run_end = col + 1;
            while (run_end < width && row_data[run_end] == label) ++run_end;

            if (label != Label{0}) {
                const uint64_t n = static_cast<uint64_t>(run_end - col);
                const int64_t col_start = static_cast<int64_t>(col);
                const int64_t col_end = static_cast<int64_t>(run_end);
                const int64_t row_i = static_cast<int64_t>(row);
                const long double sum_col =
                    (static_cast<long double>(col_start) +
                     static_cast<long double>(col_end - 1)) *
                    n / 2.0L;

                auto [region_index, inserted] = index.get_or_insert(label, regions.size());
                if (inserted) {
                    RegionStats<Label> region;
                    region.label = label;
                    region.area = n;
                    region.min_row = region.max_row = row_i;
                    region.min_col = col_start;
                    region.max_col = col_end - 1;
                    region.sum_row = static_cast<long double>(row_i) * n;
                    region.sum_col = sum_col;
                    regions.push_back(region);
                } else {
                    auto& region = regions[*region_index];
                    region.area += n;
                    region.min_row = std::min(region.min_row, row_i);
                    region.max_row = std::max(region.max_row, row_i);
                    region.min_col = std::min(region.min_col, col_start);
                    region.max_col = std::max(region.max_col, col_end - 1);
                    region.sum_row += static_cast<long double>(row_i) * n;
                    region.sum_col += sum_col;
                }
            }
            col = run_end;
        }
    }
    const auto ordered_by_label = [](const auto& a, const auto& b) { return a.label < b.label; };
    if (!std::is_sorted(regions.begin(), regions.end(), ordered_by_label)) {
        std::sort(regions.begin(), regions.end(), ordered_by_label);
    }
}

template <typename Label>
PyObject* build_result(const std::vector<RegionStats<Label>>& regions, int typenum) {
    const npy_intp n = static_cast<npy_intp>(regions.size());
    npy_intp d1[1] = {n};
    npy_intp ds[2] = {n, 6};
    npy_intp dc[2] = {n, 2};
    PyObject* labels_obj = PyArray_SimpleNew(1, d1, typenum);
    if (labels_obj == nullptr) return nullptr;
    PyObject* stats_obj = PyArray_SimpleNew(2, ds, NPY_INT64);
    if (stats_obj == nullptr) {
        Py_DECREF(labels_obj);
        return nullptr;
    }
    PyObject* centroid_obj = PyArray_SimpleNew(2, dc, NPY_FLOAT64);
    if (centroid_obj == nullptr) {
        Py_DECREF(labels_obj);
        Py_DECREF(stats_obj);
        return nullptr;
    }

    auto* labels = reinterpret_cast<Label*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(labels_obj)));
    auto* stats = reinterpret_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(stats_obj)));
    auto* centroids = reinterpret_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(centroid_obj)));

    for (std::size_t i = 0; i < regions.size(); ++i) {
        const auto& r = regions[i];
        labels[i] = r.label;
        const std::size_t s = i * 6;
        stats[s] = static_cast<int64_t>(r.area);
        stats[s + 1] = static_cast<int64_t>(r.min_row);
        stats[s + 2] = static_cast<int64_t>(r.min_col);
        stats[s + 3] = static_cast<int64_t>(r.max_row + 1);
        stats[s + 4] = static_cast<int64_t>(r.max_col + 1);
        stats[s + 5] = static_cast<int64_t>((r.max_row - r.min_row + 1) * (r.max_col - r.min_col + 1));
        const std::size_t c = i * 2;
        centroids[c] = static_cast<double>(r.sum_row / r.area);
        centroids[c + 1] = static_cast<double>(r.sum_col / r.area);
    }

    PyObject* result = PyTuple_New(3);
    if (!result) {
        Py_DECREF(labels_obj);
        Py_DECREF(stats_obj);
        Py_DECREF(centroid_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, labels_obj);
    PyTuple_SET_ITEM(result, 1, stats_obj);
    PyTuple_SET_ITEM(result, 2, centroid_obj);
    return result;
}

template <typename Label>
PyObject* dispatch(PyArrayObject* arr, int typenum) {
    const auto* data = reinterpret_cast<const Label*>(PyArray_DATA(arr));
    const npy_intp height = PyArray_DIM(arr, 0);
    const npy_intp width = PyArray_DIM(arr, 1);
    const std::size_t size = static_cast<std::size_t>(PyArray_SIZE(arr));
    std::vector<RegionStats<Label>> regions;
    const bool ok = bindings::run_with_gil_released(
        [&] { scan_labels(data, height, width, size, regions); },
        "regionprops kernel"
    );
    if (!ok) return nullptr;
    return build_result(regions, typenum);
}

PyObject* py_regionprops2d(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    if (!PyArg_ParseTuple(args, "O:regionprops2d", &obj)) return nullptr;

    auto* arr = bindings::require_array(obj, "labels");
    if (arr == nullptr) return nullptr;
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "labels must be a 2D array");
        return nullptr;
    }
    if (!bindings::require_contiguous(arr, "labels", false)) return nullptr;

    const int typenum = PyArray_TYPE(arr);
    if (typenum != NPY_UINT32 && typenum != NPY_UINT64) {
        PyErr_SetString(PyExc_TypeError, "labels dtype must be uint32 or uint64");
        return nullptr;
    }

    if (typenum == NPY_UINT32) {
        return dispatch<uint32_t>(arr, typenum);
    }
    return dispatch<uint64_t>(arr, typenum);
}

}  // namespace props_impl
