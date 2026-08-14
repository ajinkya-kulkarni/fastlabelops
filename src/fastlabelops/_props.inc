namespace props_impl {

template <typename Label>
struct RegionStats {
    Label label = Label{0};
    uint64_t area = 0;
    npy_intp min_row = 0;
    npy_intp min_col = 0;
    npy_intp max_row = 0;
    npy_intp max_col = 0;
    long double sum_row = 0;
    long double sum_col = 0;
};

template <typename Label>
void scan_labels(const Label* data, npy_intp height, npy_intp width, std::vector<RegionStats<Label>>& regions) {
    label_map_impl::FlatMap<Label, std::size_t> index;
    regions.reserve(16384);

    for (npy_intp row = 0; row < height; ++row) {
        const Label* row_data = data + row * width;
        for (npy_intp col = 0; col < width;) {
            const Label label = row_data[col];
            npy_intp run_end = col + 1;
            while (run_end < width && row_data[run_end] == label) ++run_end;

            if (label != Label{0}) {
                const uint64_t n = static_cast<uint64_t>(run_end - col);
                const long double sum_col =
                    (static_cast<long double>(col) + static_cast<long double>(run_end - 1)) * n / 2.0L;
                auto [region_index, inserted] = index.get_or_insert(label, regions.size());
                if (inserted) {
                    RegionStats<Label> r;
                    r.label = label;
                    r.area = n;
                    r.min_row = r.max_row = row;
                    r.min_col = col;
                    r.max_col = run_end - 1;
                    r.sum_row = static_cast<long double>(row) * n;
                    r.sum_col = sum_col;
                    regions.push_back(r);
                } else {
                    auto& r = regions[*region_index];
                    r.area += n;
                    r.min_row = std::min(r.min_row, row);
                    r.max_row = std::max(r.max_row, row);
                    r.min_col = std::min(r.min_col, col);
                    r.max_col = std::max(r.max_col, run_end - 1);
                    r.sum_row += static_cast<long double>(row) * n;
                    r.sum_col += sum_col;
                }
            }
            col = run_end;
        }
    }
}

template <typename Label>
PyObject* build_result(const std::vector<RegionStats<Label>>& regions, int typenum) {
    const npy_intp n = static_cast<npy_intp>(regions.size());
    npy_intp d1[1] = {n};
    npy_intp ds[2] = {n, 6};
    npy_intp dc[2] = {n, 2};
    PyObject* labels_obj = PyArray_SimpleNew(1, d1, typenum);
    PyObject* stats_obj = PyArray_SimpleNew(2, ds, NPY_INT64);
    PyObject* centroid_obj = PyArray_SimpleNew(2, dc, NPY_FLOAT64);
    if (!labels_obj || !stats_obj || !centroid_obj) {
        Py_XDECREF(labels_obj); Py_XDECREF(stats_obj); Py_XDECREF(centroid_obj); return nullptr;
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
    if (!result) { Py_DECREF(labels_obj); Py_DECREF(stats_obj); Py_DECREF(centroid_obj); return nullptr; }
    PyTuple_SET_ITEM(result, 0, labels_obj);
    PyTuple_SET_ITEM(result, 1, stats_obj);
    PyTuple_SET_ITEM(result, 2, centroid_obj);
    return result;
}

template <typename Label>
PyObject* dispatch(PyArrayObject* arr, int typenum) {
    std::vector<RegionStats<Label>> regions;
    int cpp_error = 0;
    Py_BEGIN_ALLOW_THREADS
    try {
        scan_labels(
            reinterpret_cast<const Label*>(PyArray_DATA(arr)),
            PyArray_DIM(arr, 0),
            PyArray_DIM(arr, 1),
            regions
        );
        std::sort(regions.begin(), regions.end(), [](const auto& a, const auto& b) { return a.label < b.label; });
    } catch (const std::bad_alloc&) {
        cpp_error = 1;
    } catch (...) {
        cpp_error = 2;
    }
    Py_END_ALLOW_THREADS

    if (cpp_error == 1) return PyErr_NoMemory();
    if (cpp_error == 2) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected C++ error in regionprops kernel");
        return nullptr;
    }
    return build_result(regions, typenum);
}

PyObject* py_regionprops2d(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    if (!PyArg_ParseTuple(args, "O:regionprops2d", &obj)) return nullptr;
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "labels must be a NumPy array");
        return nullptr;
    }

    auto* arr = reinterpret_cast<PyArrayObject*>(obj);
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "labels must be a 2D array");
        return nullptr;
    }
    if (!PyArray_ISCARRAY_RO(arr)) {
        PyErr_SetString(PyExc_ValueError, "labels must be aligned native-endian C-contiguous");
        return nullptr;
    }

    const int typenum = PyArray_TYPE(arr);
    if (typenum == NPY_UINT32) return dispatch<uint32_t>(arr, typenum);
    if (typenum == NPY_UINT64) return dispatch<uint64_t>(arr, typenum);
    PyErr_SetString(PyExc_TypeError, "labels dtype must be uint32 or uint64");
    return nullptr;
}

}  // namespace props_impl
