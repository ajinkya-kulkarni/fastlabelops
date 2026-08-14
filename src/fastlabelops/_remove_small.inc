namespace remove_small_impl {

template <typename Label>
void remove_small_kernel(Label* data, std::size_t n, uint64_t max_size) {
    if (n == 0 || max_size == 0) return;
    if (max_size >= static_cast<uint64_t>(n)) {
        std::fill(data, data + n, Label{0});
        return;
    }

    label_map_impl::FlatMap<Label, uint64_t> counts;

    for (std::size_t i = 0; i < n;) {
        const Label label = data[i];
        std::size_t run_end = i + 1;
        while (run_end < n && data[run_end] == label) ++run_end;

        if (label != Label{0}) {
            auto [count, inserted] = counts.get_or_insert(label, uint64_t{0});
            (void)inserted;
            *count += static_cast<uint64_t>(run_end - i);
        }
        i = run_end;
    }

    for (std::size_t i = 0; i < n;) {
        const Label label = data[i];
        std::size_t run_end = i + 1;
        while (run_end < n && data[run_end] == label) ++run_end;

        if (label != Label{0}) {
            const uint64_t* count = counts.find(label);
            if (count != nullptr && *count <= max_size) {
                std::fill(data + i, data + run_end, Label{0});
            }
        }
        i = run_end;
    }
}

template <typename Label>
void dispatch_remove_small(PyArrayObject* arr, uint64_t max_size) {
    remove_small_kernel(
        reinterpret_cast<Label*>(PyArray_DATA(arr)),
        static_cast<std::size_t>(PyArray_SIZE(arr)),
        max_size
    );
}

PyObject* py_remove_small_objects_inplace(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    unsigned long long max_size = 0;
    if (!PyArg_ParseTuple(args, "OK:remove_small_objects_inplace", &obj, &max_size)) return nullptr;
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "labels must be a NumPy array");
        return nullptr;
    }

    auto* arr = reinterpret_cast<PyArrayObject*>(obj);
    if (!PyArray_ISCARRAY(arr)) {
        PyErr_SetString(PyExc_ValueError, "array must be writable and C-contiguous");
        return nullptr;
    }

    const int typenum = PyArray_TYPE(arr);
    int cpp_error = 0;

    Py_BEGIN_ALLOW_THREADS
    try {
        if (typenum == NPY_UINT32) {
            dispatch_remove_small<uint32_t>(arr, max_size);
        } else if (typenum == NPY_UINT64) {
            dispatch_remove_small<uint64_t>(arr, max_size);
        }
    } catch (const std::bad_alloc&) {
        cpp_error = 1;
    } catch (...) {
        cpp_error = 2;
    }
    Py_END_ALLOW_THREADS

    if (cpp_error == 1) return PyErr_NoMemory();
    if (cpp_error == 2) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected C++ error in remove-small-objects kernel");
        return nullptr;
    }
    if (typenum != NPY_UINT32 && typenum != NPY_UINT64) {
        PyErr_SetString(PyExc_TypeError, "dtype must be uint32 or uint64");
        return nullptr;
    }
    Py_RETURN_NONE;
}

}  // namespace remove_small_impl
