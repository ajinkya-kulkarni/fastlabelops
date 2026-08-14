namespace relabel_impl {

template <typename Label, bool CheckOverflow>
bool relabel_kernel(Label* data, std::size_t n, uint64_t offset, std::size_t& count) {
    const uint64_t max_label = static_cast<uint64_t>(std::numeric_limits<Label>::max());
    label_map_impl::FlatMap<Label, Label> map;
    uint64_t next = offset;
    count = 0;
    Label prev_old = Label{0};
    Label prev_new = Label{0};

    for (std::size_t i = 0; i < n; ++i) {
        const Label old = data[i];
        if (old == Label{0}) continue;
        if (old == prev_old) {
            data[i] = prev_new;
            continue;
        }

        if constexpr (CheckOverflow) {
            if (next == max_label) {
                auto [mapped_ptr, inserted] = map.get_or_insert(old, Label{0});
                if (inserted) return false;
                const Label mapped = *mapped_ptr;
                data[i] = mapped;
                prev_old = old;
                prev_new = mapped;
                continue;
            }
        }

        const Label candidate = static_cast<Label>(next + 1);
        auto [mapped_ptr, inserted] = map.get_or_insert(old, candidate);
        const Label mapped = *mapped_ptr;
        if (inserted) {
            ++next;
            ++count;
        }
        data[i] = mapped;
        prev_old = old;
        prev_new = mapped;
    }
    return true;
}

template <typename Label>
bool dispatch_relabel(Label* data, std::size_t n, uint64_t offset, std::size_t& count) {
    const uint64_t max_label = static_cast<uint64_t>(std::numeric_limits<Label>::max());
    if (offset > max_label) return false;
    const uint64_t n64 = static_cast<uint64_t>(n);
    if (n64 <= max_label - offset) {
        return relabel_kernel<Label, false>(data, n, offset, count);
    }
    return relabel_kernel<Label, true>(data, n, offset, count);
}

PyObject* py_relabel_inplace(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    unsigned long long offset = 0;
    if (!PyArg_ParseTuple(args, "OK:relabel_inplace", &obj, &offset)) return nullptr;
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
    const std::size_t n = static_cast<std::size_t>(PyArray_SIZE(arr));
    std::size_t count = 0;
    bool ok = false;
    int cpp_error = 0;

    Py_BEGIN_ALLOW_THREADS
    try {
        if (typenum == NPY_UINT32) {
            ok = dispatch_relabel(reinterpret_cast<uint32_t*>(PyArray_DATA(arr)), n, offset, count);
        } else if (typenum == NPY_UINT64) {
            ok = dispatch_relabel(reinterpret_cast<uint64_t*>(PyArray_DATA(arr)), n, offset, count);
        }
    } catch (const std::bad_alloc&) {
        cpp_error = 1;
    } catch (...) {
        cpp_error = 2;
    }
    Py_END_ALLOW_THREADS

    if (cpp_error == 1) return PyErr_NoMemory();
    if (cpp_error == 2) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected C++ error in relabel kernel");
        return nullptr;
    }
    if (typenum != NPY_UINT32 && typenum != NPY_UINT64) {
        PyErr_SetString(PyExc_TypeError, "dtype must be uint32 or uint64");
        return nullptr;
    }
    if (!ok) {
        PyErr_SetString(PyExc_OverflowError, "relabel output exceeds label dtype range");
        return nullptr;
    }
    return PyLong_FromSize_t(count);
}

}  // namespace relabel_impl
