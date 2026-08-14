namespace relabel_impl {

template <typename Label, bool CheckOverflow>
bool relabel_kernel(Label* data, std::size_t n, uint64_t offset, std::size_t& count) {
    const uint64_t max_label = static_cast<uint64_t>(std::numeric_limits<Label>::max());
    label_map_impl::LabelMap<Label, Label> map(
        std::min(n, label_map_impl::kDefaultInitialCapacity)
    );
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
            // All foreground labels through max_label are already in use.
            // Any previously unseen foreground label would overflow the dtype.
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

    auto* arr = bindings::require_array(obj, "labels");
    if (arr == nullptr) return nullptr;
    if (!bindings::require_contiguous(arr, "labels", true)) return nullptr;

    const int typenum = PyArray_TYPE(arr);
    const std::size_t n = static_cast<std::size_t>(PyArray_SIZE(arr));
    if (typenum != NPY_UINT32 && typenum != NPY_UINT64) {
        PyErr_SetString(PyExc_TypeError, "labels dtype must be uint32 or uint64");
        return nullptr;
    }

    std::size_t count = 0;
    bool kernel_ok = true;
    bool ok = false;

    if (typenum == NPY_UINT32) {
        auto* data = reinterpret_cast<uint32_t*>(PyArray_DATA(arr));
        ok = bindings::run_with_gil_released(
            [&] { kernel_ok = dispatch_relabel(data, n, offset, count); },
            "relabel kernel"
        );
    } else {
        auto* data = reinterpret_cast<uint64_t*>(PyArray_DATA(arr));
        ok = bindings::run_with_gil_released(
            [&] { kernel_ok = dispatch_relabel(data, n, offset, count); },
            "relabel kernel"
        );
    }

    if (!ok) return nullptr;
    if (!kernel_ok) {
        PyErr_SetString(PyExc_OverflowError, "relabel output exceeds label dtype range");
        return nullptr;
    }
    return PyLong_FromSize_t(count);
}

}  // namespace relabel_impl
