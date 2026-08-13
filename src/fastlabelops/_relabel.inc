namespace relabel_impl {

template <typename Label>
struct Entry {
    Label key = Label{0};
    Label value = Label{0};
};

template <typename Label>
class FlatMap {
public:
    explicit FlatMap(std::size_t initial_capacity = 16384) {
        std::size_t cap = 16;
        while (cap < initial_capacity) cap <<= 1;
        entries_.resize(cap);
        configure_capacity(cap);
    }

    inline bool get_or_insert(Label key, Label candidate, Label& value) {
        std::size_t i = index_for(key);
        for (;;) {
            Entry<Label>& e = entries_[i];
            if (e.key == key) {
                value = e.value;
                return false;
            }
            if (e.key == Label{0}) {
                if (size_ >= grow_at_) {
                    grow();
                    return get_or_insert(key, candidate, value);
                }
                e.key = key;
                e.value = candidate;
                value = candidate;
                ++size_;
                return true;
            }
            i = (i + 1) & mask_;
        }
    }

private:
    inline std::size_t index_for(Label key) const noexcept {
        constexpr uint64_t kFib = 11400714819323198485ULL;
        return static_cast<std::size_t>((static_cast<uint64_t>(key) * kFib) >> shift_);
    }

    inline void insert_no_grow(Label key, Label value) noexcept {
        std::size_t i = index_for(key);
        while (entries_[i].key != Label{0}) i = (i + 1) & mask_;
        entries_[i].key = key;
        entries_[i].value = value;
        ++size_;
    }

    void configure_capacity(std::size_t cap) noexcept {
        mask_ = cap - 1;
        unsigned bits = 0;
        for (std::size_t x = cap; x > 1; x >>= 1) ++bits;
        shift_ = 64u - bits;
        grow_at_ = (cap * 7) / 10;
    }

    void grow() {
        std::vector<Entry<Label>> old = std::move(entries_);
        const std::size_t new_cap = old.size() * 2;
        entries_.clear();
        entries_.resize(new_cap);
        configure_capacity(new_cap);
        size_ = 0;
        for (const Entry<Label>& e : old) {
            if (e.key != Label{0}) insert_no_grow(e.key, e.value);
        }
    }

    std::vector<Entry<Label>> entries_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
    std::size_t grow_at_ = 0;
    unsigned shift_ = 0;
};

template <typename Label, bool CheckOverflow>
bool relabel_kernel(Label* data, std::size_t n, uint64_t offset, std::size_t& count) {
    const uint64_t max_label = static_cast<uint64_t>(std::numeric_limits<Label>::max());
    FlatMap<Label> map;
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
                Label mapped;
                const bool inserted = map.get_or_insert(old, Label{0}, mapped);
                if (inserted) return false;
                data[i] = mapped;
                prev_old = old;
                prev_new = mapped;
                continue;
            }
        }

        Label mapped;
        const Label candidate = static_cast<Label>(next + 1);
        if (map.get_or_insert(old, candidate, mapped)) {
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
