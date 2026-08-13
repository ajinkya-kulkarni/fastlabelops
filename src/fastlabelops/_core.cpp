#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

#include "_relabel.inc"
#include "_overlap.inc"
#include "_props.inc"

namespace {

PyMethodDef methods[] = {
    {"relabel_inplace", relabel_impl::py_relabel_inplace, METH_VARARGS,
     "Relabel a writable contiguous uint array in place."},
    {"overlap_counts", overlap_impl::py_overlap_counts, METH_VARARGS,
     "Count observed sparse label-pair overlaps between two arrays."},
    {"regionprops2d", props_impl::py_regionprops2d, METH_VARARGS,
     "Compute common 2D label properties."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "fastlabelops C++ core",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModule_Create(&module);
}
