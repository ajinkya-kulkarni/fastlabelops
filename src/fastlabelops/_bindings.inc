#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION
#include <numpy/arrayobject.h>

#include <new>

namespace bindings {

inline PyArrayObject* require_array(PyObject* obj, const char* name) {
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return nullptr;
    }
    return reinterpret_cast<PyArrayObject*>(obj);
}

inline bool require_contiguous(PyArrayObject* arr, const char* name, bool writable) {
    if (writable) {
        if (!PyArray_ISCARRAY(arr)) {
            PyErr_Format(PyExc_ValueError, "%s must be writable and C-contiguous", name);
            return false;
        }
    } else {
        if (!PyArray_ISCARRAY_RO(arr)) {
            PyErr_Format(PyExc_ValueError, "%s must be C-contiguous and aligned", name);
            return false;
        }
    }
    if (!PyArray_ISNBO(PyArray_DESCR(arr)->byteorder)) {
        PyErr_Format(PyExc_ValueError, "%s must have native byte order", name);
        return false;
    }
    return true;
}

template <typename Func>
bool run_with_gil_released(Func&& func, const char* context) {
    int cpp_error = 0;
    Py_BEGIN_ALLOW_THREADS
    try {
        func();
    } catch (const std::bad_alloc&) {
        cpp_error = 1;
    } catch (...) {
        cpp_error = 2;
    }
    Py_END_ALLOW_THREADS

    if (cpp_error == 1) {
        PyErr_NoMemory();
        return false;
    }
    if (cpp_error == 2) {
        PyErr_Format(PyExc_RuntimeError, "unexpected C++ error in %s", context);
        return false;
    }
    return true;
}

}  // namespace bindings
