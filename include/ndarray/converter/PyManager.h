// -*- c++ -*-
/*
 * Copyright (c) 2010-2018, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_CONVERTER_PyManager_h_INCLUDED
#define NDARRAY_CONVERTER_PyManager_h_INCLUDED

#include "Python.h"
#include "ndarray/Manager.h"

namespace ndarray {
namespace detail {

inline void destroyCapsule(PyObject * p) {
    void * m = PyCapsule_GetPointer(p, "ndarray.Manager");
    Manager::Ptr * b = reinterpret_cast<Manager::Ptr*>(m);
    delete b;
}

} // namespace ndarray::detail

inline PyObject* makePyManager(Manager::Ptr const & m) {
    return PyCapsule_New(
        new Manager::Ptr(m),
        "ndarray.Manager",
        detail::destroyCapsule
    );
}

} // namespace ndarray

#endif // !NDARRAY_CONVERTER_PyManager_h_INCLUDED
