// -*- c++ -*-
/*
 * Copyright (c) 2010-2012, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_BP_Array_h_INCLUDED
#define NDARRAY_BP_Array_h_INCLUDED

#include "boost/numpy.hpp"
#include "ndarray.h"

namespace ndarray {

inline void destroyManagerCObject(void * p) {
    Manager::Ptr * b = reinterpret_cast<Manager::Ptr*>(p);
    delete b;
}

inline boost::python::object makePyObject(Manager::Ptr const & x) {
    boost::intrusive_ptr< ExternalManager<boost::python::object> > y 
        = boost::dynamic_pointer_cast< ExternalManager<boost::python::object> >(x);
    if (y) {
        return y->getOwner();
    }
    boost::python::handle<> h(::PyCObject_FromVoidPtr(new Manager::Ptr(x), &destroyManagerCObject));
    return boost::python::object(h);
}

} // namespace ndarray

namespace boost { namespace python {

template <typename T, int N, int C>
struct to_python_value< ndarray::Array<T,N,C> const & > : public detail::builtin_to_python {
    inline PyObject * operator()(ndarray::Array<T,N,C> const & x) const {
        numpy::dtype dtype = numpy::dtype::get_builtin<typename boost::remove_const<T>::type>();
        object owner = makePyObject(x.getManager());
        int itemsize = dtype.get_itemsize();
        ndarray::Vector<int,N> shape_T = x.getShape();
        ndarray::Vector<int,N> strides_T = x.getStrides();
        std::vector<Py_intptr_t> shape_char(N);
        std::vector<Py_intptr_t> strides_char(N);
        for (int n=0; n<N; ++n) {
            shape_char[n] = shape_T[n];
            strides_char[n] = strides_T[n] * itemsize;
        }
        numpy::ndarray array = numpy::from_data(x.getData(), dtype, shape_char, strides_char, owner);
        Py_INCREF(array.ptr());
        return array.ptr();
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<numpy::ndarray>::get_pytype();
    }
};

template <typename T, int N, int C>
struct to_python_value< ndarray::Array<T,N,C> & > : public detail::builtin_to_python {
    inline PyObject * operator()(ndarray::Array<T,N,C> & x) const {
        return to_python_value< ndarray::Array<T,N,C> const & >()(x);
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<numpy::ndarray>::get_pytype();
    }
};

namespace converter {

template <typename T, int N, int C>
struct arg_to_python< ndarray::Array<T,N,C> > : public handle<> {
    inline arg_to_python(ndarray::Array<T,N,C> const & v) :
        handle<>(to_python_value<ndarray::Array<T,N,C> const &>()(v)) {}
};

template <typename T, int N, int C>
struct arg_rvalue_from_python< ndarray::Array<T,N,C> const & > {
    typedef ndarray::Array<T,N,C> result_type;

    static numpy::ndarray::bitflag const flags = 
        numpy::ndarray::bitflag(
            (boost::is_const<T>::value ? int(numpy::ndarray::WRITEABLE) : 0) |
            ((N==C) ? int(numpy::ndarray::C_CONTIGUOUS) : 0) |
            int(numpy::ndarray::ALIGNED)
        );

    arg_rvalue_from_python(PyObject * p) : arg(python::detail::borrowed_reference(p)) {}

    bool convertible() const {
        if (arg == object()) return true;
        try {
            numpy::ndarray array = extract<numpy::ndarray>(arg);
            numpy::dtype dtype = numpy::dtype::get_builtin<typename boost::remove_const<T>::type>();
            arg = numpy::from_object(array, dtype, N, flags);
        } catch (error_already_set) {
            handle_exception();
            PyErr_Clear();
            return false;
        }
        return true;
    }

    result_type operator()() const {
        if (arg == object()) return result_type();
        numpy::ndarray array = extract<numpy::ndarray>(arg);
        numpy::dtype dtype = array.get_dtype();
        int itemsize = dtype.get_itemsize();
        int total = itemsize;
        for (int i=1; i<=C; ++i) {
            if (array.strides(N-i) != total) {
                array = numpy::from_object(
                    array, dtype, N, 
                    flags | numpy::ndarray::C_CONTIGUOUS
                );
                break;
            }
            total *= array.shape(N-i);
        }
        object obj_owner = array.get_base();
        if (obj_owner == object()) {
            obj_owner = array;
        }
        ndarray::Vector<int,N> shape;
        ndarray::Vector<int,N> strides;
        for (int i=0; i<N; ++i) {
            shape[i] = array.shape(i);
            strides[i] = array.strides(i) / itemsize;
        }
        ndarray::Array<T,N,C> r = ndarray::external(
            reinterpret_cast<T*>(array.get_data()), shape, strides, obj_owner
        );
        return r;
    }

    mutable object arg;
};

template <typename T, int N, int C>
struct arg_rvalue_from_python< ndarray::Array<T,N,C> > 
    : public arg_rvalue_from_python< ndarray::Array<T,N,C> const &> 
{

    arg_rvalue_from_python(PyObject * p) : 
        arg_rvalue_from_python< ndarray::Array<T,N,C> const & >(p) {}

};

template <typename T, int N, int C>
struct arg_rvalue_from_python< ndarray::Array<T,N,C> const > 
    : public arg_rvalue_from_python< ndarray::Array<T,N,C> const &> 
{

    arg_rvalue_from_python(PyObject * p) : 
        arg_rvalue_from_python< ndarray::Array<T,N,C> const & >(p) {}

};

template <typename T, int N, int C>
struct extract_rvalue< ndarray::Array<T,N,C> > : private noncopyable {
    typedef ndarray::Array<T,N,C> result_type;

    extract_rvalue(PyObject * x) : m_converter(x) {}

    bool check() const { return m_converter.convertible(); }
    
    result_type operator()() const { return m_converter(); }

private:
    arg_rvalue_from_python< result_type const & > m_converter;
};

}} // namespace python::converter

namespace numpy {

template <typename T, int N, int C>
numpy::ndarray array(::ndarray::Array<T,N,C> const & arg) {
    python::to_python_value< ::ndarray::Array<T,N,C> const & > converter;
    numpy::ndarray result(python::detail::new_reference(converter(arg)));
    return result;
}

}} // namespace boost::numpy

#endif // !NDARRAY_BP_Array_h_INCLUDED
