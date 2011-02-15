#ifndef BOOST_PYTHON_NDARRAY_VECTOR_HPP_INCLUDED
#define BOOST_PYTHON_NDARRAY_VECTOR_HPP_INCLUDED

#include <boost/python/numpy/numpy.hpp>
#include <ndarray/Vector.hpp>

namespace boost { namespace python {

template <typename T, int N>
struct to_python_value< ndarray::Vector<T,N> const & > : public detail::builtin_to_python {
    inline PyObject * operator()(ndarray::Vector<T,N> const & x) const {
        handle<> t(PyTuple_New(N));
        try {
            for (int n=0; n<N; ++n) {
                object item(x[n]);
                Py_INCREF(item.ptr());
                PyTuple_SET_ITEM(t.get(), n, item.ptr());
            }
        } catch (error_already_set & err) {
            handle_exception();
            return NULL;
        }
        return t.release();
    }
    inline PyTypeObject const * get_pytype() const { return &PyTuple_Type; }
};

template <typename T, int N>
struct to_python_value< ndarray::Vector<T,N> & > : public detail::builtin_to_python {
    inline PyObject * operator()(ndarray::Vector<T,N> & x) const {
        return to_python_value< ndarray::Vector<T,N> & >()(x);
    }
    inline PyTypeObject const * get_pytype() const { return &PyTuple_Type; }
};

namespace converter {

template <typename T, int N>
struct arg_to_python< ndarray::Vector<T,N> > : public handle<> {
    inline arg_to_python(ndarray::Vector<T,N> const & v) :
        handle<>(to_python_value<ndarray::Vector<T,N> const &>()(v)) {}
};

template <typename T, int N>
struct arg_rvalue_from_python< ndarray::Vector<T,N> const & > {
    typedef ndarray::Vector<T,N> result_type;

    arg_rvalue_from_python(PyObject * p) : _p(python::detail::borrowed_reference(p)) {}

    bool convertible() const {
        try {
            tuple t(_p);
            if (len(t) != N) return false;
            _p = t;
        } catch (error_already_set) {
            handle_exception();
            PyErr_Clear();
            return false;
        }
        return true;
    }

    result_type operator()() const {
        tuple t = extract<tuple>(_p);
        if (len(t) != N) {
            PyErr_SetString(PyExc_ValueError, "Incorrect size for ndarray::Vector.");
            throw_error_already_set();
        }
        result_type r;
        for (int n=0; n<N; ++n) {
            r[n] = extract<T>(t[n]);
        }
        return r;
    }

private:
    mutable object _p;
};

template <typename T, int N>
struct extract_rvalue< ndarray::Vector<T,N> > : private noncopyable {
    typedef ndarray::Vector<T,N> result_type;

    extract_rvalue(PyObject * x) : m_converter(x) {}

    bool check() const { return m_converter.convertible(); }
    
    result_type operator()() const { return m_converter(); }

private:
    arg_rvalue_from_python< result_type const & > m_converter;
};

} // namespace boost::python::converter
}} // namespace boost::python

#endif // !BOOST_PYTHON_NDARRAY_VECTOR_HPP_INCLUDED
