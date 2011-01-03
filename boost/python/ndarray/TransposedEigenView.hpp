#ifndef BOOST_PYTHON_NDARRAY_TRANSPOSEDEIGENVIEW_HPP_INCLUDED
#define BOOST_PYTHON_NDARRAY_TRANSPOSEDEIGENVIEW_HPP_INCLUDED

#include <boost/python/ndarray/Array.hpp>

namespace boost { namespace python {

template <typename T, int N, int C>
struct to_python_value< ndarray::TransposedEigenView<T,N,C> const & > : public detail::builtin_to_python {
    inline PyObject * operator()(ndarray::TransposedEigenView<T,N,C> const & x) const {
        to_python_value< ndarray::Array<T,N,C> const &> array_to_python;
        try {
            numpy::ndarray array(python::detail::new_reference(array_to_python(x.getArray())));
            numpy::matrix matrix(array, array.get_dtype(), false);
            matrix = matrix.transpose();
            Py_INCREF(matrix.ptr());
            return matrix.ptr();
        } catch (error_already_set & err) {
            handle_exception();
            return NULL;
        }
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<numpy::matrix>::get_pytype();
    }
};

template <typename T, int N, int C>
struct to_python_value< ndarray::TransposedEigenView<T,N,C> & > : public detail::builtin_to_python {
    inline PyObject * operator()(ndarray::TransposedEigenView<T,N,C> & x) const {
        return to_python_value< ndarray::TransposedEigenView<T,N,C> const & >()(x);
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<numpy::matrix>::get_pytype();
    }
};

namespace converter {

template <typename T, int N, int C>
struct arg_to_python< ndarray::TransposedEigenView<T,N,C> > : public handle<> {
    inline arg_to_python(ndarray::TransposedEigenView<T,N,C> const & v) :
        handle<>(to_python_value<ndarray::TransposedEigenView<T,N,C> const &>()()) {}
};

} // namespace boost::python::converter
}} // namespace boost::python

#endif // !BOOST_PYTHON_NDARRAY_TRANSPOSEDEIGENVIEW_HPP_INCLUDED
