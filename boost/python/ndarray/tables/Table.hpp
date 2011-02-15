#ifndef BOOST_PYTHON_NDARRAY_TABLES_TABLE_HPP_INCLUDED
#define BOOST_PYTHON_NDARRAY_TABLES_TABLE_HPP_INCLUDED

#include <boost/python/numpy/numpy.hpp>
#include <boost/python/ndarray.hpp>
#include <boost/python/ndarray/tables/Layout.hpp>
#include <ndarray/tables.hpp>

namespace boost { namespace python {

template <typename T>
struct to_python_value< ndarray::tables::Table<T> const & > : public detail::builtin_to_python {
    inline PyObject * operator()(ndarray::tables::Table<T> const & x) const {
        numpy::dtype dtype = ndarray::tables::makeNumpyType(x.getLayout());
        object owner = makePyObject(x.getManager());
        std::vector<Py_intptr_t> shape_char(1, x.getSize());
        std::vector<Py_intptr_t> strides_char(1, x.getStride());
        numpy::ndarray array = numpy::from_data(x.getRaw(), dtype, shape_char, strides_char, owner);
        Py_INCREF(array.ptr());
        return array.ptr();
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<numpy::ndarray>::get_pytype();
    }
};

template <typename T>
struct to_python_value< ndarray::tables::Table<T> & > : public detail::builtin_to_python {
    inline PyObject * operator()(ndarray::tables::Table<T> & x) const {
        return to_python_value< ndarray::tables::Table<T> const & >()(x);
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<numpy::ndarray>::get_pytype();
    }
};

namespace converter {

template <typename T>
struct arg_to_python< ndarray::tables::Table<T> > : public handle<> {
    inline arg_to_python(ndarray::tables::Table<T> const & v) :
        handle<>(to_python_value<ndarray::tables::Table<T> const &>()(v)) {}
};

template <typename T>
struct arg_rvalue_from_python< ndarray::tables::Table<T> const & > {
    typedef ndarray::tables::Table<T> result_type;
    typedef typename ndarray::tables::detail::TraitsAccess<T>::Raw Raw;
    typedef typename ndarray::tables::detail::TraitsAccess<T>::Layout_ Layout_;

    static numpy::ndarray::bitflag const flags = 
        numpy::ndarray::bitflag(
            (boost::is_const<T>::value ? int(numpy::ndarray::WRITEABLE) : 0) |
            int(numpy::ndarray::ALIGNED)
        );

    arg_rvalue_from_python(PyObject * p) : arg(python::detail::borrowed_reference(p)) {}

    bool convertible() const {
        try {
            numpy::ndarray array = extract<numpy::ndarray>(arg);
            if (array.get_nd() != 1) return false;
        } catch (error_already_set) {
            handle_exception();
            PyErr_Clear();
            return false;
        }
        return true;
    }

    result_type operator()() const {
        numpy::ndarray array = extract<numpy::ndarray>(arg);
        numpy::dtype dtype = array.get_dtype();
        Layout_ layout;
        ndarray::tables::fillLayout(dtype, layout);
        object obj_owner = array.get_base();
        if (obj_owner == object()) {
            obj_owner = array;
        }
        int size = array.shape(0);
        int stride = array.strides(0);
        ndarray::tables::Table<T> r = ndarray::tables::Table<T>::external(
            reinterpret_cast<Raw*>(array.get_data()), size, stride, layout, obj_owner
        );
        return r;
    }

    mutable object arg;
};

template <typename T>
struct extract_rvalue< ndarray::tables::Table<T> > : private noncopyable {
    typedef ndarray::tables::Table<T> result_type;

    extract_rvalue(PyObject * x) : m_converter(x) {}

    bool check() const { return m_converter.convertible(); }
    
    result_type operator()() const { return m_converter(); }

private:
    arg_rvalue_from_python< result_type const & > m_converter;
};

} // namespace converter

namespace numpy {

template <typename T>
numpy::ndarray array(::ndarray::tables::Table<T> const & arg) {
    to_python_value< ::ndarray::tables::Table<T> const & > converter;
    numpy::ndarray result(python::detail::new_reference(converter(arg)));
    return result;
}

} // namespace numpy


}} // namespace boost::python

#endif // !BOOST_PYTHON_NDARRAY_TABLES_TABLE_HPP_INCLUDED
