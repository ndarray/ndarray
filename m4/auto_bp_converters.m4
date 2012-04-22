define(`BP_AUTO_CONVERTERS',
`
namespace boost { namespace python {

template <$1>
struct to_python_value< $2 const & > : public detail::builtin_to_python {
    inline PyObject * operator()($2 const & x) const {
        object result = ndarray::ToBoostPython< $2 >::apply(x);
        Py_INCREF(result.ptr());
        return result.ptr();
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<
            typename ndarray::ToBoostPython< $2 >::result_type
        >::get_pytype();
    }
};

template <$1>
struct to_python_value< $2 & > : public detail::builtin_to_python {
    inline PyObject * operator()($2 & x) const {
        object result = ndarray::ToBoostPython< $2 >::apply(x);
        Py_INCREF(result.ptr());
        return result.ptr();
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<
            typename ndarray::ToBoostPython< $2 >::result_type
        >::get_pytype();
    }
};

namespace converter {

template <$1>
struct arg_to_python< $2 > : public handle<> {
    inline arg_to_python($2 const & v) :
        handle<>(to_python_value<$2 const &>()(v)) {}
};

template <$1>
struct arg_rvalue_from_python< $2 const & > {
    typedef $2 result_type;
    arg_rvalue_from_python(PyObject * p) :
        _converter(boost::python::object(boost::python::handle<>(boost::python::borrowed(p)))) {}
    bool convertible() const { return _converter.convertible(); }
    result_type operator()() const { return _converter(); }
private:
    mutable ndarray::FromBoostPython< $2 > _converter;
};

template <$1>
struct arg_rvalue_from_python< $2 > : public arg_rvalue_from_python< $2 const &> {
    arg_rvalue_from_python(PyObject * p) : arg_rvalue_from_python< $2 const & >(p) {}
};

template <$1>
struct arg_rvalue_from_python< $2 const > : public arg_rvalue_from_python< $2 const &> {
    arg_rvalue_from_python(PyObject * p) : arg_rvalue_from_python< $2 const & >(p) {}
};

template <$1>
struct extract_rvalue< $2 > : private noncopyable {
    typedef $2 result_type;
    extract_rvalue(PyObject * x) : m_converter(x) {}
    bool check() const { return m_converter.convertible(); }
    result_type operator()() const { return m_converter(); }
private:
    arg_rvalue_from_python< result_type const & > m_converter;
};

}}} // namespace boost::python::converter
')dnl
