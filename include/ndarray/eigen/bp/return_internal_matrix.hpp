#ifndef BOOST_PYTHON_EIGEN_RETURN_INTERNAL_MATRIX_HPP_INCLUDED
#define BOOST_PYTHON_EIGEN_RETURN_INTERNAL_MATRIX_HPP_INCLUDED

#include <boost/python/numpy.hpp>
#include <Eigen/Core>

namespace boost { namespace python {

template <std::size_t owner_arg = 1, typename Base = default_call_policies>
struct return_internal_matrix : public Base {

    /**
     *  \brief A model of ResultConverter that creates a numpy.matrix view into an Eigen object.
     *
     *  Note that the returned view dooes not own its own data, and doesn't (yet) own a reference
     *  to the object that does.  For this reason, this result converter should ONLY be used by
     *  return_internal_matrix or with another model of CallPolicies with a postcall operations
     *  that installs a reference to the owning object.
     */
    template <typename Matrix, bool writeable>
    struct make_shallow_matrix {

        inline bool convertible() const {
            return eigen_to_python<Matrix>::shallow_possible;
        }

        inline PyObject * operator()(Matrix const & x) const {
            try {
                object r = eigen_to_python< Matrix >::to_python_shallow(x, object(), writeable);
                Py_INCREF(r.ptr());
                return r.ptr();
            } catch (error_already_set & exc) {
                handle_exception();
                return NULL;
            }
        }

        inline PyTypeObject const * get_pytype() const {
            return converter::object_manager_traits<numpy::matrix>::get_pytype();
        }

    };

    /// \brief Model of ResultConverterGenerator for make_shallow_matrix.
    struct result_converter {
        template <typename T>
        struct apply {
            typedef typename boost::remove_reference<T>::type dereferenced;
            
            typedef make_shallow_matrix<
                typename boost::remove_const<dereferenced>::type,
                boost::is_const<dereferenced>::value
                > type;
        };
    };

    template <typename ArgumentPackage>
    static PyObject * postcall(ArgumentPackage const & args_, PyObject * result) {
        std::size_t arity_ = detail::arity(args_);
        if (owner_arg > arity_ || owner_arg < 1) {
            PyErr_SetString(PyExc_IndexError,
                            "boost::python::return_internal_matrix: argument out of range.");
            return NULL;
        }
        PyObject * owner_ref = detail::get_prev<owner_arg>::execute(args_, result);
        if (owner_ref == NULL) return 0;
        result = Base::postcall(args_, result);
        if (result == NULL) return 0;
        try {
            object owner(reinterpret_cast<python::detail::borrowed_reference>(owner_ref));
            numpy::ndarray array(reinterpret_cast<python::detail::borrowed_reference>(result));
            array.set_base(owner);
            Py_INCREF(array.ptr());
            result = array.ptr();
        } catch (error_already_set & err) {
            handle_exception();
            return NULL;
        }
        return result;
    }
};

}} // namespace boost::python

#endif // !BOOST_PYTHON_EIGEN_RETURN_INTERNAL_MATRIX_HPP_INCLUDED
