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
#ifndef BOOST_PYTHON_EIGEN_SPECIALIZATIONS_HPP_INCLUDED
#define BOOST_PYTHON_EIGEN_SPECIALIZATIONS_HPP_INCLUDED

#include <boost/python/eigen/PythonMatrix.hpp>
#include <boost/python/eigen/eigen_to_python.hpp>
#include <boost/python/eigen/return_internal_matrix.hpp>

#define BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(ARGS, TYPE, CV)              \
    namespace boost { namespace python {                                \
    template < ARGS >                                                   \
    struct to_python_value< TYPE CV > : public detail::builtin_to_python { \
        inline PyObject * operator()(TYPE CV x) const {                 \
            try {                                                       \
                object r = eigen_to_python< TYPE >::to_python_value(x); \
                Py_INCREF(r.ptr());                                     \
                return r.ptr();                                         \
            } catch (error_already_set & exc) {                         \
                handle_exception();                                     \
                return NULL;                                            \
            }                                                           \
        }                                                               \
        inline PyTypeObject const * get_pytype() const {                \
            return converter::object_manager_traits<numpy::matrix>::get_pytype(); \
        }                                                               \
    };                                                                  \
    }}

#define BOOST_PYTHON_EIGEN_ARG_TO_PYTHON(ARGS, TYPE)                    \
    namespace boost { namespace python { namespace converter {          \
        template < ARGS >                                               \
        struct arg_to_python< TYPE > : public handle<> {                \
            inline arg_to_python(TYPE const & v) :                      \
                handle<>(python::to_python_value<TYPE const &>()(v)) {} \
        };                                                              \
    }}}

#define BOOST_PYTHON_EIGEN_FROM_PYTHON_AUTO(ARGS, TYPE)                 \
    namespace boost { namespace python { namespace converter {          \
    template < ARGS >                                                   \
    struct arg_rvalue_from_python< TYPE const & > {                     \
        typedef TYPE result_type;                                       \
        typedef typename TYPE::Scalar _Scalar;                          \
        typedef Eigen::PythonMatrix<_Scalar, TYPE::RowsAtCompileTime,TYPE::ColsAtCompileTime> PythonMatrix; \
        arg_rvalue_from_python(PyObject * p) : _p(borrowed(p)) {}       \
        bool convertible() const {                                      \
            try {                                                       \
                object arg(_p);                                         \
                python::numpy::matrix m(arg, python::numpy::dtype::get_builtin<_Scalar>()); \
                PythonMatrix pm(m);                                     \
                _p = handle<>(borrowed(m.ptr()));                       \
            } catch (error_already_set & exc) {                         \
                PyErr_Clear();                                          \
                return false;                                           \
            }                                                           \
            return true;                                                \
        }                                                               \
        result_type operator()() const {                                \
            boost::python::numpy::matrix m(python::detail::borrowed_reference(_p.get())); \
            PythonMatrix pm(m);                                         \
            return result_type(pm);                                     \
        }                                                               \
    private:                                                            \
        mutable handle<> _p;                                            \
    };                                                                  \
    }}}

#define EIGEN_MATRIX_ARGS typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols
#define EIGEN_MATRIX_TYPE Eigen::Matrix<Scalar,Rows,Cols,Options,MaxRows,MaxCols>
BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(EIGEN_MATRIX_ARGS, EIGEN_MATRIX_TYPE, &)
BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(EIGEN_MATRIX_ARGS, EIGEN_MATRIX_TYPE, const &)
BOOST_PYTHON_EIGEN_ARG_TO_PYTHON(EIGEN_MATRIX_ARGS, EIGEN_MATRIX_TYPE)
BOOST_PYTHON_EIGEN_FROM_PYTHON_AUTO(EIGEN_MATRIX_ARGS, EIGEN_MATRIX_TYPE)
#undef EIGEN_MATRIX_ARGS
#undef EIGEN_MATRIX_TYPE

#define EIGEN_MAP_ARGS typename MatrixType, int PacketAccess
#define EIGEN_MAP_TYPE Eigen::Map<MatrixType,PacketAccess>
BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(EIGEN_MAP_ARGS, EIGEN_MAP_TYPE, &)
BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(EIGEN_MAP_ARGS, EIGEN_MAP_TYPE, const &)
BOOST_PYTHON_EIGEN_ARG_TO_PYTHON(EIGEN_MAP_ARGS, EIGEN_MAP_TYPE)
#undef EIGEN_MAP_ARGS
#undef EIGEN_MAP_TYPE

#define EIGEN_BLOCK_ARGS typename MatrixType, int BlockRows, int BlockCols, int PacketAccess
#define EIGEN_BLOCK_TYPE Eigen::Block<MatrixType,BlockRows,BlockCols,PacketAccess,Eigen::HasDirectAccess>
BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(EIGEN_BLOCK_ARGS, EIGEN_BLOCK_TYPE, &)
BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(EIGEN_BLOCK_ARGS, EIGEN_BLOCK_TYPE, const &)
BOOST_PYTHON_EIGEN_ARG_TO_PYTHON(EIGEN_BLOCK_ARGS, EIGEN_BLOCK_TYPE)
#undef EIGEN_BLOCK_ARGS
#undef EIGEN_BLOCK_TYPE

#define EIGEN_TRANSPOSE_ARGS typename MatrixType
#define EIGEN_TRANSPOSE_TYPE Eigen::Transpose< MatrixType >
BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(EIGEN_TRANSPOSE_ARGS, EIGEN_TRANSPOSE_TYPE, &)
BOOST_PYTHON_EIGEN_TO_PYTHON_VALUE(EIGEN_TRANSPOSE_ARGS, EIGEN_TRANSPOSE_TYPE, const &)
BOOST_PYTHON_EIGEN_ARG_TO_PYTHON(EIGEN_TRANSPOSE_ARGS, EIGEN_TRANSPOSE_TYPE)
#undef EIGEN_TRANSPOSE_ARGS
#undef EIGEN_TRANSPOSE_TYPE

#endif // !BOOST_PYTHON_EIGEN_SPECIALIZATIONS_HPP_INCLUDED
