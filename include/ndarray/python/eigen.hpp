#ifndef NDARRAY_PYTHON_eigen_hpp_INCLUDED
#define NDARRAY_PYTHON_eigen_hpp_INCLUDED

/**
 *  @file ndarray/python/eigen.hpp
 *  @brief Python C-API conversions for Eigen matrices.
 *
 *  \note This file is not included by the main "ndarray/python.hpp" header file.
 */

#include "ndarray/python.hpp"
#include "ndarray/eigen.hpp"

namespace ndarray {
namespace detail {

/**
 *  @internal @ingroup PythonInternalGroup
 *  @brief Implementations for PyConverter for Eigen objects.
 */
template <typename Matrix>
class EigenPyConverter : public detail::PyConverterBase<Matrix> {
    typedef typename Matrix::Scalar Scalar;
    typedef boost::mpl::int_<Matrix::RowsAtCompileTime> Rows;
    typedef boost::mpl::int_<Matrix::ColsAtCompileTime> Cols;
    typedef boost::mpl::int_<Matrix::SizeAtCompileTime> Size;
    typedef Eigen::Matrix<Scalar,Rows::value,Cols::value> TrueMatrix;

    static PyPtr getNumpyMatrixType() {
        PyPtr numpyModule(PyImport_ImportModule("numpy"),false);
        if (numpyModule) {
            return PyPtr(PyObject_GetAttrString(numpyModule.get(),"matrix"),false);
        }
        return PyPtr();
    }

    static PyPtr makeNumpyMatrix(PyPtr const & array) {
        PyPtr matrixType(getNumpyMatrixType());
        if (!matrixType) return PyPtr();
        PyPtr args(PyTuple_Pack(1,array.get()),false);
        PyPtr kwds(PyDict_New(),false);
        if (PyDict_SetItemString(kwds.get(),"copy",Py_False) != 0) return PyPtr();
        return PyPtr(PyObject_Call(matrixType.get(),args.get(),kwds.get()),false);
    }

public:

    /** 
     *  @brief Convert a C++ object to a new Python object.
     *
     *  \return A new Python object, or NULL on failure (with
     *  a Python exception set).
     */
    static PyObject * toPython(
        Matrix const & input, ///< Input C++ object.
        PyObject * owner = NULL,
        bool writeable = true,
        bool squeeze = false
    ) {
        Array<Scalar,2> array(ndarray::viewMatrixAsArray(const_cast<Matrix&>(input)));
        PyPtr pyArray;
        if (writeable) {
            pyArray = PyPtr(PyConverter< Array<Scalar,2> >::toPython(array,owner),false);
        } else {
            pyArray = PyPtr(PyConverter< Array<Scalar const,2> >::toPython(array,owner),false);
        }
        PyPtr r;
        if (squeeze) {
            r.reset(PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(pyArray.get())));
        } else {
            r = makeNumpyMatrix(pyArray);
        }
        if (!r) return NULL;
        Py_XINCREF(r.get());
        return r.get();

    }

    /**
     *  @brief Return the Python TypeObject that corresponds to
     *  the object the toPython() function returns.
     */
    static PyTypeObject const * getPyType() {
        return reinterpret_cast<PyTypeObject*>(getNumpyMatrixType().get());
    }

    /**
     *  @brief Check if a Python object is convertible to T
     *  and optionally begin the conversion by replacing the
     *  input with an intermediate.
     *
     *  \return true if a conversion may be possible, and
     *  false if it is not (with a Python exception set).
     */
    static bool fromPythonStage1(
        PyPtr & p /**< On input, a Python object to be converted.
                   *   On output, a Python object to be passed to
                   *   fromPythonStage2().
                   */
    ) {
        PyPtr tmp = makeNumpyMatrix(p);
        if (!tmp) return false;
        if (Cols::value != Eigen::Dynamic && Cols::value != PyArray_DIM(tmp.get(),1)) {
            if (Cols::value == 1 && PyArray_DIM(tmp.get(),0) == 1) { 
                tmp = PyPtr(PyObject_CallMethod(tmp.get(),const_cast<char*>("transpose"),NULL),false);
            } else {
                PyErr_SetString(PyExc_ValueError,"Incorrect number of columns for matrix.");
                return false;
            }
        }
        if (Rows::value != Eigen::Dynamic && Rows::value != PyArray_DIM(tmp.get(),0)) {
            PyErr_SetString(PyExc_ValueError,"Incorrect number of rows for matrix.");
            return false;
        }
        
        p = tmp;
        return true;
    }

    /**
     *  @brief Complete a Python to C++ conversion begun
     *  with fromPythonStage1().
     *
     *  \return true if the conversion was successful,
     *  and false otherwise (with a Python exception set).
     */
    static bool fromPythonStage2(
        PyPtr const & p, ///< A Python object processed by fromPythonStage1().
        Matrix & output       ///< The output C++ object.
    ) {
        NDARRAY_ASSERT(p);
        PyPtr matrixType(getNumpyMatrixType());
        if (!matrixType) return false;
        NDARRAY_ASSERT(PyObject_IsInstance(p.get(),matrixType.get()));
        Array<Scalar,2,2> array;
        if (!PyConverter< Array<Scalar,2,2> >::fromPythonStage2(p,array)) return false;
        int rows = array.template getSize<0>();
        int cols = array.template getSize<1>();
        Eigen::Block<ndarray::EigenView<Scalar,2,2>,Rows::value,Cols::value> block(
            ndarray::viewAsEigen(array), 0, 0, rows, cols
        );
        output = block;
        return true;
    }

};

} // namespace ndarray::detail

/**
 *  @ingroup PythonGroup
 *  @brief Specialization of PyConverter for Eigen::Matrix.
 */
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct PyConverter< Eigen::Matrix<Scalar,Rows,Cols,Options,MaxRows,MaxCols> >
    : public detail::EigenPyConverter< Eigen::Matrix<Scalar,Rows,Cols,Options,MaxRows,MaxCols> > 
{};

/**
 *  @ingroup PythonGroup
 *  @brief Specialization of PyConverter for Eigen::Map.
 */
template <typename MatrixType, int PacketAccess>
struct PyConverter< Eigen::Map<MatrixType,PacketAccess> >
    : public detail::EigenPyConverter< Eigen::Map<MatrixType,PacketAccess> > 
{};

/**
 *  @ingroup PythonGroup
 *  @brief Specialization of PyConverter for Eigen::Block.
 */
template <typename MatrixType, int BlockRows, int BlockCols, int PacketAccess>
struct PyConverter< Eigen::Block<MatrixType,BlockRows,BlockCols,PacketAccess,Eigen::HasDirectAccess> >
    : public detail::EigenPyConverter< 
        Eigen::Block<MatrixType,BlockRows,BlockCols,PacketAccess,Eigen::HasDirectAccess> 
    > 
{};

/**
 *  @ingroup PythonGroup
 *  @brief Specialization of PyConverter for Eigen::Transpose.
 */
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct PyConverter< Eigen::Transpose< Eigen::Matrix<Scalar,Rows,Cols,Options,MaxRows,MaxCols> > >
    : public detail::EigenPyConverter< 
          Eigen::Transpose< Eigen::Matrix<Scalar,Rows,Cols,Options,MaxRows,MaxCols> > 
      > 
{};

/**
 *  @ingroup PythonGroup
 *  @brief Specialization of PyConverter for ndarray::EigenView
 */
template <typename T, int N, int C>
struct PyConverter< EigenView<T,N,C> > : public detail::EigenPyConverter< EigenView<T,N,C> > {};

/**
 *  @ingroup PythonGroup
 *  @brief Specialization of PyConverter for ndarray::TransposedEigenView
 */
template <typename T, int N, int C>
struct PyConverter< TransposedEigenView<T,N,C> > : 
    public detail::EigenPyConverter< TransposedEigenView<T,N,C> > {};

} // namespace ndarray

#endif // !NDARRAY_PYTHON_eigen_hpp_INCLUDED
