#ifndef LSST_NDARRAY_PYTHON_numpy_hpp_INCLUDED
#define LSST_NDARRAY_PYTHON_numpy_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/python/numpy.hpp
 *  @brief Python C-API conversions between ndarray and numpy.
 */

#include "lsst/ndarray.hpp"
#include "lsst/ndarray/python/PyConverter.hpp"

namespace lsst { namespace ndarray {
namespace detail {

/** 
 *  @internal @ingroup PythonInternalGroup
 *  @brief Traits class that specifies Numpy typecodes for numeric types. 
 */
template <typename T> struct NumpyTraits { 
    static int getCode(); 
};

/// \cond SPECIALIZATIONS
    
template <> struct NumpyTraits<bool> {
    static int getCode() {
	if (sizeof(bool)==sizeof(npy_bool)) return NPY_BOOL;
	if (sizeof(bool)==1) return NPY_UBYTE;
	if (sizeof(bool)==2 && sizeof(short)==2) return NPY_USHORT;
	if (sizeof(bool)==4 && sizeof(int)==4) return NPY_UINT;
	assert(false);
	return 0;
    }
};

template <> struct NumpyTraits<npy_ubyte> { static int getCode() { return NPY_UBYTE; } };
template <> struct NumpyTraits<npy_byte> { static int getCode() { return NPY_BYTE; } };
template <> struct NumpyTraits<npy_ushort> { static int getCode() { return NPY_USHORT; } };
template <> struct NumpyTraits<npy_short> { static int getCode() { return NPY_SHORT; } };
template <> struct NumpyTraits<npy_uint> { static int getCode() { return NPY_UINT; } };
template <> struct NumpyTraits<npy_int> { static int getCode() { return NPY_INT; } };
template <> struct NumpyTraits<npy_ulong> { static int getCode() { return NPY_ULONG; } };
template <> struct NumpyTraits<npy_long> { static int getCode() { return NPY_LONG; } };
template <> struct NumpyTraits<npy_ulonglong> { static int getCode() { return NPY_ULONGLONG; } };
template <> struct NumpyTraits<npy_longlong> { static int getCode() { return NPY_LONGLONG; } };
template <> struct NumpyTraits<npy_float> { static int getCode() { return NPY_FLOAT; } };
template <> struct NumpyTraits<npy_double> { static int getCode() { return NPY_DOUBLE; } };
template <> struct NumpyTraits<npy_longdouble> { static int getCode() { return NPY_LONGDOUBLE; } };
template <> struct NumpyTraits<npy_cfloat> { static int getCode() { return NPY_CFLOAT; } };
template <> struct NumpyTraits<npy_cdouble> { static int getCode() { return NPY_CDOUBLE; } };
template <> struct NumpyTraits<npy_clongdouble> { static int getCode() { return NPY_CLONGDOUBLE; } };

template <> struct NumpyTraits<std::complex<float> > { 
    static int getCode() { assert(sizeof(std::complex<float>)==sizeof(npy_cfloat)); return NPY_CFLOAT; } 
};

template <> struct NumpyTraits<std::complex<double> > { 
    static int getCode() { assert(sizeof(std::complex<double>)==sizeof(npy_cdouble)); return NPY_CDOUBLE; } 
};

template <> struct NumpyTraits<std::complex<long double> > { 
    static int getCode() { 
	assert(sizeof(std::complex<long double>)==sizeof(npy_clongdouble)); 
	return NPY_CLONGDOUBLE; 
    }
};

/// \endcond

/** 
 *  @internal @ingroup PythonInternalGroup
 *  @brief A shared_ptr deleter that owns a reference to a Python object.
 *
 *  @todo Analyze possible reference cycles and figure out how to deal with them.
 */
class PythonDeleter {
    PyPtr _p;
public:

    template <typename T> void operator()(T * r) { _p.reset(); }

    // steals a reference
    explicit PythonDeleter(PyPtr const & p) : _p(p) {}

};

/** 
 *  @internal @ingroup PythonInternalGroup
 *  @brief A destructor for a Python CObject that owns a shared_ptr.
 */
template <typename T>
static void destroyCObject(void * p) {
    boost::shared_ptr<T> * b = reinterpret_cast<boost::shared_ptr<T>*>(p);
    delete b;
}

} // namespace lsst::ndarray::detail

/**
 *  @ingroup PythonGroup
 *  @brief A traits class providing Python conversion functions for Array.
 *
 *  This specialization, for Array, adds addititional optional arguments
 *  to the toPython() conversion member function.
 */
template <typename T, int N, int C>
struct PyConverter< Array<T,N,C> > : public detail::PyConverterBase< Array<T,N,C> > {
    typedef typename Array<T,N,C>::Element Element;
    typedef typename boost::remove_const<Element>::type NonConst;
    typedef typename boost::shared_ptr<Element> Owner;

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
        int flags = NPY_ALIGNED;
        bool writeable = !boost::is_const<Element>::value;
        if (writeable) flags |= (NPY_WRITEABLE | NPY_UPDATEIFCOPY);
        PyPtr array(PyArray_FROMANY(p.get(),detail::NumpyTraits<NonConst>::getCode(),N,N,flags),false);
        if (!array) return false;
        p = array;
        return true;
    }

    /**
     *  @brief Complete a Python to C++ conversion begun with fromPythonStage1().
     * 
     *  The copy will be shallow if possible and deep if necessary to meet the data type
     *  and contiguousness requirements.  The NPY_UPDATEIFCOPY flag will set on the numpy
     *  array if a deep copy is necessary and the Array data type is non-const; this will
     *  cause the copy to be written back to the original when it is deallocated.
     *
     *  The output Array's shared_ptr owner attribute will own a reference to the numpy
     *  array that ultimately owns the data (either the original or the copy).
     *
     *  \return true on success, false on failure (with a Python exception set).
     */
    static bool fromPythonStage2(
        PyPtr const & input,  ///< Result of fromPythonStage1().
        Array<T,N,C> & output ///< Reference to existing output C++ object.
    ) {
        int flags = NPY_ALIGNED;
        bool writeable = !boost::is_const<Element>::value;
        if (writeable) flags |= (NPY_WRITEABLE | NPY_UPDATEIFCOPY);
        LSST_NDARRAY_ASSERT(input);
        LSST_NDARRAY_ASSERT(PyArray_Check(input.get()));
        LSST_NDARRAY_ASSERT(reinterpret_cast<PyArrayObject*>(input.get())->nd == N);
        LSST_NDARRAY_ASSERT(reinterpret_cast<PyArrayObject*>(input.get())->flags & flags);
        PyPtr array(input);
        int element_size = sizeof(Element);
        int full_size = element_size;
        for (int i=1; i<=C; ++i) { // verify that we have at least C contiguous dimensions
            int stride = PyArray_STRIDE(array.get(),N-i);
            if (stride != full_size) {
                flags |= NPY_C_CONTIGUOUS;
                array = PyPtr(
                    PyArray_FROMANY(input.get(),detail::NumpyTraits<NonConst>::getCode(),N,N,flags),
                    false
                );
                if (!array) return false;
                break;
            }
            full_size *= PyArray_DIM(array.get(),N-i);
        }
        Owner owner(reinterpret_cast<Element*>(PyArray_DATA(array.get())),detail::PythonDeleter(array));
        Vector<int,N> shape;
        Vector<int,N> strides;
        std::copy(PyArray_DIMS(array.get()),PyArray_DIMS(array.get())+N,shape.begin());
        std::copy(PyArray_STRIDES(array.get()),PyArray_STRIDES(array.get())+N,strides.begin());
        for (int i=0; i<N; ++i) strides[i] /= element_size;
        output = external(
            reinterpret_cast<Element*>(PyArray_DATA(array.get())),
            shape, strides, owner
        );
        return true;
    }

    /**
     *  @brief Create a numpy.ndarray from an ndarray::Array.
     *
     *  The Array will be shallow-copied with reference counting if either
     *  m.getOwner() is not empty or the optional owner argument is supplied;
     *  otherwise a deep copy will be made.
     *
     *  \return a new Python object, or NULL on failure (with
     *  a Python exception set).
     */
    static PyObject* toPython(
        Array<T,N,C> const & m, ///< The input Array object.
        PyObject* owner=NULL    /**< A Python object that owns the memory in the Array.
                                 *   If NULL, one will be constructed from m.getOwner(). */
    ) {
        int flags = NPY_ALIGNED;
        if (C==N) flags |= NPY_C_CONTIGUOUS;
        bool writeable = !boost::is_const<Element>::value;
        if (writeable) flags |= NPY_WRITEABLE;
        npy_intp shape[N];
        npy_intp strides[N];
        Vector<int,N> mshape = m.getShape();
        Vector<int,N> mstrides = m.getStrides();
        std::copy(mshape.begin(),mshape.end(),shape);
        for (int i=0; i<N; ++i) strides[i] = mstrides[i]*sizeof(Element);
        PyPtr array(PyArray_New(&PyArray_Type,N,shape,detail::NumpyTraits<NonConst>::getCode(),strides,
                                const_cast<NonConst*>(m.getData()),
                                sizeof(Element),flags,NULL),
                    false);
        if (!array) return NULL;
        if (!m.getOwner() && owner == NULL) {
            flags = NPY_CARRAY_RO | NPY_ENSURECOPY | NPY_C_CONTIGUOUS;
            if (writeable) flags |= NPY_WRITEABLE;
            PyPtr r = PyArray_FROM_OF(array.get(),flags);
            if (!r) return NULL;
            array.swap(r);
        } else {
            if (owner != NULL) {
                Py_INCREF(owner);
            } else {
                owner = PyCObject_FromVoidPtr(new Owner(m.getOwner()),detail::destroyCObject<Element>);
            }
            reinterpret_cast<PyArrayObject*>(array.get())->base = owner;
        }
        Py_INCREF(array.get());
        return PyArray_Return(reinterpret_cast<PyArrayObject*>(array.get()));
    }

    static PyTypeObject const * getPyType() { return &PyArray_Type; }
};

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_PYTHON_numpy_hpp_INCLUDED
