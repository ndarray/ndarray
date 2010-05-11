#ifndef LSST_NDARRAY_ArrayTraits_hpp_INCLUDED
#define LSST_NDARRAY_ArrayTraits_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/ArrayTraits.hpp
 *
 *  @brief Traits for Array.
 */

#include "lsst/ndarray_fwd.hpp"
#include "lsst/ndarray/ExpressionTraits.hpp"
#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

namespace lsst { namespace ndarray {

/**
 *  @brief Dimension-specialized traits shared by Array and ArrayRef.
 *
 *  @ingroup MainGroup
 */
template <typename T, int N, int C>
struct ArrayTraits {
    typedef T Element;
    typedef boost::mpl::int_<N> ND;
    typedef boost::mpl::int_<C> RMC;
    typedef detail::NestedIterator<T,N,C> Iterator;
    typedef ArrayRef<T,N-1,(N==C)?(N-1):C> Reference;
    typedef Array<T,N-1,(N==C)?(N-1):C> Value;
    typedef typename detail::Core<typename boost::remove_const<T>::type,N> Core;
    typedef typename Core::ConstPtr CorePtr;

    static Reference makeReference(T * data, CorePtr const & core) {
        return Reference(data, core);
    }
    static Iterator makeIterator(T * data, CorePtr const & core, int stride) {
        return Iterator(Reference(data, core), stride);
    }
};

template <typename T>
struct ArrayTraits<T,1,0> {
    typedef T Element;
    typedef boost::mpl::int_<1> ND;
    typedef boost::mpl::int_<0> RMC;
    typedef detail::StridedIterator<T> Iterator;
    typedef T & Reference;
    typedef T Value;
    typedef typename detail::Core<typename boost::remove_const<T>::type,1> Core;
    typedef typename Core::ConstPtr CorePtr;

    static Reference makeReference(T * data, CorePtr const & core) {
        return *data;
    }
    static Iterator makeIterator(T * data, CorePtr const & core, int stride) {
        return Iterator(data, stride);
    }
};

template <typename T>
struct ArrayTraits<T,1,1> {
    typedef T Element;
    typedef boost::mpl::int_<1> ND;
    typedef boost::mpl::int_<1> RMC;
    typedef T * Iterator;
    typedef T & Reference;
    typedef T Value;
    typedef typename detail::Core<typename boost::remove_const<T>::type,1> Core;
    typedef typename Core::ConstPtr CorePtr;

    static Reference makeReference(T * data, CorePtr const & core) {
        return *data;
    }
    static Iterator makeIterator(T * data, CorePtr const & core, int stride) {
        return data;
    }
};

template <typename T, int N, int C>
struct ExpressionTraits< Array<T,N,C> > : public ArrayTraits<T,N,C> {
    typedef Array<T,N,C> Self;
    typedef boost::mpl::false_ IsScalar;
};

template <typename T, int N, int C>
struct ExpressionTraits< ArrayRef<T,N,C> > : public ArrayTraits<T,N,C> {
    typedef ArrayRef<T,N,C> Self;
    typedef boost::mpl::false_ IsScalar;
};

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_ArrayTraits_hpp_INCLUDED
