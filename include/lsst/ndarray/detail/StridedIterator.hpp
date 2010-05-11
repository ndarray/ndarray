#ifndef LSST_NDARRAY_DETAIL_StridedIterator_hpp_INCLUDED
#define LSST_NDARRAY_DETAIL_StridedIterator_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/detail/StridedIterator.hpp
 *
 *  @brief Definition of StridedIterator.
 */

#include "lsst/ndarray_fwd.hpp"
#include <boost/iterator/iterator_facade.hpp>

namespace lsst { namespace ndarray {
namespace detail {

/**
 *  @internal @brief Strided iterator for noncontiguous 1D arrays.
 *
 *  @ingroup InternalGroup
 */
template <typename T>
class StridedIterator : public boost::iterator_facade<
    StridedIterator<T>, 
    T, boost::random_access_traversal_tag
    >
{
public:
    typedef T Value;
    typedef T & Reference;
    
    StridedIterator() : _data(0), _stride(0) {}

    StridedIterator(T * data, int stride) : _data(data), _stride(stride) {}

    StridedIterator(StridedIterator const & other) : _data(other._data), _stride(other._stride) {}

    template <typename U>
    StridedIterator(StridedIterator<U> const & other) : _data(other._data), _stride(other._stride) {
        BOOST_STATIC_ASSERT((boost::is_convertible<U*,T*>::value));
    }

    StridedIterator & operator=(StridedIterator const & other) {
        if (&other != this) {
            _data = other._data;
            _stride = other._stride;
        }
        return *this;
    }

    template <typename U>
    StridedIterator & operator=(StridedIterator<U> const & other) {
        BOOST_STATIC_ASSERT((boost::is_convertible<U*,T*>::value));
        _data = other._data;
        _stride = other._stride;
        return *this;
    }

private:

    friend class boost::iterator_core_access;

    template <typename OtherT> friend class StridedIterator;

    Reference dereference() const { return *_data; }

    void increment() { _data += _stride; }
    void decrement() { _data -= _stride; }
    void advance(int n) { _data += _stride * n; }

    template <typename U>
    int distance_to(StridedIterator<U> const & other) const {
        return std::distance(_data, other._data) / _stride; 
    }

    template <typename U>
    bool equal(StridedIterator<U> const & other) const {
        return _data == other._data;
    }

    T * _data;
    int _stride;

};

} // namespace lsst::ndarray::detail
}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_DETAIL_StridedIterator_hpp_INCLUDED
