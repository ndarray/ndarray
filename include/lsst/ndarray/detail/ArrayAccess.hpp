#ifndef LSST_NDARRAY_DETAIL_ArrayAccess_hpp_INCLUDED
#define LSST_NDARRAY_DETAIL_ArrayAccess_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/detail/ArrayAccess.hpp
 *
 *  @brief Definitions for ArrayAccess
 */

#include "lsst/ndarray/ExpressionTraits.hpp"

namespace lsst { namespace ndarray {
namespace detail {

template <typename Array_>
struct ArrayAccess {
    typedef typename ExpressionTraits< Array_ >::Element Element;
    typedef typename ExpressionTraits< Array_ >::Core Core;
    typedef typename ExpressionTraits< Array_ >::CorePtr CorePtr;

    static CorePtr const & getCore(Array_ const & array) {
        return array._core;
    }

    static Array_ construct(Element * data, CorePtr const & core) {
        return Array_(data, core);
    }

};

} // namespace lsst::ndarray::detail
}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_DETAIL_ArrayAccess_hpp_INCLUDED
