#ifndef NDARRAY_DETAIL_ArrayAccess_hpp_INCLUDED
#define NDARRAY_DETAIL_ArrayAccess_hpp_INCLUDED

/** 
 *  @file ndarray/detail/ArrayAccess.hpp
 *
 *  @brief Definitions for ArrayAccess
 */

#include "ndarray/ExpressionTraits.hpp"

namespace ndarray {
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

} // namespace ndarray::detail
} // namespace ndarray

#endif // !NDARRAY_DETAIL_ArrayAccess_hpp_INCLUDED
