#ifndef LSST_NDARRAY_casts_hpp_INCLUDED
#define LSST_NDARRAY_casts_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/casts.hpp
 *
 *  @brief Specialized casts for Array.
 */

#include "lsst/ndarray/Array.hpp"
#include "lsst/ndarray/ArrayRef.hpp"
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/static_assert.hpp>

namespace lsst { namespace ndarray {

template <typename Original, typename Casted>
class ReinterpretDeleter {
    boost::shared_ptr<Original> _original;
public:
    void operator()(Casted * p) { _original.reset(); }
    explicit ReinterpretDeleter(boost::shared_ptr<Original> const & original) : _original(original) {}
};

namespace detail {

template <typename Array_>
struct ComplexExtractor {
    typedef typename ExpressionTraits<Array_>::Element Element;
    typedef typename ExpressionTraits<Array_>::ND ND;
    typedef typename boost::remove_const<Element>::type NonConst;
    typedef typename boost::is_const<Element>::type IsConst;
    BOOST_STATIC_ASSERT( boost::is_complex<NonConst>::value );
    typedef typename NonConst::value_type NonConstValue;
    typedef typename boost::add_const<NonConstValue>::type ConstValue;
    typedef typename boost::mpl::if_<IsConst, ConstValue, NonConstValue>::type Value;
    typedef ArrayRef<Value,ND::value,0> View;
    typedef Vector<int,ND::value> Index;
    typedef ReinterpretDeleter<Element,Value> Deleter;

    static inline View apply(Array_ const & array, int offset) {
        Value * p = reinterpret_cast<Value*>(array.getData()) + offset;
        boost::shared_ptr<Value> owner(p, Deleter(array.getOwner()));
        return View(Array<Value,ND::value,0>(external(p, array.getShape(), array.getStrides() * 2, owner)));
    }
};

} // namespace detail

/// @addtogroup MainGroup
/// @{

/**
 *  Convert an Array with a const data type to an array
 *  with a non-const data type.
 */
template <typename T_, typename T, int N, int C>
Array<T_,N,C>
const_array_cast(Array<T,N,C> const & array) {
    return detail::ArrayAccess< Array<T_,N,C> >::construct(
        const_cast<T_*>(array.getData()),
        detail::ArrayAccess< Array<T,N,C> >::getCore(array)
    );
}

/**
 *  Convert an Array to a type with more guaranteed
 *  row-major-contiguous dimensions with no checking.
 */
template <int C_, typename T, int N, int C>
Array<T,N,C_>
static_dimension_cast(Array<T,N,C> const & array) {
    return detail::ArrayAccess< Array<T,N,C_> >::construct(
        array.getData(),
        detail::ArrayAccess< Array<T,N,C> >::getCore(array)
    );
}

/**
 *  Convert an Array to a type with more guaranteed
 *  row-major-contiguous dimensions, if the strides
 *  of the array match the desired number of RMC
 *  dimensions.  If the cast fails, an empty Array
 *  is returned.
 */
template <int C_, typename T, int N, int C>
Array<T,N,C_>
dynamic_dimension_cast(Array<T,N,C> const & array) {
    Vector<int,N> shape = array.getShape();
    Vector<int,N> strides = array.getStrides();
    int n = 1;
    for (int i=1; i <= C_; ++i) {
        if (strides[N-i] != n) return Array<T,N,C_>();
        n *= shape[N-i];
    }
    return static_dimension_cast<C_>(array);
}

/**
 *  @brief Return an ArrayRef view into the real part of a complex array.
 */
template <typename Array_>
typename detail::ComplexExtractor<Array_>::View
getReal(Array_ const & array) {
    return detail::ComplexExtractor<Array_>::apply(array, 0);
}

/**
 *  @brief Return an ArrayRef view into the imaginary part of a complex array.
 */
template <typename Array_>
typename detail::ComplexExtractor<Array_>::View
getImag(Array_ const & array) {
    return detail::ComplexExtractor<Array_>::apply(array, 1);
}

/**
 *  @brief Create a view into an array with trailing contiguous dimensions merged.
 *
 *  The first template parameter sets the dimension of the output array and must
 *  be specified directly.  Only row-major contiguous dimensions can be flattened.
 */
template <int Nf, typename T, int N, int C>
inline typename boost::enable_if_c< ((C+Nf-N)>=1), ArrayRef<T,Nf,(C+Nf-N)> >::type
flatten(Array<T,N,C> const & input) {
    BOOST_STATIC_ASSERT(C+Nf-N >= 1);
    Vector<int,N> oldShape = input.getShape();
    Vector<int,Nf> newShape = oldShape.template first<Nf>();
    for (int n=Nf; n<N; ++n)
        newShape[Nf-1] *= oldShape[n];
    Vector<int,Nf> newStrides = input.getStrides().template first<Nf>();
    newStrides[Nf-1] = 1;
    return ArrayRef<T,Nf,(C+Nf-N)>(
        Array<T,Nf,(C+Nf-N)>(
            ndarray::external(input.getData(), newShape, newStrides, input.getOwner())
        )
    );
}

/// @}

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_casts_hpp_INCLUDED
