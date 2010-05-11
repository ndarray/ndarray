#ifndef LSST_NDARRAY_ArrayBase_hpp_INCLUDED
#define LSST_NDARRAY_ArrayBase_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/ArrayBase.hpp
 *
 *  @brief Definitions for ArrayBase.
 */


#include <boost/iterator/counting_iterator.hpp>

#include "lsst/ndarray/ExpressionBase.hpp"
#include "lsst/ndarray/Vector.hpp"
#include "lsst/ndarray/detail/Core.hpp"
#include "lsst/ndarray/detail/NestedIterator.hpp"
#include "lsst/ndarray/detail/StridedIterator.hpp"
#include "lsst/ndarray/detail/ArrayAccess.hpp"
#include "lsst/ndarray/ArrayTraits.hpp"
#include "lsst/ndarray/views.hpp"

namespace lsst { namespace ndarray {

/**
 *  @class ArrayBase
 *  @brief CRTP implementation for Array and ArrayRef.
 *
 *  @ingroup MainGroup
 *
 *  Implements member functions that need specialization for 1D arrays.
 */
template <typename Derived>
class ArrayBase : public ExpressionBase<Derived> {
    typedef ExpressionTraits<Derived> Traits;
    typedef typename Traits::Core Core;
    typedef typename Traits::CorePtr CorePtr;
public:
    /// @brief Data type of array elements.
    typedef typename Traits::Element Element;
    /// @brief Nested array or element iterator.
    typedef typename Traits::Iterator Iterator;
    /// @brief Nested array or element reference.
    typedef typename Traits::Reference Reference;
    /// @brief Nested array or element value type.
    typedef typename Traits::Value Value;
    /// @brief Number of dimensions (boost::mpl::int_).
    typedef typename Traits::ND ND;
    /// @brief Number of guaranteed row-major contiguous dimensions, counted from the end (boost::mpl::int_).
    typedef typename Traits::RMC RMC;
    /// @brief Vector type for N-dimensional indices.
    typedef Vector<int,ND::value> Index;
    /// @brief Shared pointer that manages memory.
    typedef boost::shared_ptr<Element> Owner;
    /// @brief ArrayRef to a noncontiguous array; the result of a call to transpose().
    typedef ArrayRef<Element,ND::value,0> Transpose;
    /// @brief The corresponding Array type.
    typedef Array<Element,ND::value,RMC::value> Shallow;
    /// @brief The corresponding ArrayRef type.
    typedef ArrayRef<Element,ND::value,RMC::value> Deep;

    /// @brief Return a single subarray.
    Reference operator[](int n) const {
        return Traits::makeReference(this->_data + n * this->template getStride<0>(), this->_core);
    }

    /// @brief Return a single element from the array.
    Element & operator[](Index const & i) const {
        return *(this->_data + this->_core->template computeOffset(i));
    }

    /// @brief Return an Iterator to the beginning of the array.
    Iterator begin() const {
        return Traits::makeIterator(this->_data, this->_core, this->template getStride<0>());
    }

    /// @brief Return an Iterator to one past the end of the array.
    Iterator end() const {
        return Traits::makeIterator(
            this->_data + this->template getSize<0>() * this->template getStride<0>(), 
            this->_core,
            this->template getStride<0>()
        );
    }

    /// @brief Return a pointer to the first element of the array.
    Element * getData() const { return _data; }
    
    /// @brief Return the shared_ptr that manages the lifetime of the array data.
    Owner getOwner() const { return this->_core->getOwner(); }

    /// @brief Return the size of a specific dimension.
    template <int P> int getSize() const {
        return detail::getDimension<P>(*this->_core).getSize();
    }

    /// @brief Return the stride in a specific dimension.
    template <int P> int getStride() const {
        return detail::getDimension<P>(*this->_core).getStride();
    }

    /// @brief Return a Vector of the sizes of all dimensions.
    Index getShape() const { Index r; this->_core->fillShape(r); return r; }

    /// @brief Return a Vector of the strides of all dimensions.
    Index getStrides() const { Index r; this->_core->fillStrides(r); return r; }

    /// @brief Return the total number of elements in the array.
    int getNumElements() const { return this->_core->getNumElements(); }

    /// @brief Return a view of the array with the order of the dimensions reversed.
    Transpose transpose() const {
        Index order;
        std::copy(
            boost::counting_iterator<int>(0),
            boost::counting_iterator<int>(ND::value),
            order.rbegin()
        );
        return transpose(order);
    }

    /// @brief Return a view of the array with the dimensions permuted.
    Transpose transpose(Index const & order) const {
        Index newShape;
        Index newStrides;
        Index oldShape = getShape();
        Index oldStrides = getStrides();
        for (int n=0; n < ND::value; ++n) {
            newShape[n] = oldShape[order[n]];
            newStrides[n] = oldStrides[order[n]];
        }
        return Transpose(getData(), Core::create(newShape,newStrides,getOwner()));
    }

    /// @brief Return a Array view to this.
    Shallow const shallow() const { return Shallow(this->getSelf()); }

    /// @brief Return an ArrayRef view to this.
    Deep const deep() const { return Deep(this->getSelf()); }

    /// @brief A template metafunction class to determine the result of a view indexing operation.
    template <
        typename View_, 
        typename Seq_ = typename detail::ViewNormalizer<ND::value,typename View_::Sequence>::Output
        >
    struct ResultOf {
        typedef Element Element_;
        typedef typename detail::ViewTraits<ND::value,RMC::value,Seq_>::ND ND_;
        typedef typename detail::ViewTraits<ND::value,RMC::value,Seq_>::RMC RMC_;
        typedef ArrayRef<Element_,ND_::value,RMC_::value> Type;
        typedef Array<Element_,ND_::value,RMC_::value> Value;
    };

    /// @brief Return a general view into this array (see \ref tutorial).
    template <typename View_>
    typename ResultOf<View_>::Type
    operator[](View_ const & def) const {
        return detail::buildView(this->getSelf(), def._seq);
    }

protected:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;
    template <typename T_, int N_, int C_> friend struct ArrayTraits;
    template <typename T_, int N_, int C_> friend class detail::NestedIterator;
    template <typename Derived_> friend class ArrayBase;
    template <typename Array_> friend struct detail::ArrayAccess;

    Element * _data;
    CorePtr _core;

    void operator=(ArrayBase const & other) {
        _data = other._data;
        _core = other._core;
    }

    ArrayBase(Element * data, CorePtr const & core) : _data(data), _core(core) {}
};

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_ArrayBase_hpp_INCLUDED
