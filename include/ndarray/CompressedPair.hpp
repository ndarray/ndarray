// -*- c++ -*-

/*
 * This is a copy of code from the Boost library, heavily modified to
 * rely on C++11 standard library functionality instead of C++98 Boost
 * functionality, and to simplify considerably.
 * These modifications copyright (C) 2010-2016, Jim Bosch.
 *
 * (C) Copyright Steve Cleary, Beman Dawes, Howard Hinnant & John Maddock 2000.
 * Use, modification and distribution are subject to the Boost Software License,
 * Version 1.0. (See the included boost_license_1_0.txt file or
 * http://www.boost.org/LICENSE_1_0.txt).
 *
 *  See updated version of the original (including documentation) at
 *  http://www.boost.org/libs/utility
 *
 */
#ifndef NDARRAY_CompressedPair_hpp_INCLUDED
#define NDARRAY_CompressedPair_hpp_INCLUDED

#include <algorithm>
#include <type_traits>

#include "ndarray/common.hpp"

namespace ndarray {

namespace detail {

template <typename T>
using add_ref = typename std::add_lvalue_reference<T>::type;

template <typename T>
using add_const_ref =
    typename std::add_lvalue_reference<
        typename std::add_const<T>::type
    >::type;

template <typename T1, typename T2>
using is_same_without_cv =
    typename std::is_same<
        typename std::remove_cv<T1>::type,
        typename std::remove_cv<T2>::type
    >::type;

template <typename T1, typename T2,
          bool IsSame=is_same_without_cv<T1,T2>::value,
          bool FirstEmpty=std::is_empty<T1>::value,
          bool SecondEmpty=std::is_empty<T2>::value>
struct CompressedPairSwitch;

template <typename T1, typename T2, bool IsSame>
struct CompressedPairSwitch<T1, T2, IsSame, false, false> {
    static constexpr int value = 0;
};

template <typename T1, typename T2>
struct CompressedPairSwitch<T1, T2, false, true, true> {
    static constexpr int value = 3;
};

template <typename T1, typename T2>
struct CompressedPairSwitch<T1, T2, false, true, false> {
    static constexpr int value = 1;
};

template <typename T1, typename T2>
struct CompressedPairSwitch<T1, T2, false, false, true> {
    static constexpr int value = 2;
};

// When the classes are the same and empty, prevent &first() == &second()
// by explicitly storing second.
template <typename T1, typename T2>
struct CompressedPairSwitch<T1, T2, true, true, true> {
    static constexpr int value = 2;
};

//
// can't call unqualified swap from within classname::swap
// as Koenig lookup rules will find only the classname::swap
// member function not the global declaration, so use generic_swap
// as a forwarding function (JM):
template <typename T>
inline void generic_swap(T & t1, T & t2)
{
  using std::swap;
  swap(t1, t2);
}

} // detail

template <typename T1, typename T2,
          int Version=detail::CompressedPairSwitch<T1,T2>::value>
class CompressedPair;

// 0    derive from neither
template <typename T1, typename T2>
class CompressedPair<T1, T2, 0> {
public:
    typedef T1 first_type;
    typedef T2 second_type;
    typedef detail::add_ref<first_type> first_reference;
    typedef detail::add_ref<second_type> second_reference;
    typedef detail::add_const_ref<first_type>  first_const_reference;
    typedef detail::add_const_ref<second_type> second_const_reference;

    CompressedPair() {}

    template <typename U1, typename U2>
    CompressedPair(U1 && x, U2 && y)
      : first_(std::forward<U1>(x)), second_(std::forward<U2>(y))
    {}

    first_reference first() { return first_; }
    first_const_reference first() const { return first_; }

    second_reference second() { return second_; }
    second_const_reference second() const { return second_; }

    void swap(CompressedPair & y) {
       detail::generic_swap(first_, y.first());
       detail::generic_swap(second_, y.second());
    }

private:
    first_type first_;
    second_type second_;
};

// 1    derive from T1
template <typename T1, typename T2>
class CompressedPair<T1, T2, 1> : private std::remove_cv<T1>::type {
public:
    typedef T1 first_type;
    typedef T2 second_type;
    typedef detail::add_ref<first_type> first_reference;
    typedef detail::add_ref<second_type> second_reference;
    typedef detail::add_const_ref<first_type>  first_const_reference;
    typedef detail::add_const_ref<second_type> second_const_reference;

    CompressedPair() {}

    template <typename U1, typename U2>
    CompressedPair(U1 && x, U2 && y)
      : first_type(std::forward<U1>(x)), second_(std::forward<U2>(y))
    {}

    first_reference first() { return *this; }
    first_const_reference first() const { return *this; }

    second_reference second() { return second_; }
    second_const_reference second() const { return second_; }

    void swap(CompressedPair & y) {
       // no need to swap empty base class:
       detail::generic_swap(second_, y.second());
    }

private:
    second_type second_;
};

// 2    derive from T2
template <typename T1, typename T2>
class CompressedPair<T1, T2, 2> : private std::remove_cv<T2>::type {
public:
    typedef T1 first_type;
    typedef T2 second_type;
    typedef detail::add_ref<first_type> first_reference;
    typedef detail::add_ref<second_type> second_reference;
    typedef detail::add_const_ref<first_type> first_const_reference;
    typedef detail::add_const_ref<second_type> second_const_reference;

    CompressedPair() {}

    template <typename U1, typename U2>
    CompressedPair(U1 && x, U2 && y)
      : second_type(std::forward<U2>(y)), first_(std::forward<U1>(x))
    {}

    first_reference first() { return first_; }
    first_const_reference first() const { return first_; }

    second_reference second() { return *this; }
    second_const_reference second() const { return *this; }

    void swap(CompressedPair & y) {
       // no need to swap empty base class:
       detail::generic_swap(first_, y.first());
    }

private:
    first_type first_;
};

// 3    derive from T1 and T2
template <typename T1, typename T2>
class CompressedPair<T1, T2, 3> :
    private std::remove_cv<T1>::type,
    private std::remove_cv<T2>::type
{
public:
    typedef T1 first_type;
    typedef T2 second_type;
    typedef detail::add_ref<first_type> first_reference;
    typedef detail::add_ref<second_type> second_reference;
    typedef detail::add_const_ref<first_type> first_const_reference;
    typedef detail::add_const_ref<second_type> second_const_reference;

    CompressedPair() {}

    template <typename U1, typename U2>
    CompressedPair(U1 && x, U2 && y)
      : first_type(std::forward<U1>(x)), second_type(std::forward<U2>(y))
    {}

    first_reference first() { return *this; }
    first_const_reference first() const { return *this; }

    second_reference second() { return *this; }
    second_const_reference second() const { return *this; }

    void swap(CompressedPair & y) {
       // no need to swap empty base classes
    }

};

template <typename T1, typename T2>
inline void swap(CompressedPair<T1, T2>& x, CompressedPair<T1, T2>& y) {
   x.swap(y);
}

} // namespace ndarray

#endif // !NDARRAY_CompressedPair_hpp_INCLUDED
