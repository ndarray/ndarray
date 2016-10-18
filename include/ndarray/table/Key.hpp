// -*- c++ -*-
/*
 * Copyright (c) 2010-2016, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_table_Key_hpp_INCLUDED
#define NDARRAY_table_Key_hpp_INCLUDED

#include <memory>

#include "ndarray/common.hpp"
#include "ndarray/formatting/types.hpp"

namespace ndarray {


template <typename T> class Key;


class TypeError : public std::logic_error {

    std::string format(
        std::string const & desired,
        std::string const & actual
    ) {
        std::ostringstream s;
        s << "Key has type '" << actual << "'', not '" << desired << "'.";
        return s.str();
    }

public:

    explicit TypeError(char const * msg) : std::logic_error(msg) {}

    explicit TypeError(
        std::string const & desired,
        std::string const & actual
    ) :
        std::logic_error(format(desired, actual))
    {}

};


class KeyBase {
public:

    KeyBase() {}

    KeyBase(KeyBase const &) = delete;

    KeyBase(KeyBase &&) = delete;

    KeyBase & operator=(KeyBase const &) = delete;

    KeyBase & operator=(KeyBase &&) = delete;

    virtual bool is_direct() const = 0;

    virtual std::string const & type_name() const = 0;

    template <typename T>
    operator Key<T> const & () const;

    virtual ~KeyBase() {}

};


template <typename T>
class Key : public KeyBase {
public:

    typedef DType<T> dtype_t;
    typedef typename DType<T>::value_type value_type;
    typedef typename DType<T>::reference reference;
    typedef typename DType<T>::const_reference const_reference;

    explicit Key(offset_ offset, DType<T> dtype) :
        _offset_and_dtype(offset, std::move(dtype))
    {}

    Key(Key const &) = delete;

    Key(Key &&) = delete;

    Key & operator=(Key const &) = delete;

    Key & operator=(Key &&) = delete;

    reference make_reference(byte_t * buffer) const {
        return _offset_and_dtype().second().make_reference_at(
            buffer + _offset_and_dtype().first();
        );
    }

    const_reference make_const_reference(byte_t const * buffer) const {
        return _offset_and_dtype().second().make_const_reference_at(
            buffer + _offset_and_dtype().first();
        );
    }

    virtual bool is_direct() const {
        return DType<T>::is_direct();
    }

    virtual std::string const & type_name() const {
        return DType<T>::name();
    }

private:
    detail::CompressedPair<offset_t,DType<T>> _offset_and_dtype;
};


template <typename T>
inline KeyBase::operator Key<T> const & () const {
    try {
        return dynamic_cast<Key<T> const &>(*this);
    } catch (std::bad_cast &) {
        throw TypeError(DType<T>::name(), this->name());
    }
}


} // ndarray

#endif // !NDARRAY_table_Key_hpp_INCLUDED