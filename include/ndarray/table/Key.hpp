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
#include "ndarray/CompressedPair.hpp"

namespace ndarray {


class Schema;


template <typename T> class Key;


class TypeError : public std::logic_error {

    std::string format(std::string const & desired, std::string const & actual);

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

protected:

    friend class SchemaField;

    friend class Schema;

    friend class DType<Schema>;

    template <typename S> friend class Record;

    template <typename S> friend class RecordRef;

    virtual void initialize(byte_t * buffer) const = 0;

    virtual void destroy(byte_t * buffer) const = 0;

    virtual bool equals(KeyBase const & other) const = 0;

    virtual void assign(
        KeyBase const & other,
        byte_t const * in,
        byte_t * out
    ) const = 0;

    virtual void move(
        KeyBase const & other,
        byte_t * in,
        byte_t * out
    ) const = 0;

    virtual std::unique_ptr<KeyBase> clone() const = 0;
};


template <typename T>
class Key : public KeyBase {
public:

    typedef DType<T> dtype_t;
    typedef typename DType<T>::value_type value_type;
    typedef typename DType<T>::reference reference;
    typedef typename DType<T>::const_reference const_reference;

    explicit Key(offset_t offset, DType<T> dtype) :
        _offset_and_dtype(offset, std::move(dtype))
    {}

    Key(Key const &) = delete;

    Key(Key &&) = delete;

    Key & operator=(Key const &) = delete;

    Key & operator=(Key &&) = delete;

    reference make_reference(
        byte_t * buffer,
        std::shared_ptr<Manager> const & manager
    ) const {
        return _offset_and_dtype.second().make_reference_at(
            buffer + _offset_and_dtype.first(),
            manager
        );
    }

    const_reference make_const_reference(
        byte_t const * buffer,
        std::shared_ptr<Manager> const & manager
    ) const {
        return _offset_and_dtype.second().make_const_reference_at(
            buffer + _offset_and_dtype.first(),
            manager
        );
    }

    virtual bool is_direct() const {
        return DType<T>::is_direct;
    }

    virtual std::string const & type_name() const {
        return DType<T>::name();
    }

protected:

    virtual void initialize(byte_t * buffer) const {
        _offset_and_dtype.second().initialize(
            buffer + _offset_and_dtype.first()
        );
    }

    virtual void destroy(byte_t * buffer) const {
        _offset_and_dtype.second().destroy(
            buffer + _offset_and_dtype.first()
        );
    }

    virtual bool equals(KeyBase const & other) const {
        Key<T> const * k = dynamic_cast<Key<T> const *>(this);
        return k && _offset_and_dtype.first() == k->_offset_and_dtype.first()
            && _offset_and_dtype.second() == k->_offset_and_dtype.second();
    }

    virtual void assign(
        KeyBase const & other,
        byte_t const * in_buffer,
        byte_t * out_buffer
    ) const {
        Key<T> const & out_key = static_cast<Key<T> const &>(other);
        out_key.make_reference(out_buffer, nullptr)
            = this->make_const_reference(in_buffer, nullptr);
    }

    virtual void move(
        KeyBase const & other,
        byte_t * in_buffer,
        byte_t * out_buffer
    ) const {
        Key<T> const & out_key = static_cast<Key<T> const &>(other);
        out_key.make_reference(out_buffer, nullptr)
            = std::move(this->make_reference(in_buffer, nullptr));
    }

    virtual std::unique_ptr<KeyBase> clone() const {
        return std::unique_ptr<KeyBase>(
            new Key<T>(_offset_and_dtype.first(), _offset_and_dtype.second())
        );
    }

private:
    CompressedPair<offset_t,DType<T>> _offset_and_dtype;
};


template <typename T>
inline KeyBase::operator Key<T> const & () const {
    Key<T> * r = dynamic_cast<Key<T> const *>(this);
    if (!r) {
        throw TypeError(DType<T>::name(), this->type_name());
    }
    return *r;
}


} // ndarray

#endif // !NDARRAY_table_Key_hpp_INCLUDED