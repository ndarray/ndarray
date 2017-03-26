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

#include "ndarray/table/common.hpp"
#include "ndarray/formatting/types.hpp"
#include "ndarray/CompressedPair.hpp"

namespace ndarray {


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

    virtual std::string const & type_name() const = 0;

    template <typename T>
    operator Key<T> const & () const;

    size_t index() const { return _index; }

    virtual bool is_pod() const = 0;

    virtual size_t nbytes() const = 0;

    virtual size_t alignment() const = 0;

    virtual ~KeyBase() {}

protected:

    friend class SchemaField;

    friend class Schema;

    template <typename S> friend class Record;

    template <typename S> friend class RecordRef;


    explicit KeyBase(size_t index_) : _index(index_) {}

    KeyBase(KeyBase const &) = default;

    KeyBase(KeyBase &&) = delete;

    KeyBase & operator=(KeyBase const &) = delete;

    KeyBase & operator=(KeyBase &&) = delete;


    virtual void initialize(byte_t * buffer) const = 0;

    virtual void destroy(byte_t * buffer) const = 0;

    virtual bool equals(KeyBase const & other) const = 0;

    virtual void assign(
        byte_t * destination,
        byte_t const * source
    ) const = 0;

    virtual std::unique_ptr<KeyBase> clone() const = 0;

private:
    size_t _index;
};


template <typename T>
class Key : public KeyBase {
public:

    typedef DType<T> dtype_t;
    typedef typename DType<T>::value_type value_type;
    typedef typename DType<T>::reference reference;
    typedef typename DType<T>::const_reference const_reference;

    explicit Key(size_t index, DType<T> dtype) :
        KeyBase(index), _dtype(std::move(dtype))
    {}

    virtual bool is_direct() const {
        return DType<T>::is_direct;
    }

    virtual std::string const & type_name() const {
        return DType<T>::name();
    }

    dtype_t const & dtype() const { return _dtype; }

    virtual bool is_pod() const { return dtype_t::is_pod; }

    virtual size_t nbytes() const { return dtype().nbytes(); }

    virtual size_t alignment() const { return dtype().alignment(); }

protected:

    Key(Key const &) = default;

    Key(Key &&) = delete;

    Key & operator=(Key const &) = delete;

    Key & operator=(Key &&) = delete;

    virtual void initialize(byte_t * buffer) const {
        dtype().initialize(buffer);
    }

    virtual void destroy(byte_t * buffer) const {
        dtype().destroy(buffer);
    }

    virtual bool equals(KeyBase const & other) const {
        Key<T> const * k = dynamic_cast<Key<T> const *>(this);
        return k && index() == k->index() && dtype() == k->dtype();
    }

    virtual void assign(
        byte_t * destination,
        byte_t const * source
    ) const {
        dtype().make_reference_at(destination, nullptr)
            = dtype().make_const_reference_at(source, nullptr);
    }

    virtual std::unique_ptr<KeyBase> clone() const {
        return std::unique_ptr<KeyBase>(new Key<T>(*this));
    }

private:
    DType<T> _dtype;
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