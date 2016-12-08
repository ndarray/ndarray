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

    template <typename S> friend class Record;

    template <typename S> friend class RecordRef;

    virtual void initialize(byte_t * buffer) const = 0;

    virtual void destroy(byte_t * buffer) const = 0;

    virtual bool equals(KeyBase const & other) const = 0;


    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedRow> const & target,
        detail::RecordImpl<FixedRow> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedRow> const & target,
        detail::RecordImpl<FlexRow> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedRow> const & target,
        detail::RecordImpl<FixedCol> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedRow> const & target,
        detail::RecordImpl<FlexCol> const & source
    ) const = 0;


    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexRow> const & target,
        detail::RecordImpl<FixedRow> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexRow> const & target,
        detail::RecordImpl<FlexRow> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexRow> const & target,
        detail::RecordImpl<FixedCol> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexRow> const & target,
        detail::RecordImpl<FlexCol> const & source
    ) const = 0;


    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedCol> const & target,
        detail::RecordImpl<FixedRow> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedCol> const & target,
        detail::RecordImpl<FlexRow> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedCol> const & target,
        detail::RecordImpl<FixedCol> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedCol> const & target,
        detail::RecordImpl<FlexCol> const & source
    ) const = 0;


    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexCol> const & target,
        detail::RecordImpl<FixedRow> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexCol> const & target,
        detail::RecordImpl<FlexRow> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexCol> const & target,
        detail::RecordImpl<FixedCol> const & source
    ) const = 0;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexCol> const & target,
        detail::RecordImpl<FlexCol> const & source
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

    explicit Key(size_t index, DType<T> dtype) :
        _index_and_dtype(index, std::move(dtype))
    {}

    Key(Key const &) = delete;

    Key(Key &&) = delete;

    Key & operator=(Key const &) = delete;

    Key & operator=(Key &&) = delete;

    virtual bool is_direct() const {
        return DType<T>::is_direct;
    }

    virtual std::string const & type_name() const {
        return DType<T>::name();
    }

    size_t index() const { return _index_and_dtype.first(); }

    dtype_t const & dtype() const { return _index_and_dtype.second(); }

protected:

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

    template <typename Target, typename Source>
    void assign_impl(
        KeyBase const & other,
        detail::RecordImpl<Target> const & target,
        detail::RecordImpl<Source> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedRow> const & target,
        detail::RecordImpl<FixedRow> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedRow> const & target,
        detail::RecordImpl<FlexRow> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedRow> const & target,
        detail::RecordImpl<FixedCol> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedRow> const & target,
        detail::RecordImpl<FlexCol> const & source
    ) const;


    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexRow> const & target,
        detail::RecordImpl<FixedRow> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexRow> const & target,
        detail::RecordImpl<FlexRow> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexRow> const & target,
        detail::RecordImpl<FixedCol> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexRow> const & target,
        detail::RecordImpl<FlexCol> const & source
    ) const;


    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedCol> const & target,
        detail::RecordImpl<FixedRow> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedCol> const & target,
        detail::RecordImpl<FlexRow> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedCol> const & target,
        detail::RecordImpl<FixedCol> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FixedCol> const & target,
        detail::RecordImpl<FlexCol> const & source
    ) const;


    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexCol> const & target,
        detail::RecordImpl<FixedRow> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexCol> const & target,
        detail::RecordImpl<FlexRow> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexCol> const & target,
        detail::RecordImpl<FixedCol> const & source
    ) const;

    virtual void assign(
        KeyBase const & other,
        detail::RecordImpl<FlexCol> const & target,
        detail::RecordImpl<FlexCol> const & source
    ) const;

    virtual std::unique_ptr<KeyBase> clone() const {
        return std::unique_ptr<KeyBase>(new Key<T>(index(), dtype()));
    }

private:
    CompressedPair<size_t,DType<T>> _index_and_dtype;
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