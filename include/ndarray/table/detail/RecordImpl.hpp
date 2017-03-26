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
#ifndef NDARRAY_table_detail_RecordImpl_hpp_INCLUDED
#define NDARRAY_table_detail_RecordImpl_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/table/Storage.hpp"
#include "ndarray/table/Schema.hpp"

namespace ndarray {
namespace detail {

template <typename Storage> class RecordImpl;

template <>
class RecordImpl<FixedRow> {
public:

    RecordImpl(
        size_t index,
        std::shared_ptr<FixedRow> storage
    );

    RecordImpl(RecordImpl const &) = default;

    RecordImpl(RecordImpl &&) = default;

    RecordImpl & operator=(RecordImpl const &) = default;

    RecordImpl & operator=(RecordImpl &&) = default;

    void swap(RecordImpl & other);

    std::shared_ptr<Schema> const & schema() const {
        return _storage->schema();
    }

    byte_t * get_raw(KeyBase const & key) const {
        return _buffer + _storage->_offsets[key.index()];
    }

    template <typename T>
    byte_t const * cget_raw(KeyBase const & key) const {
        return _buffer + _storage->_offsets[key.index()];
    }

    template <typename T>
    typename Key<T>::reference get(Key<T> const & key) const {
        return key.dtype().make_reference_at(
            get_raw(key),
            _storage->_manager
        );
    }

    template <typename T>
    typename Key<T>::const_reference cget(Key<T> const & key) const {
        return key.dtype().make_const_reference_at(
            cget_raw(key),
            _storage->_manager
        );
    }

private:
    byte_t * _buffer;
    std::shared_ptr<FixedRow> _storage;
};

template <>
class RecordImpl<FlexRow> {
public:

    RecordImpl(
        size_t index,
        std::shared_ptr<FlexRow> storage
    );

    RecordImpl(RecordImpl const &) = default;

    RecordImpl(RecordImpl &&) = default;

    RecordImpl & operator=(RecordImpl const &) = default;

    RecordImpl & operator=(RecordImpl &&) = default;

    void swap(RecordImpl & other);

    std::shared_ptr<Schema> const & schema() const {
        return _storage->schema();
    }

    byte_t * get_raw(KeyBase const & key) const {
        return _buffer + _storage->_offsets[key.index()];
    }

    template <typename T>
    byte_t const * cget_raw(KeyBase const & key) const {
        return _buffer + _storage->_offsets[key.index()];
    }

    template <typename T>
    typename Key<T>::reference get(Key<T> const & key) const {
        return key.dtype().make_reference_at(
            get_raw(key),
            _manager
        );
    }

    template <typename T>
    typename Key<T>::const_reference cget(Key<T> const & key) const {
        return key.dtype().make_const_reference_at(
            cget_raw(key),
            _manager
        );
    }

private:
    byte_t * _buffer;
    std::shared_ptr<Manager> _manager;
    std::shared_ptr<FlexRow> _storage;
};


template <>
class RecordImpl<FixedCol> {
public:

    RecordImpl(
        size_t index,
        std::shared_ptr<FixedCol> storage
    );

    RecordImpl(RecordImpl const &) = default;

    RecordImpl(RecordImpl &&) = default;

    RecordImpl & operator=(RecordImpl const &) = default;

    RecordImpl & operator=(RecordImpl &&) = default;

    void swap(RecordImpl & other);

    std::shared_ptr<Schema> const & schema() const {
        return _storage->schema();
    }

    byte_t * get_raw(KeyBase const & key) const {
        return _buffer + _storage->_offsets[key.index()];
    }

    template <typename T>
    byte_t const * cget_raw(KeyBase const & key) const {
        return _buffer + _storage->_offsets[key.index()];
    }

    template <typename T>
    typename Key<T>::reference get(Key<T> const & key) const {
        return key.dtype().make_reference_at(
            get_raw(key),
            _storage->_manager
        );
    }

    template <typename T>
    typename Key<T>::const_reference cget(Key<T> const & key) const {
        return key.dtype().make_const_reference_at(
            cget_raw(key),
            _storage->_manager
        );
    }

private:
    byte_t * _buffer;
    std::shared_ptr<FixedCol> _storage;
};


template <>
class RecordImpl<FlexCol> {
public:

    RecordImpl(
        size_t index,
        std::shared_ptr<FlexCol> storage
    );

    RecordImpl(RecordImpl const &) = default;

    RecordImpl(RecordImpl &&) = default;

    RecordImpl & operator=(RecordImpl const &) = default;

    RecordImpl & operator=(RecordImpl &&) = default;

    void swap(RecordImpl & other);

    std::shared_ptr<Schema> const & schema() const {
        return _storage->schema();
    }

    byte_t * get_raw(KeyBase const & key) const {
        auto const & array = _storage->_columns[key.index()]->array;
        return reinterpret_cast<byte_t*>(array.data() + array.stride()*_index);
    }

    template <typename T>
    byte_t const * cget_raw(KeyBase const & key) const {
        return get_raw(key);
    }

    template <typename T>
    typename Key<T>::reference get(Key<T> const & key) const {
        return _storage->_columns[key.index()][_index];
    }

    template <typename T>
    typename Key<T>::const_reference cget(Key<T> const & key) const {
        return _storage->_columns[key.index()][_index];
    }

private:
    size_t _index;
    std::shared_ptr<FlexCol> _storage;
};

template <typename Storage>
void swap(RecordImpl<Storage> & a, RecordImpl<Storage> & b) {
    a.swap(b);
}


} // detail
} // ndarray

#endif // !NDARRAY_table_detail_RecordImpl_hpp_INCLUDED