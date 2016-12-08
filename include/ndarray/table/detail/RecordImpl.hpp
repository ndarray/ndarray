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

    std::shared_ptr<Schema> const & schema() const { return _storage->_schema; }

    template <typename T>
    typename Key<T>::reference get(Key<T> const & key) const {
        return key.dtype().make_reference_at(
            _buffer + _storage->_offsets[key.index()],
            _storage->_manager
        );
    }

    template <typename T>
    typename Key<T>::const_reference cget(Key<T> const & key) const {
        return key.dtype().make_const_reference_at(
            _buffer + _storage->_offsets[key.index()],
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

    std::shared_ptr<Schema> const & schema() const { return _storage->schema(); }

    template <typename T>
    typename Key<T>::reference get(Key<T> const & key) const {
        return key.dtype().make_reference_at(
            _buffer + _storage->_offsets[key.index()],
            _manager
        );
    }

    template <typename T>
    typename Key<T>::const_reference cget(Key<T> const & key) const {
        return key.dtype().make_const_reference_at(
            _buffer + _storage->_offsets[key.index()],
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

    std::shared_ptr<Schema> const & schema() const { return _storage->_schema; }

    template <typename T>
    typename Key<T>::reference get(Key<T> const & key) const {
        return key.dtype().make_reference_at(
            _buffer + _storage->_offsets[key.index()],
            _storage->_manager
        );
    }

    template <typename T>
    typename Key<T>::const_reference cget(Key<T> const & key) const {
        return key.dtype().make_const_reference_at(
            _buffer + _storage->_offsets[key.index()],
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

    std::shared_ptr<Schema> const & schema() const { return _storage->_schema; }

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


template <typename T>
template <typename Target, typename Source>
void Key<T>::assign_impl(
    KeyBase const & other,
    detail::RecordImpl<Target> const & target,
    detail::RecordImpl<Source> const & source
) const {
    Key<T> const & k = static_cast<Key<T> const &>(other);
    target.get(*this) = source.cget(k);
}


template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FixedRow> const & target,
    detail::RecordImpl<FixedRow> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FixedRow> const & target,
    detail::RecordImpl<FlexRow> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FixedRow> const & target,
    detail::RecordImpl<FixedCol> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FixedRow> const & target,
    detail::RecordImpl<FlexCol> const & source
) const {
    assign_impl(other, target, source);
}


template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FlexRow> const & target,
    detail::RecordImpl<FixedRow> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FlexRow> const & target,
    detail::RecordImpl<FlexRow> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FlexRow> const & target,
    detail::RecordImpl<FixedCol> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FlexRow> const & target,
    detail::RecordImpl<FlexCol> const & source
) const {
    assign_impl(other, target, source);
}


template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FixedCol> const & target,
    detail::RecordImpl<FixedRow> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FixedCol> const & target,
    detail::RecordImpl<FlexRow> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FixedCol> const & target,
    detail::RecordImpl<FixedCol> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FixedCol> const & target,
    detail::RecordImpl<FlexCol> const & source
) const {
    assign_impl(other, target, source);
}


template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FlexCol> const & target,
    detail::RecordImpl<FixedRow> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FlexCol> const & target,
    detail::RecordImpl<FlexRow> const & source
) const {
    assign_impl(other, target, source);
}

template <typename T>
void Key<T>::assign(
    KeyBase const & other,
    detail::RecordImpl<FlexCol> const & target,
    detail::RecordImpl<FixedCol> const & source
) const {
    assign_impl(other, target, source);
}

} // ndarray

#endif // !NDARRAY_table_detail_RecordImpl_hpp_INCLUDED