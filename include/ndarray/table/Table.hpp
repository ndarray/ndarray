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
#ifndef NDARRAY_table_Table_hpp_INCLUDED
#define NDARRAY_table_Table_hpp_INCLUDED

#include "ndarray/detail/utils.hpp"
#include "ndarray/table/Storage.hpp"

namespace ndarray {


template <typename S>
class TableBase {
    using Storage = std::remove_const<S>::type;
public:

    template <typename T>
    using Column = Storage::Column<typename transfer_const<S,T>::type>;

    std::shared_ptr<Schema> const & schema() const {
        return _storage->schema();
    }

    size_t size() const { return _storage->size(); }

    bool empty() const { return size() == 0u; }

    template <typename T>
    Column<T> operator[](Key<T> const & key) const {
        return _storage->column(key);
    }

    Record<S> operator[](size_t index) const {
        return Record<S>(RecordImpl<Storage>(index, _storage));
    }

protected:

    explicit TableBase(std::shared_ptr<Storage> storage_) :
        _storage(storage) {}

    TableBase(size_t size_, std::shared_ptr<Schema> schema_) :
        _storage(std::make_shared<Storage>(size_, std::move(schema_)))
    {}

    TableBase(TableBase const & other) :
        _storage(std::make_shared<Storage>(*other._storage))
    {}

    TableBase(TableBase && other) :
        _storage(std::make_shared<Storage>(std::move(*other._storage)))
    {}

    TableBase & operator=(TableBase const & other) {
        *_storage = *other._storage;
        return *this;
    }

    TableBase & operator=(TableBase && other) {
        *_storage = std::move(*other._storage);
        return *this;
    }

    void swap(TableBase & other) {
        _storage->swap(*other._storage);
    }

    Storage & storage() { return *_storage; }

    Storage const & storage() const { return *_storage; }

private:
    std::shared_ptr<Storage> _storage;
};


} // ndarray

#endif // !NDARRAY_table_Table_hpp_INCLUDED