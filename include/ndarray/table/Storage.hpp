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
#ifndef NDARRAY_table_Storage_hpp_INCLUDED
#define NDARRAY_table_Storage_hpp_INCLUDED

#include "ndarray/Array.hpp"
#include "ndarray/table/SchemaWatcher.hpp"
#include "ndarray/table/Schema.hpp"

namespace ndarray {


// Schema + buffer + Manager + size
// Always row-major contiguous.
// Cannot add or reorder columns.
// Cannot add or reorder records.
// Can access Records.
// Can access column Arrays (noncontiguous).
// Can be viewed as structured numpy.ndarray.
// Allows only superficial modification of watched Schemas.
class FixedRow : public SchemaWatcher {
public:

    template <typename T>
    using Column = Array<T,1,0>;

    FixedRow(size_t size_, std::shared_ptr<Schema> schema_);

    FixedRow(FixedRow const &) = default;

    FixedRow(FixedRow &&) = default;

    FixedRow & operator=(FixedRow const &) = default;

    FixedRow & operator=(FixedRow &&) = default;

    void swap(FixedRow & other);

    std::shared_ptr<Schema> const & schema() const { return _schema; }

    size_t size() const { return _layout->size(); }

    template <typename T>
    Column<T> column(Key<T> const & key) const {
        return Column<T>(_buffer + _offsets[key.index()],
                         _manager, _layout, key.dtype());
    }

private:

    template <typename Storage> friend class detail::RecordImpl;

    byte_t * _buffer;
    std::shared_ptr<Schema> _schema;
    std::shared_ptr<Manager> _manager;
    std::vector<size_t> _offsets;
    std::shared_ptr<detail::Layout<1>> _layout;
};


// Schema + std::vector<Record>
// Cannot add or reorder columns.
// Each record may be allocated separately, and may not be contiguous.
// Can add Records.  Can reorder records (shallow).
// Can access Records.
// Can access column Arrays (inefficient)?
// Can shallow-construct from FixedCol, FixedRow.
// Allows only superficial modification of watched Schemas.
class FlexRow : public SchemaWatcher {
public:

    FlexRow(size_t size_, std::shared_ptr<Schema> schema_);

    FlexRow(FlexRow const &) = default;

    FlexRow(FlexRow &&) = default;

    FlexRow & operator=(FlexRow const &) = default;

    FlexRow & operator=(FlexRow &&) = default;

    void swap(FlexRow & other);

    std::shared_ptr<Schema> const & schema() const { return _schema; }

    size_t size() const { return _rows.size(); }

private:

    template <typename Storage> friend class detail::RecordImpl;

    std::shared_ptr<Schema> _schema;
    std::vector<size_t> _offsets;
    std::vector<std::pair<byte_t*,std::shared_ptr<Manager>>> _rows;
};


// Schema + buffer + Manager + size
// Always col-major contiguous.
// Cannot add or reorder columns.
// Cannot add or reorder records.
// Can access Records (noncontiguous).
// Can acccess column Arrays.
// Allows only superficial modification of watched Schemas.
class FixedCol : public SchemaWatcher {
public:

    template <typename T>
    using Column = Array<T,1,1>;

    FixedCol(size_t size_, std::shared_ptr<Schema> schema_);

    FixedCol(FixedCol const &) = default;

    FixedCol(FixedCol &&) = default;

    FixedCol & operator=(FixedCol const &) = default;

    FixedCol & operator=(FixedCol &&) = default;

    void swap(FixedCol & other);

    std::shared_ptr<Schema> const & schema() const { return _schema; }

    size_t size() const { return _layout->size(); }

    template <typename T>
    Column<T> column(Key<T> const & key) const {
        return Column<T>(_buffer + _offsets[key.index()],
                         _manager, _layout, key.dtype());
    }

private:

    template <typename Storage> friend class detail::RecordImpl;

    byte_t * _buffer;
    std::shared_ptr<Schema> _schema;
    std::shared_ptr<Manager> _manager;
    std::vector<size_t> _offsets;
    std::shared_ptr<detail::Layout<1>> _layout;
};


// Schema + std::vector of column Arrays
// Each column may be allocated separately, and may not be contiguous.
// Can add new columns and/or reorder them (shallow).
// Cannot add Records.  Can reorder (deep).
// Can access column Arrays.
// Can access Records (inefficient)?
// Can be viewed as astropy.table.Table.
// Can shallow-construct from FixedCol, FixedRow.
// Allows full modification of watched Schemas.
class FlexCol : public SchemaWatcher {
public:

    template <typename T>
    using Column = Array<T,1,0>;

    FlexCol(size_t size_, std::shared_ptr<Schema> schema_);

    FlexCol(FlexCol const &) = default;

    FlexCol(FlexCol &&) = default;

    FlexCol & operator=(FlexCol const &) = default;

    FlexCol & operator=(FlexCol &&) = default;

    void swap(FlexCol & other);

    std::shared_ptr<Schema> const & schema() const { return _schema; }

    size_t size() const { return _nrows; }

    template <typename T>
    Column<T> column(Key<T> const & key) const {
        return static_cast<Holder<T> const &>(_columns[key.index()]).array;
    }

private:

    template <typename Storage> friend class detail::RecordImpl;

    class HolderBase {
    public:
        virtual ~HolderBase() {}
    };

    template <typename T>
    class Holder : public HolderBase {
    public:
        Column<T> array;
    };

    std::shared_ptr<Schema> _schema;
    std::vector<std::unique_ptr<HolderBase>> _columns;
    size_t _nrows;
};


} // ndarray

#endif // !NDARRAY_table_Storage_hpp_INCLUDED