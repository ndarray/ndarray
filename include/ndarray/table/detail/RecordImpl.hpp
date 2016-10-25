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
#include "ndarray/table/Schema.hpp"
#include "ndarray/table/detail/SchemaDType.hpp"

namespace ndarray {
namespace detail {

class RecordImpl {
    typedef DType<Schema> dtype_t;
public:

    RecordImpl() : buffer(nullptr), _dtype(), _manager() {}

    RecordImpl(
        byte_t * buffer_,
        dtype_t dtype_,
        std::shared_ptr<Manager> manager_
    ) : buffer(buffer_),
        _dtype(std::move(dtype_)),
        _manager(std::move(manager_))
    {}

    RecordImpl(RecordImpl const &) = default;

    RecordImpl(RecordImpl &&) = default;

    RecordImpl & operator=(RecordImpl const &) = default;

    RecordImpl & operator=(RecordImpl &&) = default;

    void swap(RecordImpl & other) {
        std::swap(buffer, other.buffer);
        _dtype.swap(other._dtype);
        _manager.swap(other._manager);
    }

    dtype_t const & dtype() const { return _dtype; }

    std::shared_ptr<Manager> const & manager() const { return _manager; }

    byte_t * buffer;

private:
    dtype_t _dtype;
    std::shared_ptr<Manager> _manager;
};

void swap(RecordImpl & a, RecordImpl & b) {
    a.swap(b);
}

} // detail
} // ndarray

#endif // !NDARRAY_table_detail_RecordImpl_hpp_INCLUDED