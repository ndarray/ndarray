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
#ifndef NDARRAY_table_detail_SchemaDType_hpp_INCLUDED
#define NDARRAY_table_detail_SchemaDType_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/DType.hpp"
#include "ndarray/table/Schema.hpp"

namespace ndarray {

template <typename S> class Record;
template <typename S> class RecordRef;

template <>
class DType<Schema> {
public:

    typedef Record<Schema> value_type;
    typedef RecordRef<Schema> reference;
    typedef RecordRef<Schema const> const_reference;
    typedef Record<Schema> const * pointer;
    typedef Record<Schema const> const * const_pointer;
    static constexpr bool is_pod = false;
    static constexpr bool is_direct = false;

    DType() : _schema() {}

    explicit DType(Schema const & schema);

    explicit DType(Schema && schema);

    DType(DType const & other) = default;
    DType(DType && other) = default;

    DType & operator=(DType const & other) = default;
    DType & operator=(DType && other) = default;

    void swap(DType & other) { _schema.swap(other._schema); }

    size_t alignment() const { return _schema->alignment(); }

    size_t nbytes() const { return _schema->nbytes(); }

    static std::string const & name();

    bool operator==(DType const & other) const {
        return *_schema == *other._schema;
    }

    bool operator!=(DType const & other) const {
        return *_schema != *other._schema;
    }

    void initialize(byte_t * buffer) const;

    void destroy(byte_t * buffer) const;

    reference make_reference_at(
        byte_t * buffer,
        std::shared_ptr<Manager> const & manager
    ) const;

    const_reference make_const_reference_at(
        byte_t const * buffer,
        std::shared_ptr<Manager> const & manager
    ) const;

    std::shared_ptr<Schema> const & schema() const { return _schema; }

private:
    std::shared_ptr<Schema> _schema;
};


} // ndarray

#endif // !NDARRAY_table_detail_SchemaDType_hpp_INCLUDED