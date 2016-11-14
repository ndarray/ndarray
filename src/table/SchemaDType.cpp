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
#include "ndarray/table/SchemaDType.hpp"
#include "ndarray/table/Record.hpp"

namespace ndarray {


DType<Schema>::DType(Schema const & schema) : _schema(schema) {
    _schema.align();
}

DType<Schema>::DType(Schema && schema) : _schema(std::move(schema)) {
    _schema.align();
}

std::string const & DType<Schema>::name() {
    static std::string const x("Schema");
    return x;
}

void DType<Schema>::initialize(byte_t * buffer) const {
    for (auto const & field : *_schema) {
        field.key().initialize(buffer);
    }
}

void DType<Schema>::destroy(byte_t * buffer) const {
    for (auto const & field : *_schema) {
        field.key().destroy(buffer);
    }
}

auto DType<Schema>::make_reference_at(
    byte_t * buffer,
    std::shared_ptr<Manager> const & manager
) const -> reference {
    return RecordRef<Schema>(buffer, *this, manager);
}

auto DType<Schema>::make_const_reference_at(
    byte_t const * buffer,
    std::shared_ptr<Manager> const & manager
) const -> const_reference {
    return RecordRef<Schema const>(buffer, *this, manager);
}


} // ndarray
