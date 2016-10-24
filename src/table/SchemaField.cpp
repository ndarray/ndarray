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
#include "ndarray/table/SchemaField.hpp"

namespace ndarray {

SchemaField & SchemaField::operator=(SchemaField && other) {
    if (name() != other.name()) {
        throw std::logic_error(
            "Cannot rename a SchemaField in-place; use Schema::rename."
        );
    }
    _key = std::move(other._key);
    Field::operator=(std::move(other));
    return *this;
}

SchemaField & SchemaField::operator=(Field const & other) {
    if (name() != other.name()) {
        throw std::logic_error(
            "Cannot rename a SchemaField in-place; use Schema::rename."
        );
    }
    Field::operator=(other);
    return *this;
}

SchemaField & SchemaField::operator=(Field && other) {
    if (name() != other.name()) {
        throw std::logic_error(
            "Cannot rename a SchemaField in-place; use Schema::rename."
        );
    }
    Field::operator=(std::move(other));
    return *this;
}

std::unique_ptr<SchemaField> SchemaField::copy() const {
    return std::unique_ptr<SchemaField>(
        new SchemaField(*this, _key->clone())
    );
}


} // ndarray
