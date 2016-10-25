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
#include "ndarray/table/Record.hpp"

#define NDARRAY_table_Record_cpp_ACTIVE

namespace ndarray {

template <typename S>
RecordRef<S> const & RecordRef<S>::operator=(
    Record<S const> const & other
) const {
    if (!this->schema()->equal_keys(*other.schema())) {
        throw std::logic_error("Cannot assign records with unequal keys.");
    }
    for (auto const & field : *this->schema()) {
        field.key().assign(
            field.key(), this->_impl.buffer, other._impl.buffer
        );
    }
    return *this;
}

template <typename S>
RecordRef<S> const & RecordRef<S>::operator=(Record<S> && other) const {
    if (!this->schema()->equal_keys(*other.schema())) {
        throw std::logic_error("Cannot assign records with unequal keys.");
    }
    for (auto const & field : *this->schema()) {
        field.key().move(
            field.key(), this->_impl.buffer, other._impl.buffer
        );
    }
    other = Record<S>();  // seems safer to reset other; is it worthwhile?
    return *this;
}

template class RecordBase<Schema>;
template class RecordBase<Schema const>;
template class Record<Schema>;
template class Record<Schema const>;
template class RecordRef<Schema>;
template class RecordRef<Schema const>;

} // ndarray
