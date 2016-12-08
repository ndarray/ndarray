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
template <typename Other>
RecordRef<S> const & RecordRef<S>::operator=(
    Record<Other> const & other
) const {
    if (!this->schema()->equal_keys(*other.schema())) {
        throw std::logic_error("Cannot assign records with unequal keys.");
    }
    for (auto const & field : *this->schema()) {
        field.key().assign(
            field.key(), this->_impl, other._impl
        );
    }
    return *this;
}

template RecordRef<FixedRow> const &
    RecordRef<FixedRow>::operator=(Record<FixedRow> const &) const;
template RecordRef<FixedRow> const &
    RecordRef<FixedRow>::operator=(Record<FlexRow> const &) const;
template RecordRef<FixedRow> const &
    RecordRef<FixedRow>::operator=(Record<FixedCol> const &) const;
template RecordRef<FixedRow> const &
    RecordRef<FixedRow>::operator=(Record<FlexCol> const &) const;

template RecordRef<FlexRow> const &
    RecordRef<FlexRow>::operator=(Record<FixedRow> const &) const;
template RecordRef<FlexRow> const &
    RecordRef<FlexRow>::operator=(Record<FlexRow> const &) const;
template RecordRef<FlexRow> const &
    RecordRef<FlexRow>::operator=(Record<FixedCol> const &) const;
template RecordRef<FlexRow> const &
    RecordRef<FlexRow>::operator=(Record<FlexCol> const &) const;

template RecordRef<FixedCol> const &
    RecordRef<FixedCol>::operator=(Record<FixedRow> const &) const;
template RecordRef<FixedCol> const &
    RecordRef<FixedCol>::operator=(Record<FlexRow> const &) const;
template RecordRef<FixedCol> const &
    RecordRef<FixedCol>::operator=(Record<FixedCol> const &) const;
template RecordRef<FixedCol> const &
    RecordRef<FixedCol>::operator=(Record<FlexCol> const &) const;

template RecordRef<FlexCol> const &
    RecordRef<FlexCol>::operator=(Record<FixedRow> const &) const;
template RecordRef<FlexCol> const &
    RecordRef<FlexCol>::operator=(Record<FlexRow> const &) const;
template RecordRef<FlexCol> const &
    RecordRef<FlexCol>::operator=(Record<FixedCol> const &) const;
template RecordRef<FlexCol> const &
    RecordRef<FlexCol>::operator=(Record<FlexCol> const &) const;

} // ndarray
