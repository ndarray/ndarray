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
#ifndef NDARRAY_table_common_hpp_INCLUDED
#define NDARRAY_table_common_hpp_INCLUDED

#include "ndarray/common.hpp"

namespace ndarray {

namespace detail {

template <typename Storage> class RecordImpl;

} // detail

class FixedRow;
class FlexRow;
class FixedCol;
class FlexCol;

class SchemaWatcher;
class Schema;

class KeyBase;

template <typename T> class Key;

template <typename S> class RecordBase;
template <typename S> class Record;
template <typename S> class RecordRef;

} // ndarray

#endif // !NDARRAY_table_common_hpp_INCLUDED
