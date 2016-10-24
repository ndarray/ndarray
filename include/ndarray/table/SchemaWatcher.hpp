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
#ifndef NDARRAY_table_SchemaWatcher_hpp_INCLUDED
#define NDARRAY_table_SchemaWatcher_hpp_INCLUDED

#include "ndarray/common.hpp"

namespace ndarray {

class SchemaWatcher {
public:
    virtual ~SchemaWatcher() {}
};

} // ndarray

#endif // !NDARRAY_table_SchemaWatcher_hpp_INCLUDED