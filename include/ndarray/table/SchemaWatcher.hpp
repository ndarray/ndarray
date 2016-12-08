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

#include <mutex>

#include "ndarray/table/common.hpp"
#include "ndarray/table/Schema.hpp"

namespace ndarray {

class SchemaWatcher {
public:

    virtual ~SchemaWatcher() = 0;

protected:

    // Should be called by subclasses during construction.
    void attach(Schema & schema) {
        std::lock_guard<std::mutex> guard(schema._watchers_mutex);
        schema._watchers.push_front(this);
    }

    // Should be called by derived classes in their destructors.
    void detach(Schema & schema) {
        std::lock_guard<std::mutex> guard(schema._watchers_mutex);
        schema._watchers.remove(this);
    }

};

// Pure virtual doesn't save us from having to define it, since base class
// destructors are always called.
inline SchemaWatcher::~SchemaWatcher() {}

} // ndarray

#endif // !NDARRAY_table_SchemaWatcher_hpp_INCLUDED