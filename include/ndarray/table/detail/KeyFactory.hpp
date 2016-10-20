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
#ifndef NDARRAY_table_detail_KeyFactory_hpp_INCLUDED
#define NDARRAY_table_detail_KeyFactory_hpp_INCLUDED

#include <memory>

#include "ndarray/table/Key.hpp"

namespace ndarray {
namespace detail {

class KeyFactory {
public:

    static void declare(
        std::string const & type_name,
        KeyFactory const * factory
    );

    static std::unique_ptr<KeyBase> invoke(
        size_t & offset,
        std::string const & type_name,
        void const * dtype
    );

    KeyFactory(KeyFactory const &) = delete;

    KeyFactory(KeyFactory &&) = delete;

    KeyFactory & operator=(KeyFactory const &) = delete;

    KeyFactory & operator=(KeyFactory &&) = delete;

    virtual ~KeyFactory() {}

protected:

    virtual std::unique_ptr<KeyBase> apply(
        size_t & offset,
        void const * dtype
    ) const = 0;

    KeyFactory() {}
};


} // detail
} // ndarray

#endif // !NDARRAY_table_detail_KeyFactory_hpp_INCLUDED