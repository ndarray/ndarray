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
#ifndef NDARRAY_table_detail_SimpleKeyFactory_hpp_INCLUDED
#define NDARRAY_table_detail_SimpleKeyFactory_hpp_INCLUDED

#include <memory>

#include "ndarray/DType.hpp"
#include "ndarray/table/detail/KeyFactory.hpp"

namespace ndarray {
namespace detail {

template <typename T>
class SimpleKeyFactory {
public:

    SimpleKeyFactory() {
        KeyFactory::declare(DType<T>::name(), this);
    }

    virtual std::unique_ptr<KeyBase> apply(
        size_t & offset,
        void const * dtype
    ) const {
        DType<T> const & dt = *reinterpret_cast<DType<T> const*>(dtype);
        if (offset % dt.alignment()) {
            offset += dt.alignment() - offset % dt.alignment();
        }
        std::unique_ptr<KeyBase> r(new Key<T>(offset, dt));
        offset += dt.nbytes();
        return r;
    }

};

} // detail
} // ndarray

#endif // !NDARRAY_table_detail_SimpleKeyFactory_hpp_INCLUDED