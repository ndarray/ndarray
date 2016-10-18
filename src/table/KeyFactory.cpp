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

#include "ndarray/table/detail/KeyFactory.hpp"
#include "ndarray/table/detail/SimpleKeyFactory.hpp"

namespace ndarray {
namespace detail {

namespace {

std::unordered_map<std::string, KeyFactory const *> registry;

} // anonymous

void KeyFactory::declare(
    std::string const & type_name,
    KeyFactory const * factory
) {
    registry.emplace(type_name, factory);
}

std::unique_ptr<KeyBase> KeyFactory::invoke(
    offset_t & offset,
    std::string const & type_name,
    void * dtype
) {
    auto iter = registry.find(type_name);
    if (iter == registry.end()) {
        throw std::runtime_error(
            "Keys of type '" + type_name
            + "' cannot be added directly to Schemas."
        );
    }
    return iter->second->apply(offset, dtype);
}

template class SimpleKeyFactory<char>;
static SimpleKeyFactory<char> char_KeyFactory;

template class SimpleKeyFactory<signed char>;
static SimpleKeyFactory<signed char> schar_KeyFactory;

template class SimpleKeyFactory<unsigned char>;
static SimpleKeyFactory<unsigned char> uchar_KeyFactory;

template class SimpleKeyFactory<short>;
static SimpleKeyFactory<short> short_KeyFactory;

template class SimpleKeyFactory<unsigned short>;
static SimpleKeyFactory<unsigned short> ushort_KeyFactory;

template class SimpleKeyFactory<int>;
static SimpleKeyFactory<int> int_KeyFactory;

template class SimpleKeyFactory<unsigned int>;
static SimpleKeyFactory<unsigned int> uint_KeyFactory;

template class SimpleKeyFactory<long>;
static SimpleKeyFactory<long> long_KeyFactory;

template class SimpleKeyFactory<unsigned long>;
static SimpleKeyFactory<unsigned long> ulong_KeyFactory;

template class SimpleKeyFactory<long long>;
static SimpleKeyFactory<long long> longlong_KeyFactory;

template class SimpleKeyFactory<unsigned long long>;
static SimpleKeyFactory<unsigned long long> ulonglong_KeyFactory;

template class SimpleKeyFactory<float>;
static SimpleKeyFactory<float> float_KeyFactory;

template class SimpleKeyFactory<double>;
static SimpleKeyFactory<double> double_KeyFactory;

} // detail
} // ndarray
