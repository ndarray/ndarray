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
#ifndef NDARRAY_managers_hpp_INCLUDED
#define NDARRAY_managers_hpp_INCLUDED

#include <type_traits>
#include <memory>

#include "ndarray/common.hpp"
#include "ndarray/dtype.hpp"

namespace ndarray {

// Base class for managers doesn't need a virtual dtor since it's always
// held by shared_ptr.
class manager {};

namespace detail {

template <typename T, typename Owner, typename IsPOD=DType<T>::is_pod>
class primary_manager;

template <typename T, typename Owner>
class primary_manager<T,Owner,std::true_type> : public manager {
public:

    primary_manager(dtype<T> dtype, Owner owner, std::size_t nbytes) :
        _owner(std::move(owner))
    {}

protected:
    Owner _owner;
};

template <typename T, typename Owner>
class primary_manager<T,Owner,std::false_type> : public manager {
public:

    primary_manager(dtype<T> dtype, Owner owner, std::size_t nbytes) :
        _owner(std::move(owner)),
        _holder(std::move(dtype), _begin() + nbytes)
    {
        for (byte_t * p = _begin(); p != _end(); p += this->dtype().nbytes()) {
            this->dtype().initialize(p);
        }
    }

    ~primary_manager() {
        for (Byte * p = _begin(); p != _end(); p += this->dtype().nbytes()) {
            this->dtype().destroy(p);
        }
    }

protected:

    byte_t * _begin() const { return reintepret_cast<byte_t*>(_owner.get()); }
    byte_t * _end() const { return _holder.other(); }

    Owner const _owner;
    dtype_holder<T,byte_t *> const _holder;
};

template <typename Owner>
class external_manager : public manager {
public:

    Owner(Owner owner) : _owner(std::move(owner)) {}

private:
    Owner _owner;
};

} // detail

template <typename T>
std::pair<byte_t*,std::shared_ptr<manager>>
manage_new(size_t size, dtype<T> dtype) {
    size_t nbytes = dtype.nbytes() * size;
    std::unique_ptr<byte_t[]> owner(new byte_t[nbytes]);
    auto manager = std::make_shared<detail::primary_manager<T,std::unique_ptr<byte_t[]>>>(
        std::move(dtype), std::move(owner), nbytes
    );
    return std::make_pair(owner.get(), std::move(manager));
}

template <typename T, typename Owner>
std::shared_ptr<manager>
manage_external(Owner owner) {
    return std::make_shared<detail::external_manager<Owner>>(std::move(owner));
}

} // ndarray

#endif // !NDARRAY_managers_hpp_INCLUDED
