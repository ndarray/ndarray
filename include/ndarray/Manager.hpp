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
#ifndef NDARRAY_Manager_hpp_INCLUDED
#define NDARRAY_Manager_hpp_INCLUDED

#include <type_traits>
#include <memory>

#include "ndarray/common.hpp"
#include "ndarray/DType.hpp"
#include "ndarray/CompressedPair.hpp"

namespace ndarray {

// Base class for managers doesn't need a virtual dtor since it's always
// held by shared_ptr.
class Manager {};

namespace detail {

template <typename T, typename Owner, bool is_pod=DType<T>::is_pod>
class PrimaryManager;

template <typename T, typename Owner>
class PrimaryManager<T,Owner,true> : public Manager {
public:

    static_assert(
        DType<T>::is_direct,
        "Cannot construct manager for indirect dtype"
    );

    PrimaryManager(DType<T> dtype, Owner owner, std::size_t nbytes) :
        _owner(std::move(owner))
    {}

protected:
    Owner _owner;
};

template <typename T, typename Owner>
class PrimaryManager<T,Owner,false> : public Manager {
public:

    static_assert(
        DType<T>::is_direct,
        "Cannot construct manager for indirect dtype"
    );

    PrimaryManager(DType<T> dtype, Owner owner, std::size_t nbytes) :
        _owner(std::move(owner)),
        _dtype_and_end(std::move(dtype), _begin() + nbytes)
    {
        for (byte_t * p = _begin(); p != _end(); p += this->dtype().nbytes()) {
            this->dtype().initialize(p);
        }
    }

    ~PrimaryManager() {
        for (byte_t * p = _begin(); p != _end(); p += this->dtype().nbytes()) {
            this->dtype().destroy(p);
        }
    }

    DType<T> const & dtype() const { return _dtype_and_end.first(); }

protected:

    byte_t * _begin() const { return reinterpret_cast<byte_t*>(_owner.get()); }
    byte_t * _end() const { return _dtype_and_end.second(); }

    Owner const _owner;
    CompressedPair<DType<T>,byte_t*> const _dtype_and_end;
};

template <typename Owner>
class ExternalManager : public Manager {
public:

    ExternalManager(Owner owner) : _owner(std::move(owner)) {}

private:
    Owner _owner;
};

} // detail

template <typename T>
std::pair<byte_t*,std::shared_ptr<Manager>>
manage_new(size_t size, DType<T> dtype) {
    size_t nbytes = dtype.nbytes() * size;
    std::unique_ptr<byte_t[]> owner(new byte_t[nbytes]);
    byte_t * buffer = owner.get();
    auto manager = std::make_shared<detail::PrimaryManager<T,std::unique_ptr<byte_t[]>>>(
        std::move(dtype), std::move(owner), nbytes
    );
    return std::pair<byte_t*,std::shared_ptr<Manager>>(buffer, std::move(manager));
}

template <typename T, typename Owner>
std::shared_ptr<Manager>
manage_external(Owner owner) {
    return std::make_shared<detail::ExternalManager<Owner>>(std::move(owner));
}

} // ndarray

#endif // !NDARRAY_Manager_hpp_INCLUDED
