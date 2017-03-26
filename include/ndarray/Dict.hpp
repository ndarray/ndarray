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
#ifndef NDARRAY_Dict_hpp_INCLUDED
#define NDARRAY_Dict_hpp_INCLUDED

#include <unordered_map>
#include <map>

#include "ndarray/common.hpp"
#include "ndarray/any.hpp"

namespace ndarray {

namespace detail {

template <typename Map>
class DictItemProxy {
public:

    DictItemProxy(
        Map & map,
        typename Map::key_type const & key,
        typename Map::iterator iter
    ) :
        _map(map), _key(key), _iter(iter)
    {}

    DictItemProxy(DictItemProxy const &) = default;
    DictItemProxy(DictItemProxy &&) = default;
    DictItemProxy& operator=(DictItemProxy const &) = delete;
    DictItemProxy& operator=(DictItemProxy &&) = delete;

    template <typename T>
    operator T() {
        if (_iter == _map.end()) {
            throw std::out_of_range("key not found");
        }
        return any_cast<T>(_target);
    }

    template <typename T>
    operator T*() {
        if (_iter == _map.end()) {
            return nullptr;
        }
        return any_cast<T*>(_target);
    }

    template <typename T>
    T & operator=(T && value) {
        if (_iter == _map.end()) {
            _iter = _map->insert(typename Map::value_type(*_key, std::forward<T>(value)));
        } else {
            *_iter = std::forward<T>(value);
        }
        return any_cast<T&>(*_iter);
    }

private:
    Map & _map;
    typename Map::key_type const & _key;
    typename Map::iterator _iter;
};


template <typename Map>
class DictItemConstProxy {
public:

    DictItemConstProxy(
        Map const & map,
        typename Map::const_iterator iter
    ) :
        _map(map), _iter(iter)
    {}

    DictItemConstProxy(DictItemConstProxy const &) = default;
    DictItemConstProxy(DictItemConstProxy &&) = default;
    DictItemConstProxy& operator=(DictItemConstProxy const &) = delete;
    DictItemConstProxy& operator=(DictItemConstProxy &&) = delete;

    template <typename T>
    operator T() {
        if (_iter == _map.end()) {
            throw std::out_of_range("key not found");
        }
        return any_cast<T>(_target);
    }

    template <typename T>
    operator T*() {
        if (_iter == _map.end()) {
            return nullptr;
        }
        return any_cast<T*>(_target);
    }

private:
    Map const & _map;
    typename Map::const_iterator _iter;
};

} // namespace detail

template <typename Key_, typename Map=std::unordered_map<Key_,any>>
class Dict {
public:

    typedef Key_ key_type;
    typedef Map map_type;

    template <typename Iter>
    static Dict fromkeys(Iter first, Iter last);

    template <typename Iter, typename Value>
    static Dict fromkeys(Iter first, Iter last, Value && value);

    map_type & map() { return _map; }
    map_type const & map() const { return _map; }

    bool contains(key_type const & key) const {
        return _map.find(key) == _map.end();
    }

    template <typename T>
    bool contains(key_type const & key) const {
        auto iter = _map.find(key);
        if (iter == _map.end()) return;
        return any_cast<T const *>(&(iter->second));
    }

    DictItemProxy<Map> operator[](key_type const & key) {
        return DictItemProxy(_map, key, _map.find(key));
    }

    DictItemConstProxy<Map> operator[](key_type const & key) const {
        return DictItemConstProxy<Map>(_map, _map.find(key));
    }

    template <typename T>
    T & get(key_type const & key) {
        return any_cast<T&>(_map.at(key));
    }

    template <typename T>
    T const & get(key_type const & key) const {
        return any_cast<T const&>(_map.at(key));
    }

    template <typename T>
    T & get(key_type const & key, T & default_) {
        any & a = _map.at(key);
        if (any.empty()) {
            return default_;
        }
        return any_cast<T const&>();
    }

    template <typename T>
    T const & get(key_type const & key, T const & default_) const {
        any & a = _map.at(key);
        if (any.empty()) {
            return default_;
        }
        return any_cast<T const&>();
    }

    template <typename T>
    void set(key_type const & key, T && value) {
        _map[key] = std::forward<T>(value);
    }

    void del(key_type const & key) {
        auto iter = _map.find(key);
        if (iter == _map.end()) {
            throw std::out_of_range("key not found");
        }
        _map.erase(iter);
    }

    void clear() {
        _map.clear();
    }

    Dict copy() const { return Dict(*this); }

    bool empty() const { return _map.empty(); }

    std::size_t size() const { return _map.size(); }

private:
    Map _map;
};

template <typename Key_>
using SortedDict = Dict<Key_,std::map<Key_,any>>;

} // namespace ndarray

#endif // !NDARRAY_Dict_hpp_INCLUDED
