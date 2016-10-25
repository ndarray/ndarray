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
#ifndef NDARRAY_table_Schema_hpp_INCLUDED
#define NDARRAY_table_Schema_hpp_INCLUDED

#include <memory>
#include <vector>
#include <type_traits>

#include "ndarray/common.hpp"
#include "ndarray/table/SchemaField.hpp"
#include "ndarray/table/detail/SchemaIter.hpp"
#include "ndarray/table/SchemaWatcher.hpp"

namespace ndarray {


class Schema {

    struct StandardIncrementer {

        template <typename Target>
        void operator()(Target * & target) const {
            target = target->_next;
        }

    };

public:

    typedef detail::SchemaIter<SchemaField,StandardIncrementer>
        iterator;
    typedef detail::SchemaIter<SchemaField const,StandardIncrementer>
        const_iterator;

    Schema();

    Schema(Schema const &);

    Schema(Schema &&);

    Schema & operator=(Schema const &);

    Schema & operator=(Schema &&);

    void swap(Schema & other);

    iterator begin() { return iterator(_first); }
    iterator end() { return iterator(nullptr); }

    const_iterator begin() const { return const_iterator(_first); }
    const_iterator end() const { return const_iterator(nullptr); }

    const_iterator cbegin() const { return const_iterator(_first); }
    const_iterator cend() const { return const_iterator(nullptr); }

    size_t size() const { return _by_name.size(); }
    bool empty() const { return _by_name.empty(); }

    size_t alignment() const { return _alignment; }
    size_t nbytes() const { return _nbytes; }

    bool operator==(Schema const & other) const;
    bool operator!=(Schema const & other) const { return !(*this == other); }

    bool equal_keys(Schema const & other) const;

    SchemaField * get(std::string const & name);

    SchemaField const * get(std::string const & name) const;

    SchemaField & operator[](std::string const & name);

    SchemaField const & operator[](std::string const & name) const;

    iterator find(std::string const & name);

    const_iterator find(std::string const & name) const;

    iterator find(KeyBase const & key);

    const_iterator find(KeyBase const & key) const;

    bool contains(std::string const & name) const;

    bool contains(KeyBase const & key) const;

    template <typename T>
    Key<T> const & append(Field field, DType<T> const & dtype=DType<T>()) {
        return static_cast<Key<T> const &>(
            append(DType<T>::name(), std::move(field), &dtype)
        );
    }

    template <typename T>
    Key<T> const & append(
        std::string name, std::string doc="", std::string unit="",
        DType<T> const & dtype=DType<T>()
    ) {
        return append<T>(Field(name, doc, unit), dtype);
    }

    KeyBase const & append(
        std::string const & type,
        Field field,
        void const * dtype=nullptr
    );

    KeyBase const & append(
        std::string const & type,
        std::string name, std::string doc="", std::string unit="",
        void const * dtype=nullptr
    ) {
        return append(type, Field(name, doc, unit), dtype);
    }

    // TODO: append indirect fields

    // TODO: insert keys

    void set(std::string const & name, Field field);

    void set(iterator iter, Field field);

    void rename(std::string const & old_name, std::string const & new_name);

    void rename(iterator iter, std::string const & new_name);

private:

    // SchemaFields are ordered by name in the _by_name vector, which manages
    // their lifetime.  But each SchemaField is also a linked list node that
    // remembers the order in which they were added.  This gives us a
    // container that remembers its insertion order while also supporting
    // O(log N) name, lookup without a complex tree or hash data structure
    std::vector<std::unique_ptr<SchemaField>> _by_name;
    SchemaField * _first;
    SchemaField * _last;
    size_t _nbytes;
    size_t _alignment;
    std::weak_ptr<SchemaWatcher> _watcher;
};


} // ndarray

#endif // !NDARRAY_table_Schema_hpp_INCLUDED