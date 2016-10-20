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

#include "ndarray/common.hpp"
#include "ndarray/table/Key.hpp"

namespace ndarray {


class Field {
public:

    explicit Field(
        std::string name_,
        std::string doc_="",
        std::string unit_=""
    ) :
        _name(std::move(name_)),
        _doc(std::move(doc_)),
        _unit(std::move(unit_))
    {}

    Field(Field const &) = default;

    Field(Field &&) = default;

    Field & operator=(Field const &) = default;

    Field & operator=(Field &&) = default;

    std::string const & name() const { return _name; }

    virtual void set_name(std::string const & name_) { _name = name_; }

    std::string const & doc() const { return _doc; }

    void set_doc(std::string const & doc_) { _doc = doc_; }

    std::string const & unit() const { return _unit; }

    void set_unit(std::string const & unit_) { _unit = unit_; }

    virtual ~Field() {}

private:
    std::string _name;
    std::string _doc;
    std::string _unit;
};


class SchemaField : public Field {
public:

    SchemaField(Field field, std::unique_ptr<KeyBase> key_) :
        Field(std::move(field)),
        _key(std::move(key_)),
        _next(nullptr)
    {}

    SchemaField(SchemaField const &) = delete;

    SchemaField(SchemaField &&) = default;

    SchemaField & operator=(SchemaField const &) = delete;

    SchemaField & operator=(SchemaField &&);

    SchemaField & operator=(Field const &);

    SchemaField & operator=(Field &&);

    KeyBase const & key() const { return *_key; }

    virtual void set_name(std::string const & name_) {
        throw std::logic_error(
            "Cannot rename a SchemaField in-place; use Schema::rename."
        );
    }

private:
    friend class Schema;

    std::unique_ptr<SchemaField> copy() const;

    std::unique_ptr<KeyBase> _key;
    SchemaField * _next;
};


class SchemaWatcher {
public:
    virtual ~SchemaWatcher() {}
};

template <typename Target, typename Incrementer>
class SchemaIter {
public:
    typedef typename std::remove_const<Target>::type value_type;
    typedef Target & reference;
    typedef Target * pointer;
    typedef size_t size_type;
    typedef offset_t difference_type;
    typedef std::forward_iterator_tag iterator_category;

    SchemaIter() : _current_and_incrementer(nullptr, Incrementer()) {}

    SchemaIter(SchemaIter const &) = default;
    SchemaIter(SchemaIter &&) = default;

    template <typename Target2>
    SchemaIter(SchemaIter<Target2,Incrementer> const & other) :
        _current_and_incrementer(other._current_and_incrementer) {}

    SchemaIter & operator=(SchemaIter const &) = default;
    SchemaIter & operator=(SchemaIter &&) = default;

    template <typename Target2>
    SchemaIter & operator=(SchemaIter<Target2,Incrementer> const & other) {
        _current_and_incrementer.first()
            = other._current_and_incrementer.first();
        return *this;
    }

    void swap(SchemaIter & other) {
        other._current_and_incrementer.swap(_current_and_incrementer);
    }

    reference operator*() const {
        return *_current_and_incrementer.first();
    }

    pointer operator->() const {
        return _current_and_incrementer.first();
    }

    pointer get() const {
        return _current_and_incrementer.first();
    }

    SchemaIter & operator++() {
        _current_and_incrementer.second()(_current_and_incrementer.first());
        return *this;
    }

    SchemaIter operator++(int) {
        SchemaIter tmp(*this);
        ++(*this);
        return tmp;
    }

    template <typename Target2, typename Incrementer2>
    bool operator==(SchemaIter<Target2,Incrementer2> const & other) const {
        return other._current_and_incrementer.first() ==
            _current_and_incrementer.first();
    }

    template <typename Target2, typename Incrementer2>
    bool operator!=(SchemaIter<Target2,Incrementer2> const & other) const {
        return other._current_and_incrementer.first() !=
            _current_and_incrementer.first();
    }

private:

    friend class Schema;

    explicit SchemaIter(pointer first, Incrementer incrementer=Incrementer()) :
        _current_and_incrementer(first, std::move(incrementer))
    {}

    CompressedPair<pointer,Incrementer> _current_and_incrementer;
};

template <typename Target, typename Incrementer>
inline void swap(
    SchemaIter<Target,Incrementer> & a,
    SchemaIter<Target,Incrementer> & b
) {
    a.swap(b);
}


class Schema {

    struct StandardIncrementer {

        template <typename Target>
        void operator()(Target * & target) const {
            target = target->_next;
        }

    };

public:

    typedef SchemaIter<SchemaField,StandardIncrementer> iterator;
    typedef SchemaIter<SchemaField const,StandardIncrementer> const_iterator;

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
    // their memory.  But each SchemaField is also a linked list node that
    // remembers the order in which they were added.  This gives us a
    // container that remembers its insertion order while also supporting
    // O(log N) name lookup without a complex tree or hash data structure
    std::vector<std::unique_ptr<SchemaField>> _by_name;
    SchemaField * _first;
    SchemaField * _last;
    size_t _next_offset;
    std::weak_ptr<SchemaWatcher> _watcher;
};


} // ndarray

#endif // !NDARRAY_table_Schema_hpp_INCLUDED