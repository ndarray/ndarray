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
#include <map>
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
        _key(std::move(key_))
    {}

    SchemaField(SchemaField const &) = delete;

    SchemaField(SchemaField &&) = default;

    SchemaField & operator=(SchemaField const &) = delete;

    SchemaField & operator=(SchemaField &&);

    KeyBase const & key() const { return *_key; }

    virtual void set_name(std::string const & name_) {
        throw std::logic_error(
            "Cannot rename a SchemaField in-place; use Schema::rename."
        );
    }

private:
    std::unique_ptr<KeyBase> _key;
};


class SchemaWatcher {
public:

    virtual void start_append_direct(Field const & field) const = 0;

    virtual ~SchemaWatcher() {}
};


template <typename Internal, typename Reference, typename Predicate>
class SchemaIter {
    typedef typename std::remove_reference<Reference>::type Target;
public:
    typedef SchemaItem value_type;
    typedef Reference reference;
    typedef Target * pointer;
    typedef size_t size_type;
    typedef offset_t difference_type;
    typedef std::bidirectional_iterator_tag iterator_category;

    explicit SchemaIter(
        Internal const & internal=Internal(),
        Predicate const & predicate=Predicate()
    ) :
        _internal_and_predicate(internal, predicate)
    {}

    SchemaIter(SchemaIter const &) = default;

    SchemaIter(SchemaIter &&) = default;

    template <typename U>
    SchemaIter(SchemaIter<U,Target> const & other) :
        _internal_and_predicate(other._internal_and_predicate)
    {}

    template <typename U>
    SchemaIter(SchemaIter<U,Target> && other) :
        _internal_and_predicate(std::move(other._internal_and_predicate))
    {}

    SchemaIter & operator=(SchemaIter const &) = default;

    SchemaIter & operator=(SchemaIter &&) = default;

    template <typename U>
    SchemaIter & operator=(SchemaIter<U,Target> const & other) {
        _internal_and_predicate = other._internal_and_predicate;
        return *this;
    }

    template <typename U>
    SchemaIter & operator=(SchemaIter<U,Target> && other) {
        _internal_and_predicate = std::move(other._internal_and_predicate);
        return *this;
    }

    reference operator*() const {
        return **_internal_and_predicate.first();
    }

    pointer operator->() const {
        return *_internal_and_predicate.first();
    }

    SchemaIter & operator++() {
        do {
            ++_internal_and_predicate.first();
        } while (predicate_is_false());
        return *this;
    }

    SchemaIter operator++(int) {
        SchemaIter tmp(*this);
        ++(*this);
        return tmp;
    }

    SchemaIter & operator--() {
        do {
            --_internal_and_predicate.first();
        } while (predicate_is_false());
        return *this;
    }

    SchemaIter operator--(int) {
        SchemaIter tmp(*this);
        --(*this);
        return tmp;
    }

    template <typename U>
    bool operator==(SchemaIter<U,Target> const & other) const {
        return _internal_and_predicate.first()
            == other._internal_and_predicate.first();
    }

    template <typename U>
    bool operator!=(SchemaIter<U,Target> const & other) const {
        return _internal_and_predicate.first()
            != other._internal_and_predicate.first();
    }

private:

    bool predicate_is_false() const {
        return !_internal_and_predicate.second()(_internal);
    }

    detail::CompressedPair<Internal,Predicate> _internal_and_predicate;
};


class Schema {
    typedef std::vector<SchemaField*> OrderVector;
    typedef std::unordered_map<std::string,SchemaField> NameMap;

    template <typename Internal>
    struct NullPredicate {
        bool operator()(Internal const &) const { return true; }
    };

    template <typename Internal>
    struct DirectOnlyPredicate {

        bool operator()(Internal const & internal) const {
            return internal == _bounds.first || internal == _bounds.second
                || (**internal).is_direct();
        }

        std::pair<Internal,Internal> _bounds;
    }

    template <typename Internal>
    struct IndirectOnlyPredicate {

        bool operator()(Internal const & internal) const {
            return internal == _bounds.first || internal == _bounds.second
                || !(**internal).is_direct();
        }

        std::pair<Internal,Internal> _bounds;
    }

public:

    typedef SchemaIter<OrderVector::iterator, SchemaField &,
                       NullPredicate> iterator;
    typedef SchemaIter<OrderVector::const_iterator, SchemaField const &,
                       NullPredicate> const_iterator;

    Schema();

    Schema(Schema const &);

    Schema(Schema &&);

    Schema & operator=(Schema const &);

    Schema & operator=(Schema &&);

    iterator begin() { return iterator(_by_order.begin()); }
    iterator end() { return iterator(_by_order.end()); }

    const_iterator begin() const { return const_iterator(_by_order.cbegin(); )}
    const_iterator end() const { return const_iterator(_by_order.cend(); )}

    const_iterator cbegin() const { return const_iterator(_by_order.cbegin(); )}
    const_iterator cend() const { return const_iterator(_by_order.cend(); )}

    size_t size() const { return _by_order.size(); }
    bool empty() const { return _by_order.empty(); }

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

    offset_t allocate(size_t alignment, size_t nbytes);

    NameMap _by_name;
    OrderVector _by_order;
    std::weak_ptr<SchemaWatcher> _watcher;
    size_t _next_offset;
};


} // ndarray

#endif // !NDARRAY_table_Schema_hpp_INCLUDED