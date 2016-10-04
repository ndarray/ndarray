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

#include "ndarray/common.hpp"
#include "ndarray/formatting/types.hpp"

namespace ndarray {

template <typename T>
struct FieldTraits {

    static constexpr bool is_proxy = false;

    static std::string const & type() {
        static std::string const _type = type_string<T>();
        return _type;
    }

};


class TypeError : public std::logic_error {

    std::string format(
        std::string const & desired,
        std::string const & actual
    ) {
        std::ostringstream s;
        s << "Key has type '" << actual << "'', not '" << desired << "'.";
        return s.str();
    }

public:

    explicit TypeError(char const * msg) : std::logic_error(msg) {}

    explicit TypeError(
        std::string const & desired,
        std::string const & actual
    ) :
        std::logic_error(format(desired, actual))
    {}

};


template <typename T, bool is_proxy=FieldTraits<T>::is_proxy> class Key;


class KeyBase {
public:

    KeyBase() {}

    KeyBase(KeyBase const &) = delete;

    KeyBase(KeyBase &&) = delete;

    KeyBase & operator=(KeyBase const &) = delete;

    KeyBase & operator=(KeyBase &&) = delete;

    virtual std::string const & type() const = 0;

    template <typename T>
    operator Key<T> const & () const;

    virtual ~KeyBase() {}

};


template <typename T>
class Key<T,false> : public KeyBase {
public:

    explicit Key(offset_t offset_) : _offset(offset_) {}

    Key(Key const &) = delete;

    Key(Key &&) = delete;

    Key & operator=(Key const &) = delete;

    Key & operator=(Key &&) = delete;

    virtual std::string const & type() const {
        return FieldTraits<T>::type();
    }

    offset_t offset() const { return _offset; }

private:
    offset_t _offset;
};


template <typename T>
KeyBase::operator Key<T> const & () const {
    try {
        return dynamic_cast<Key<T> const &>(*this);
    } catch (std::bad_cast &) {
        return TypeError(FieldTraits<T>::type(), this->type());
    }
}


class Field {
public:

    Field(std::string name_, std::string doc_="", std::string unit_="");

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

    SchemaField(Field field, std::unique_ptr<KeyBase> key, bool is_direct_);

    SchemaField(SchemaField const &) = delete;

    SchemaField(SchemaField &&) = default;

    SchemaField & operator=(SchemaField const &) = delete;

    SchemaField & operator=(SchemaField &&) = default;

    KeyBase const & key() const { return *_key; }

    virtual void set_name(std::string const & name_) {
        throw std::logic_error(
            "Cannot rename a SchemaField in-place; use Schema::rename."
        );
    }

    bool is_direct() const { return _is_indirect; }

private:
    bool _is_direct;
    std::unique_ptr<KeyBase> _key;
};


class SchemaWatcher {
public:

    virtual void start_append_direct(Field const & field) const = 0;

    virtual void start_insert_direct(Field const & field) const = 0;

    virtual ~SchemaWatcher() {}
};


class Schema {
public:

    typedef SchemaIter iterator;
    typedef ConstSchemaIter const_iterator;

    Schema();

    Schema(Schema const &);

    Schema(Schema &&);

    Schema & operator=(Schema const &);

    Schema & operator=(Schema &&);

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    const_iterator cbegin() const;
    const_iterator cend() const;

    std::pair<iterator,iterator> direct_only();
    std::pair<const_iterator,const_iterator> direct_only() const;

    std::pair<iterator,iterator> indirect_only();
    std::pair<const_iterator,const_iterator> indirect_only() const;

    size_t size() const { return _by_offset.size(); }
    bool empty() const { return _by_offset.empty(); }

    SchemaField & get(std::string const & name);

    SchemaField const & get(std::string const & name) const;

    SchemaField & operator[](std::string const & name) {
        return get(name);
    }

    SchemaField const & operator[](std::string const & name) const {
        return get(name);
    }

    const_iterator find(std::string const & name) const;

    iterator find(std::string const & name);

    template <typename T>
    const_iterator find(Key<T> const & key) const;

    template <typename T>
    iterator find(Key<T> const & key);

    bool contains(std::string const & name) const;

    template <typename T>
    bool contains(Key<T> const & key) const;

    template <typename T>
    Key<T> const & append(Field field) {
        static_assert(
            !FieldTraits<T>::is_proxy,
            "Proxy types cannot be added directly to a schema."
        );
        return static_cast<Key<T> const &>(
            append(FieldTraits<T>::type(), std::move(field))
        );
    }

    template <typename T>
    Key<T> const & append(
        std::string name, std::string doc="", std::string unit=""
    ) {
        return append<T>(Field(name, doc, unit));
    }

    KeyBase const & append(std::string const & type, Field field);

    KeyBase const & append(
        std::string const & type, std::string name,
        std::string doc="", std::string unit=""
    ) {
        return append(type, Field(name, doc, unit));
    }

    template <typename T>
    Key<T> const & append(std::unique_ptr<Key<T>> key, Field field);

    template <typename T>
    Key<T> const & append(
        std::unique_ptr<Key<T>> key,
        std::string name, std::string doc="", std::string unit=""
    ) {
        return append<T>(std::move(key), Field(name, doc, unit));
    }

    KeyBase const & append(std::unique_ptr<KeyBase> key, Field field);

    KeyBase const & append(
        std::unique_ptr<KeyBase> key, std::string name,
        std::string doc="", std::string unit=""
    ) {
        return append(std::move(key), Field(name, doc, unit));
    }

    template <typename T>
    std::pair<iterator,Key<T> const &> insert(iterator pos, Field field);

    template <typename T>
    std::pair<iterator,Key<T> const &> insert(
        iterator pos, std::string name, std::string doc="", std::string unit=""
    ) {
        return insert<T>(pos, Field(name, doc, unit));
    }

    iterator insert(iterator const & pos, std::string const & type, Field field);

    iterator insert(
        iterator const & pos, std::string const & type, std::string name,
        std::string doc="", std::string unit=""
    ) {
        return insert(pos, Field(name, doc, unit));
    }

    template <typename T>
    std::pair<iterator,Key<T> const &> insert(
        iterator const & pos, std::unique_ptr<Key<T>> key, Field field
    );

    template <typename T>
    std::pair<iterator,Key<T> const &> insert(
        iterator const & pos, std::unique_ptr<Key<T>> key,
        std::string name, std::string doc="", std::string unit=""
    ) {
        return insert<T>(pos, std::move(key), Field(name, doc, unit));
    }

    iterator insert(iterator const & pos, std::unique_ptr<KeyBase> key, Field field);

    iterator insert(
        iterator const & pos, std::unique_ptr<KeyBase> key, std::string name,
        std::string doc="", std::string unit=""
    ) {
        return insert(pos, std::move(key), Field(name, doc, unit));
    }

    void set(std::string const & name, Field field);

    void set(iterator iter, Field field);

    void rename(std::string const & old_name, std::string const & new_name);

    void rename(iterator iter, std::string const & new_name);

private:
    std::unordered_map<std::string,SchemaField> _by_name;
    std::vector<Field*> _by_offset;
    std::weak_ptr<SchemaWatcher> _watcher;
};


} // ndarray

#endif // !NDARRAY_table_Schema_hpp_INCLUDED