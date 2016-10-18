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
#include "ndarray/table/Schema.hpp"
#include "ndarray/table/detail/KeyFactory.hpp"

namespace ndarray {

SchemaField & SchemaField::operator=(SchemaField && other) {
    if (name() != other.name()) {
        throw std::logic_error(
            "Cannot rename a SchemaField in-place; use Schema::rename."
        );
    }
    _key = std::move(other._key);
    Field::operator=(std::move(other));
    return *this;
}


Schema::Schema() :
    _ranges({OffsetRange(0, -1)})
{}

Schema::Schema(Schema const & other) :
    _by_name(other._by_name),
    _by_order(other._by_order),
    _watcher()
{
    for (auto & ptr : _by_order) {
        ptr = _by_name.find(ptr->name())->second;
    }
}

Schema::Schema(Schema && other) :
    _by_name(std::move(other._by_name)),
    _by_order(std::move(other._by_order)),
    _watcher()
{}

Schema & Schema::operator=(Schema const & other) {
    if (&other != this) {
        if (!_watcher) {
            _by_name = other._by_name;
            _by_order = other._by_order;
        } else {
            // TODO
        }
    }
    return *this;
}

Schema & Schema::operator=(Schema && other) {
    if (!_watcher && !_other._watcher) {
        _by_name = std::move(other._by_name);
        _by_order = std::move(other._by_order);
    } else {
        // TODO
    }
    return *this;
}

SchemaField * Schema::get(std::string const & name) {
    NameMap::iterator i = _by_name.find(name);
    if (i == _by_name.end()) {
        return nullptr;
    }
    return &i->second;
}

SchemaField const * Schema::get(std::string const & name) const {
    NameMap::const_iterator i = _by_name.find(name);
    if (i == _by_name.end()) {
        return nullptr;
    }
    return &i->second;
}

SchemaField & Schema::operator[](std::string const & name) {
    SchemaField * r = get(name);
    if (!r) {
        throw std::runtime_error("Field with name '" + name + "' not found.");
    }
    return *r;
}

SchemaField const & operator[](std::string const & name) const {
    SchemaField const * r = get(name);
    if (!r) {
        throw std::runtime_error("Field with name '" + name + "' not found.");
    }
    return *r;
}

Schema::iterator Schema::find(std::string const & name) {
    auto internal = std::find_if(
        _by_order.begin(), _by_order.end(),
        [&name](SchemaField const * f) {
            return f->name() == name;
        }
    );
    return iterator(internal);
}

Schema::const_iterator Schema::find(std::string const & name) const {
    auto internal = std::find_if(
        _by_order.begin(), _by_order.end(),
        [&name](SchemaField const * f) {
            return f->name() == name;
        }
    );
    return const_iterator(internal);
}

Schema::iterator Schema::find(KeyBase const & key) {
    auto internal = std::find_if(
        _by_order.begin(), _by_order.end(),
        [&name](SchemaField const * f) {
            return &f->key() == &key;
        }
    );
    return iterator(internal);
}

Schema::const_iterator Schema::find(KeyBase const & key) const {
    auto internal = std::find_if(
        _by_order.begin(), _by_order.end(),
        [&name](SchemaField const * f) {
            return &f->key() == &key;
        }
    );
    return const_iterator(internal);
}

bool Schema::contains(std::string const & name) const {
    return std::find_if(
        _by_order.begin(), _by_order.end(),
        [&name](SchemaField const * f) {
            return f->name() == name;
        }
    ) != _by_order.end();
}

bool Schema::contains(KeyBase const & key) const {
    return std::find_if(
        _by_order.begin(), _by_order.end(),
        [&name](SchemaField const * f) {
            return &f->key() == &key;
        }
    ) != _by_order.end();
}


KeyBase const & Schema::append(
    std::string const & type,
    Field field,
    void const * dtype
) {
    auto watcher = _watcher.lock();
    if (watcher) {
        watcher->start_append_direct(field);
    }
    auto key = detail::KeyFactory::invoke(_next_offset, type, dtype);
    std::string name = field.name();
    auto result = _by_name.emplace(
        name,
        SchemaField(std::move(field), std::move(key))
    );
    if (!result.second) {
        throw std::runtime_error(
            "Field with name '" + name + "' already present in schema."
        );
    }
    _by_order.push_back(&result.first.second); // pointer to SchemaField.
    return result.first.second.key();
}

void Schema::set(std::string const & name, Field field) {
    if (name == field.name()) {
        // easy case: doesn't require removing and reinserting into _by_name
        SchemaField * f = get(name);
        *f = std::move(field);
    } else {
        // TODO
    }
}

void Schema::set(iterator iter, Field field) {
    if (iter == end()) {
        throw std::out_of_range(
            "Invalid past-the-end iterator passed to Schema::set()"
        );
    }
    if (iter->name() == field.name()) {
        // easy case: doesn't require removing and reinserting into _by_name.
        *iter = std::move(field);
    } else {
        // TODO
    }
}

void Schema::rename(std::string const & old_name, std::string const & new_name) {
    // TODO
}

void Schema::rename(iterator iter, std::string const & new_name) {
    // TODO
}



} // ndarray
