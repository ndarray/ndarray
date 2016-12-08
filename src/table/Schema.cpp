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

#include <cassert>
#include "ndarray/table/Schema.hpp"
#include "ndarray/table/detail/KeyFactory.hpp"

namespace ndarray {

namespace {

template <typename Iter>
Iter search_name(Iter begin, Iter end, std::string const & name) {
    auto iter = std::lower_bound(
        begin,
        end,
        name,
        [](std::unique_ptr<SchemaField> const & f, std::string const & v) {
            return f->name() < v;
        }
    );
    return iter;
}

template <typename Iter>
Iter find_key(Iter begin, Iter end, KeyBase const & key) {
    return std::find_if(
        begin, end,
        [&key](SchemaField const & f) {
            return &f.key() == &key;
        }
    );
}

} // anonymous


Schema::Schema() :
    _by_name(), _first(nullptr), _last(nullptr)
{}

Schema::Schema(Schema const & other) :
    _by_name(),
    _first(nullptr),
    _last(nullptr)
{
    _by_name.reserve(other.size());
    for (auto const & field : other) {
        auto copy = field.copy();
        if (!_first) {
            _first = copy.get();
            _last = copy.get();
        } else {
            _last->_next = copy.get();
            _last = copy.get();
        }
        auto iter = search_name(_by_name.begin(), _by_name.end(), copy->name());
        // Using vector::insert here makes this copy algorithm formally
        // O(N log N), and it seems like we ought to be able to do better than
        // that.  That would probably require a temporary container of some
        // sort, though, and the operations that we're doing (copying
        // pointers) should be fast enough that it doesn't matter.
        _by_name.insert(iter, std::move(copy));
    }
}

Schema::Schema(Schema && other) :
    _by_name(),
    _first(nullptr),
    _last(nullptr)
{
    assert(other._watchers.empty()); // other has tables that depend on it!
    detail::generic_swap(_by_name, other._by_name);
    detail::generic_swap(_first, other._first);
    detail::generic_swap(_last, other._last);
}

bool Schema::operator==(Schema const & other) const {
    if (&other == this) {
        return true;
    }
    return std::equal(begin(), end(), other.begin(), other.end());
}

bool Schema::equal_keys(Schema const & other) const {
    if (&other == this) {
        return true;
    }
    return std::equal(
        begin(), end(), other.begin(), other.end(),
        [](SchemaField const & a, SchemaField const & b) {
            return a.key().equals(b.key());
        }
    );
}

SchemaField * Schema::get(std::string const & name) {
    auto iter = search_name(_by_name.begin(), _by_name.end(), name);
    if (iter == _by_name.end() || (**iter).name() != name) {
        return nullptr;
    }
    return iter->get();
}

SchemaField const * Schema::get(std::string const & name) const {
    auto iter = search_name(_by_name.begin(), _by_name.end(), name);
    if (iter == _by_name.end() || (**iter).name() != name) {
        return nullptr;
    }
    return iter->get();
}


SchemaField & Schema::operator[](std::string const & name) {
    auto iter = search_name(_by_name.begin(), _by_name.end(), name);
    if (iter == _by_name.end() || (**iter).name() != name) {
        throw std::out_of_range("Field with name '" + name + "' not found.");
    }
    return **iter;
}

SchemaField const & Schema::operator[](std::string const & name) const {
    auto iter = search_name(_by_name.begin(), _by_name.end(), name);
    if (iter == _by_name.end() || (**iter).name() != name) {
        throw std::out_of_range("Field with name '" + name + "' not found.");
    }
    return **iter;
}

Schema::iterator Schema::find(std::string const & name) {
    return iterator(get(name));
}

Schema::const_iterator Schema::find(std::string const & name) const {
    return const_iterator(get(name));
}

Schema::iterator Schema::find(KeyBase const & key) {
    return find_key(begin(), end(), key);
}

Schema::const_iterator Schema::find(KeyBase const & key) const {
    return find_key(cbegin(), cend(), key);
}

bool Schema::contains(std::string const & name) const {
    auto iter = search_name(_by_name.begin(), _by_name.end(), name);
    return iter != _by_name.end() && (**iter).name() == name;
}

bool Schema::contains(KeyBase const & key) const {
    return find_key(cbegin(), cend(), key) != cend();
}


KeyBase const & Schema::append(
    std::string const & type,
    Field field,
    void const * dtype
) {
    auto iter = search_name(_by_name.begin(), _by_name.end(), field.name());
    if (iter != _by_name.end() && (**iter).name() == field.name()) {
        throw std::invalid_argument(
            "Field with name '" + field.name() + "' already present in schema."
        );
    }
    for (auto watcher : _watchers) {
        (void)watcher; // TODO
    }
    auto key = detail::KeyFactory::invoke(_by_name.size(), type, dtype);
    std::unique_ptr<SchemaField> new_field(
        new SchemaField(std::move(field), std::move(key))
    );
    if (!_first) {
        _first = new_field.get();
        _last = new_field.get();
    } else {
        _last->_next = new_field.get();
        _last = new_field.get();
    }
    auto result = _by_name.insert(iter, std::move(new_field));
    return (**result).key();
}

void Schema::set(std::string const & name, Field field) {
    auto iter = search_name(_by_name.begin(), _by_name.end(), name);
    if (iter != _by_name.end() && (**iter).name() == name) {
        throw std::out_of_range("Field with name '" + name + "' not found.");
    }
    if (name == field.name()) {
        // easy case: doesn't require removing and reinserting into _by_name
        **iter = std::move(field);
    } else {
        auto new_iter = search_name(_by_name.begin(), _by_name.end(), field.name());
        if (new_iter != _by_name.end() && (**new_iter).name() == field.name()) {
            throw std::invalid_argument(
                "A field with name '" + field.name() + "' already exists."
            );
        }
        if (new_iter == iter) {
            // name has changed, but it's in the same place alphabetically
            **iter = std::move(field);
        } else {
            // extract the old SchemaField unique_ptr from _by_name, and remove
            auto field_ptr = std::move(*iter);
            if (name < field.name()) {
                // We're moving this field to the right, so we shift the ones
                // in between the old and new positions to the left.
                auto start(iter);
                ++start;
                std::move(start, new_iter, iter);
            } else {
                // We're moving this field to the left, so we shift the ones
                // in between the new and old positions to the right.
                auto finish(new_iter);
                --finish;
                std::move_backward(iter, finish, new_iter);
            }
            // Now we can insert the field back in at the spot that opened up
            *iter = std::move(field_ptr);
        }
    }
}

void Schema::set(iterator iter, Field field) {
    if (iter == end()) {
        throw std::out_of_range(
            "Invalid past-the-end iterator passed to Schema::set()"
        );
    }
    set(iter->name(), std::move(field));
}

void Schema::rename(
    std::string const & old_name,
    std::string const & new_name
) {
    auto iter = search_name(_by_name.begin(), _by_name.end(), old_name);
    if (iter != _by_name.end() && (**iter).name() == old_name) {
        throw std::out_of_range(
            "Field with name '" + old_name + "' not found."
        );
    }
    auto new_iter = search_name(_by_name.begin(), _by_name.end(), new_name);
    if (new_iter != _by_name.end() && (**new_iter).name() == new_name) {
        throw std::invalid_argument(
            "A field with name '" + new_name + "' already exists."
        );
    }
    Field field(**iter);
    field.set_name(new_name);
    if (new_iter == iter) {
        // name has changed, but it's in the same place alphabetically
        **iter = std::move(field);
    } else {
        // extract the old SchemaField unique_ptr from _by_name, and remove
        auto field_ptr = std::move(*iter);
        if (old_name < new_name) {
            // We're moving this field to the right, so we shift the ones
            // in between the old and new positions to the left.
            auto start(iter);
            ++start;
            std::move(start, new_iter, iter);
        } else {
            // We're moving this field to the left, so we shift the ones
            // in between the new and old positions to the right.
            auto finish(new_iter);
            --finish;
            std::move_backward(iter, finish, new_iter);
        }
        // Now we can insert the field back in at the spot that opened up
        *iter = std::move(field_ptr);
    }
}

void Schema::rename(iterator iter, std::string const & new_name) {
    if (iter == end()) {
        throw std::out_of_range(
            "Invalid past-the-end iterator passed to Schema::rename()"
        );
    }
    rename(iter->name(), new_name);
}


} // ndarray
