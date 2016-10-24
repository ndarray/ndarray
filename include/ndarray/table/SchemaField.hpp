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
#ifndef NDARRAY_table_SchemaField_hpp_INCLUDED
#define NDARRAY_table_SchemaField_hpp_INCLUDED

#include <memory>
#include <vector>

#include "ndarray/common.hpp"
#include "ndarray/table/Key.hpp"
#include "ndarray/table/Field.hpp"

namespace ndarray {


class SchemaField : public Field {
public:

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

    SchemaField(Field field, std::unique_ptr<KeyBase> key_) :
        Field(std::move(field)),
        _key(std::move(key_)),
        _next(nullptr)
    {}

    std::unique_ptr<SchemaField> copy() const;

    std::unique_ptr<KeyBase> _key;
    SchemaField * _next;
};


} // ndarray

#endif // !NDARRAY_table_SchemaField_hpp_INCLUDED