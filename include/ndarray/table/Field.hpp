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
#ifndef NDARRAY_table_Field_hpp_INCLUDED
#define NDARRAY_table_Field_hpp_INCLUDED

#include <string>

#include "ndarray/common.hpp"

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

    bool operator==(Field const & other) const {
        return name() == other.name()
            && doc() == other.doc()
            && unit() == other.unit();
    }

    bool operator!=(Field const & other) const {
        return !(*this == other);
    }

    std::string const & name() const { return _name; }

    virtual void set_name(std::string const & name_) { _name = name_; }

    std::string const & doc() const { return _doc; }

    virtual void set_doc(std::string const & doc_) { _doc = doc_; }

    std::string const & unit() const { return _unit; }

    virtual void set_unit(std::string const & unit_) { _unit = unit_; }

    virtual ~Field() {}

private:
    std::string _name;
    std::string _doc;
    std::string _unit;
};


} // ndarray

#endif // !NDARRAY_table_Field_hpp_INCLUDED
