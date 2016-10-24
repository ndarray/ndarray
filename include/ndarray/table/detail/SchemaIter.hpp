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
#ifndef NDARRAY_table_detail_SchemaIter_hpp_INCLUDED
#define NDARRAY_table_detail_SchemaIter_hpp_INCLUDED

#include <memory>
#include <vector>
#include <type_traits>

#include "ndarray/common.hpp"
#include "ndarray/table/SchemaField.hpp"

namespace ndarray {

class Schema;

namespace detail {

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

    friend class ndarray::Schema;

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

} // detail
} // ndarray

#endif // !NDARRAY_table_detail_SchemaIter_hpp_INCLUDED