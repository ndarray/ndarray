// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#ifndef LSST_NDARRAY_TABLES_Layout_h_INCLUDED
#define LSST_NDARRAY_TABLES_Layout_h_INCLUDED

#include <boost/fusion/sequence/intrinsic/size.hpp>
#include <boost/fusion/sequence/intrinsic/value_at.hpp>
#include <boost/fusion/sequence/intrinsic/at.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>

#include "lsst/ndarray/tables_fwd.h"
#include "lsst/ndarray/tables/detail/TraitsAccess.h"
#include "lsst/ndarray/tables/Field.h"
#include "lsst/ndarray/tables/detail/functional.h"

namespace lsst { namespace ndarray { namespace tables {

template <typename T>
class Layout {
public:
    typedef typename detail::TraitsAccess<T>::FieldSequence FieldSequence;
    typedef typename boost::fusion::result_of::size<FieldSequence>::type Size;

    template <int N>
    struct At {
        typedef typename boost::fusion::result_of::value_at_c<FieldSequence,N>::type Type;
    };

    template <int N>
    typename At<N>::Type const &
    operator[](Index<N> index) const {
        return boost::fusion::at_c<N>(_sequence);
    }

    template <int N>
    typename At<N>::Type &
    operator[](Index<N> index) {
        return boost::fusion::at_c<N>(_sequence);
    }

    void normalize(bool pack=false) {
        detail::SetOffsets function(pack, _bytes, _alignment);
        boost::fusion::for_each(_sequence, function);
    }

    FieldSequence const & getSequence() const { return _sequence; }

    FieldSequence & getSequence() { return _sequence; }

    int getMinStride() const { return _bytes + _bytes % _alignment; }

    int getBytes() const { return _bytes; }

    int getAlignment() const { return _alignment; }

    explicit Layout() : _sequence(), _bytes(0), _alignment(1) {}

    Layout(Layout const & other) :
        _sequence(other._sequence), _bytes(other._bytes), _alignment(other._alignment) {}

    bool operator==(Layout const & other) const {
        return _sequence == other._sequence;
    }

    bool operator!=(Layout const & other) const {
        return _sequence != other._sequence;
    }

private:
    FieldSequence _sequence;
    int _bytes;
    int _alignment;
};

}}} // namespace lsst::ndarray::tables

#endif // !LSST_NDARRAY_TABLES_Layout_h_INCLUDED
