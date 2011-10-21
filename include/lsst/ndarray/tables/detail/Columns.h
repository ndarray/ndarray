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
#ifndef LSST_NDARRAY_TABLES_DETAIL_Columns_h_INCLUDED
#define LSST_NDARRAY_TABLES_DETAIL_Columns_h_INCLUDED

#include <boost/intrusive_ptr.hpp>
#include <boost/fusion/algorithm/transformation/transform.hpp>

#include "lsst/ndarray/tables/detail/functional.h"
#include "lsst/ndarray/tables/Layout.h"

namespace lsst { namespace ndarray { namespace tables { namespace detail {

template <typename T>
class Columns {
    BOOST_STATIC_ASSERT( !boost::is_const<T>::value );
public:

    typedef boost::intrusive_ptr<Columns> Ptr;

    typedef unsigned char Raw;

    typedef typename TraitsAccess<T>::FieldSequence FieldSequence;

    typedef typename boost::fusion::result_of::as_vector<
        typename boost::fusion::result_of::transform<FieldSequence,MakeColumns>::type
        >::type ColumnSequence;

    template <int N>
    struct At {
        typedef typename TraitsAccess<T>::template Fields<N>::Info::ColumnValue Type;
    };

    template <int N>
    typename At<N>::Type operator[](Index<N> index) const {
        return boost::fusion::at_c<N>(_sequence);
    }

    Ptr index(lsst::ndarray::index::Range const & dim) const {
        return Ptr(new Columns(*this, dim));
    }

    Ptr index(lsst::ndarray::index::Slice const & dim) const {
        return Ptr(new Columns(*this, dim));
    }

    int getSize() const { return _size; }

    int getStride() const { return _stride; }

    Raw * getRaw() const { return _raw; }

    Manager::Ptr getManager() const { return _manager; }

    Layout<T> const & getLayout() const { return _layout; }

    ColumnSequence const & getSequence() const { return _sequence; }

    static Ptr create(
        int size, int stride, Raw * raw, Manager::Ptr const & manager, Layout<T> const & layout
    ) {
        return Ptr(new Columns(size, stride, raw, manager, layout));
    }

    friend inline void intrusive_ptr_add_ref(Columns const * self) {
        ++self->_rc;
    }
 
    friend inline void intrusive_ptr_release(Columns const * self) {
        if ((--self->_rc)==0) delete self;
    }

private:

    static FieldSequence const & getNormalizedFieldSequence(Layout<T> & layout, int & stride) {
        layout.normalize();
        if (stride < 0) {
            stride = layout.getMinStride();
        } else {
            if (stride < layout.getBytes()) {
                throw std::logic_error("Table stride is smaller than layout size.");
            }
            if (stride % layout.getAlignment() != 0) {
                throw std::logic_error("Table stride is not evenly disible by its maximum element size.");
            }
        }
        return layout.getSequence();
    }

    Columns(Columns const & other, lsst::ndarray::index::Range const & dim) :
        _rc(1), _size(dim.stop - dim.start), _stride(other.getStride()),
        _raw(other.getRaw() + dim.start * other.getStride()),
        _manager(other.getManager()), _layout(other.getLayout()),
        _sequence(boost::fusion::transform(other.getSequence(), makeViewColumns(dim)))
    {}

    Columns(Columns const & other, lsst::ndarray::index::Slice const & dim) :
        _rc(1), _size(dim.stop - dim.start), _stride(other.getStride()),
        _raw(other.getRaw() + dim.start * other.getStride()),
        _manager(other.getManager()), _layout(other.getLayout()),
        _sequence(boost::fusion::transform(other.getSequence(), makeViewColumns(dim)))
    {}

    Columns(int size, int stride, Raw * raw, Manager::Ptr const & manager, Layout<T> const & layout) :
        _rc(1), _size(size), _stride(stride), _raw(raw), _manager(manager), _layout(layout),
        _sequence(
            boost::fusion::transform(
                getNormalizedFieldSequence(_layout, _stride),
                MakeColumns(_size, _stride, _raw, _manager)
            )
        )
    {}

    mutable int _rc;
    int _size;
    int _stride;
    Raw * _raw;
    Manager::Ptr _manager;
    Layout<T> _layout;
    ColumnSequence _sequence;
};

}}}} // namespace lsst::ndarray::tables::detail

#endif // !LSST_NDARRAY_TABLES_DETAIL_Columns_h_INCLUDED
