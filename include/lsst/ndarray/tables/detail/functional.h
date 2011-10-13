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
#ifndef LSST_NDARRAY_TABLES_DETAIL_functional_h_INCLUDED
#define LSST_NDARRAY_TABLES_DETAIL_functional_h_INCLUDED

#include "lsst/ndarray/tables_fwd.h"
#include "lsst/ndarray.h"
#include "lsst/ndarray/tables/detail/TraitsAccess.h"
#include "lsst/ndarray/tables/detail/FieldInfo.h"

#include <boost/fusion/container/generation/make_vector.hpp>

#include <stdexcept>

namespace lsst { namespace ndarray { namespace tables { namespace detail {

struct SetOffsets {

    template <typename T, int N>
    void operator()(Field<T,N> & field) const {
        int element_size = sizeof(T);
        alignment = std::max(element_size, alignment);
        if (field.offset < 0) {
            if (!pack) bytes += bytes % element_size;
            field.offset = bytes;
        } else {
            if (field.offset < bytes) {
                throw std::logic_error("Field offsets are not compatible.");
            }
        }
        bytes = field.offset + element_size * ((N == 0) ? 1 : field.shape.product());
    }

    explicit SetOffsets(bool pack_, int & bytes_, int & alignment_) :
        pack(pack_), bytes(bytes_), alignment(alignment_)
    {
        bytes = 0;
        alignment = 1;
    }

    bool const pack;
    mutable int & bytes;
    mutable int & alignment;
};

struct MakeColumns {

    typedef unsigned char Raw;

    template <typename> struct result;

    template <typename Func, typename U>
    struct result<Func(U)> {
        typedef typename FieldInfo<U>::ColumnValue type;
    };

    template <typename U>
    typename FieldInfo<U>::ColumnValue
    operator()(U const & field) const {
        typedef typename lsst::ndarray::detail::ArrayAccess< typename FieldInfo<U>::ColumnValue > Access;
        typename Access::Core::Ptr core = Access::Core::create(
            lsst::ndarray::concatenate(size, field.shape),
            manager
        );
        core->setStride(stride / sizeof(typename Access::Element));
        return Access::construct(reinterpret_cast<typename Access::Element*>(raw + field.offset), core);
    }

    explicit MakeColumns(int size_, int stride_, Raw * raw_, Manager::Ptr const & manager_) :
        size(size_), stride(stride_), raw(raw_), manager(manager_) {}

    int size;
    int stride;
    Raw * raw;
    Manager::Ptr manager;
};

template <typename Dim>
struct ViewColumns {

    template <typename> struct result;

    template <typename Func, typename T, int N>
    struct result<Func(Array<T,N+1,N> const &)> {
        typedef Array<T,N+1,N> type;
    };

    template <typename Func, typename T, int N>
    struct result<Func(Array<T,N+1,N> &)> {
        typedef Array<T,N+1,N> type;
    };

    template <typename Func, typename T, int N>
    struct result<Func(Array<T,N+1,N>)> {
        typedef Array<T,N+1,N> type;
    };

    template <typename T, int N>
    Array<T,N+1,N> operator()(Array<T,N+1,N> const & x) const {
        return Array<T,N+1,N>(x[def]);
    }

    explicit ViewColumns(Dim const & dim) : def(boost::fusion::make_vector(dim)) {}

    View< boost::fusion::vector<Dim> > def;
};

template <typename Dim>
ViewColumns<Dim> makeViewColumns(Dim const & dim) {
    return ViewColumns<Dim>(dim);
}

}}}} // namespace lsst::ndarray::tables::detail

#endif // !LSST_NDARRAY_TABLES_DETAIL_functional_h_INCLUDED
