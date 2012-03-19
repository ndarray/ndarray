// -*- c++ -*-
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
#ifndef NDARRAY_views_h_INCLUDED
#define NDARRAY_views_h_INCLUDED

/** 
 *  \file ndarray/views.h @brief Public interface for arbitrary views into arrays.
 */

#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/mpl.hpp>

namespace ndarray {
namespace index {

/** 
 *  @brief Simple structure defining a noncontiguous range of indices.
 */
struct Slice {
    int start;
    int stop;
    int step;

    Slice(int start_, int stop_, int step_) : start(start_), stop(stop_), step(step_) {}

    int computeSize() const { return (step > 1) ? (stop - start + 1) / step : stop - start; }
};

/**
 *  @brief Simple structure defining a contiguous range of indices.
 */
struct Range {
    int start;
    int stop;

    Range(int start_, int stop_) : start(start_), stop(stop_) {}
};

/**
 *  @brief Empty structure marking a view of an entire dimension.
 */
struct Full {};

/**
 *  @brief Structure marking a single element of a dimension.
 */
struct Scalar {
    int n;

    explicit Scalar(int n_) : n(n_) {}
};

} // namespace index

/** 
 *  @brief A template meta-sequence that defines an arbitrary view into an unspecified array. 
 *
 *  A View is constructed from a call to the global view() function
 *  and subsequent chained calls to operator().
 */
template <typename Seq_ = boost::fusion::vector<> >
struct View {
    typedef Seq_ Sequence; ///< A boost::fusion sequence type
    Sequence _seq; ///< A boost::fusion sequence of index objects.

    explicit View(Sequence seq) : _seq(seq) {}

    template <typename OtherSequence>
    explicit View(OtherSequence const & other) : _seq(other) {}

    template <typename OtherSequence>
    View(View<OtherSequence> const & other) : _seq(other._seq) {}

    /// @brief The View that results from chaining an full dimension index <b><tt>()</tt></b> to this.
    typedef View<typename boost::fusion::result_of::push_back<Sequence const,index::Full>::type> Full;

    /// @brief The View that results from chaining a range <b><tt>(start,stop)</tt></b> to this.
    typedef View<typename boost::fusion::result_of::push_back<Sequence const,index::Range>::type> Range;

    /// @brief The View that results from chaining a slice <b><tt>(start,stop,step)</tt></b> to this.
    typedef View<typename boost::fusion::result_of::push_back<Sequence const,index::Slice>::type> Slice;

    /// @brief The View that results from chaining a scalar <b><tt>(n)</tt></b> to this.
    typedef View<typename boost::fusion::result_of::push_back<Sequence const,index::Scalar>::type> Scalar;

    /// @brief Chain the full next dimension to this.
    Full operator()() const { return Full(boost::fusion::push_back(_seq, index::Full())); }
    
    /// @brief Chain a contiguous range of the next dimension to this.
    Range operator()(int start, int stop) const {
        return Range(boost::fusion::push_back(_seq, index::Range(start, stop)));
    }

    /// @brief Chain a noncontiguous slice of the next dimension to this.
    Slice operator()(int start, int stop, int step) const {
        return Slice(boost::fusion::push_back(_seq, index::Slice(start, stop, step)));
    }

    /// @brief Chain a single element of the next dimension to this.
    Scalar operator()(int n) const {
        return Scalar(boost::fusion::push_back(_seq, index::Scalar(n)));
    }
};

/// @addtogroup ndarrayMainGroup
/// @{

/** @brief Start a view definition that includes the entire first dimension. */
inline View< boost::fusion::vector1<index::Full> > view() {
    return View< boost::fusion::vector1<index::Full> >(
        boost::fusion::make_vector(index::Full())
    );
}

/** @brief Start a view definition that selects a contiguous range in the first dimension. */
inline View< boost::fusion::vector1<index::Range> > view(int start, int stop) {
    return View< boost::fusion::vector1<index::Range> >(
        boost::fusion::make_vector(index::Range(start, stop))
    );
}

/** @brief Start a view definition that selects a noncontiguous slice of the first dimension. */
inline View< boost::fusion::vector1<index::Slice> > view(int start, int stop, int step) {
    return View< boost::fusion::vector1<index::Slice> >(
        boost::fusion::make_vector(index::Slice(start, stop, step))
    );
}

/** @brief Start a view definition that selects single element from the first dimension. */
inline View< boost::fusion::vector1<index::Scalar> > view(int n) {
    return View< boost::fusion::vector1<index::Scalar> >(
        boost::fusion::make_vector(index::Scalar(n))
    );
}

/// @brief Create a view definition from a boost::fusion sequence of index objects.
template <typename Sequence>
inline View<Sequence> view(Sequence const & sequence) {
    return View<Sequence>(sequence);
}

/// @}

} // namespace ndarray

#endif // !NDARRAY_views_h_INCLUDED
