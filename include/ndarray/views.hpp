// -*- c++ -*-
/*
 * Copyright (c) 2010-2018, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_views_hpp_INCLUDED
#define NDARRAY_views_hpp_INCLUDED

#include "ndarray/common.hpp"

namespace ndarray {
namespace views {

struct Begin {};
struct End {};
struct All {};

struct NewAxis {};

struct NegUnit;

struct PosUnit {
    PosUnit operator+() const { return PosUnit(); }
    NegUnit operator-() const;
};

struct NegUnit {
    NegUnit operator+() const { return NegUnit(); }
    PosUnit operator-() const;
};

inline NegUnit PosUnit::operator-() const { return NegUnit(); }
inline PosUnit NegUnit::operator-() const { return PosUnit(); }

#if __cplusplus >= 201703L
    inline Begin begin;
    inline End end;
    inline PosUnit unit;
    inline NewAxis newaxis;
    inline All all;
#endif

namespace detail {

struct Index {
    Offset value;
};

template <typename Start, typename Stop, typename Step=PosUnit>
struct Slice {
    Start start;
    Stop stop;
    Step step;
};

inline Index interpret(Offset index) { return Index{index}; }
inline NewAxis interpret(NewAxis) { return NewAxis(); }
inline Slice<Begin, End> interpret(All) { return Slice<Begin, End>{Begin(), End(), PosUnit()}; }

inline Slice<Offset, Offset, Offset> interpret(Offset a, Offset b, Offset c) { return Slice<Offset, Offset, Offset>{a, b, c}; }
inline Slice<Offset,    End, Offset> interpret(Offset a,    End b, Offset c) { return Slice<Offset,    End, Offset>{a, b, c}; }
inline Slice< Begin, Offset, Offset> interpret( Begin a, Offset b, Offset c) { return Slice< Begin, Offset, Offset>{a, b, c}; }
inline Slice< Begin,    End, Offset> interpret( Begin a,    End b, Offset c) { return Slice< Begin,    End, Offset>{a, b, c}; }

inline Slice<Offset, Offset, PosUnit> interpret(Offset a, Offset b, PosUnit c=PosUnit()) { return Slice<Offset, Offset, PosUnit>{a, b, c}; }
inline Slice<Offset,    End, PosUnit> interpret(Offset a,    End b, PosUnit c=PosUnit()) { return Slice<Offset,    End, PosUnit>{a, b, c}; }
inline Slice< Begin, Offset, PosUnit> interpret( Begin a, Offset b, PosUnit c=PosUnit()) { return Slice< Begin, Offset, PosUnit>{a, b, c}; }
inline Slice< Begin,    End, PosUnit> interpret( Begin a,    End b, PosUnit c=PosUnit()) { return Slice< Begin,    End, PosUnit>{a, b, c}; }

inline Slice<Offset, Offset, NegUnit> interpret(Offset a, Offset b, NegUnit c) { return Slice<Offset, Offset, NegUnit>{a, b, c}; }
inline Slice<Offset,    End, NegUnit> interpret(Offset a,    End b, NegUnit c) { return Slice<Offset,    End, NegUnit>{a, b, c}; }
inline Slice< Begin, Offset, NegUnit> interpret( Begin a, Offset b, NegUnit c) { return Slice< Begin, Offset, NegUnit>{a, b, c}; }
inline Slice< Begin,    End, NegUnit> interpret( Begin a,    End b, NegUnit c) { return Slice< Begin,    End, NegUnit>{a, b, c}; }

template <typename ...E> struct Sequence;

template <typename Current>
class Sequence<Current> {
public:

    template <typename Index>
    auto operator()(Index index) const {
        return append(interpret(index));
    }

    template <typename Start, typename Stop>
    auto operator()(Start start, Stop stop) const {
        return append(interpret(start, stop));
    }

    template <typename Start, typename Stop, typename Step>
    auto operator()(Start start, Stop stop, Step step) const {
        return append(interpret(start, stop, step));
    }

    template <typename Next>
    Sequence<Next, Current> append(Next const & next) const {
        return Sequence<Next, Current>{next, *this};
    }

    Current current;
};

template <typename Current, typename ...Previous>
class Sequence<Current, Previous...> {
public:

    template <typename Index>
    auto operator()(Index index) const {
        return append(interpret(index));
    }

    template <typename Start, typename Stop>
    auto operator()(Start start, Stop stop) const {
        return append(interpret(start, stop));
    }

    template <typename Start, typename Stop, typename Step>
    auto operator()(Start start, Stop stop, Step step) const {
        return append(interpret(start, stop, step));
    }

    template <typename Next>
    Sequence<Next, Current, Previous...> append(Next const & next) const {
        return Sequence<Next, Current, Previous...>{next, *this};
    }

    Current current;
    Sequence<Previous...> previous;
};

template <typename Next>
Sequence<Next> make_sequence(Next const & next) {
    return Sequence<Next>{next};
}

} // namespace detail

template <typename Index>
inline auto view(Index index) {
    return detail::make_sequence(detail::interpret(index));
}

template <typename Start, typename Stop>
inline auto view(Start start, Stop stop) {
    return detail::make_sequence(detail::interpret(start, stop));
}

template <typename Start, typename Stop, typename Step>
inline auto view(Start start, Stop stop, Step step) {
    return detail::make_sequence(detail::interpret(start, stop, step));
}

} // namespace views
} // ndarray

#endif // !NDARRAY_views_hpp_INCLUDED
