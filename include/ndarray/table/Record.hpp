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
#ifndef NDARRAY_table_Record_hpp_INCLUDED
#define NDARRAY_table_Record_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/table/common.hpp"
#include "ndarray/table/Schema.hpp"
#include "ndarray/table/detail/RecordImpl.hpp"

namespace ndarray {

template <typename S> class RecordRef;

template <typename S>
class RecordBase {
    using Storage = typename std::remove_const<S>::type;
public:

    std::shared_ptr<Schema> const & schema() const { return _impl.schema(); }

protected:

    RecordBase(detail::RecordImpl<Storage> const & impl) : _impl(impl) {}
    RecordBase(detail::RecordImpl<Storage> && impl) : _impl(std::move(impl)) {}

    RecordBase(RecordBase const &) = default;
    RecordBase(RecordBase &&) = default;

    RecordBase & operator=(RecordBase const &) = default;
    RecordBase & operator=(RecordBase &&) = default;

    detail::RecordImpl<Storage> _impl;
};


template <typename S>
class Record<S const> : public RecordBase<S const> {
    typedef RecordBase<S const> base_t;
    using Storage = typename std::remove_const<S>::type;
public:

    Record(Record const &) = default;
    Record(Record &&) = default;

    Record & operator=(Record const &) = default;
    Record & operator=(Record &&) = default;

    void swap(Record & other) {
        this->_impl.swap(other._impl);
    }

    template <typename T>
    typename Key<T>::const_reference operator[](Key<T> const & key) const {
        return this->_impl.cget(key);
    }

#ifdef NDARRAY_FAST_CONVERSIONS

    RecordRef<S const> const & operator*() const;

#else

    RecordRef<S const> operator*() const;

#endif

protected:
    Record(detail::RecordImpl<Storage> const & impl) : base_t(impl) {}
    Record(detail::RecordImpl<Storage> && impl) : base_t(std::move(impl)) {}
private:
    friend class Record<S>;
    template <typename T> friend class RecordRef;
};

template <typename S>
class Record : public RecordBase<S> {
    typedef RecordBase<S> base_t;
    using Storage = typename std::remove_const<S>::type;
public:

    Record(Record const &) = default;
    Record(Record &&) = default;

    Record & operator=(Record const &) = default;
    Record & operator=(Record &&) = default;

    void swap(Record & other) {
        this->_impl.swap(other._impl);
    }

    template <typename T>
    typename Key<T>::reference operator[](Key<T> const & key) const {
        return this->_impl.get(key);
    }

#ifdef NDARRAY_FAST_CONVERSIONS

    RecordRef<S> const & operator*() const;

    operator Record<S const> const & () const {
        return *reinterpret_cast<Record<S> const *>(this);
    }

#else

    RecordRef<S> operator*() const;

    operator Record<S const>() const {
        return Record<S const>(this->_impl);
    }

#endif

protected:
    Record(detail::RecordImpl<Storage> const & impl) : base_t(impl) {}
    Record(detail::RecordImpl<Storage> && impl) : base_t(std::move(impl)) {}
private:
    friend class Record<S const>;
    template <typename T> friend class RecordRef;
};

template <typename S>
void swap(Record<S> & a, Record<S> & b) {
    a.swap(b);
}


template <typename S>
class RecordRef<S const> : public Record<S const> {
    typedef Record<S const> base_t;
    using Storage = typename std::remove_const<S>::type;
public:

    RecordRef(RecordRef const &) = default;
    RecordRef(RecordRef &&) = default;

    RecordRef & operator=(RecordRef const &) = delete;
    RecordRef & operator=(RecordRef &&) = delete;

    Record<S const> & shallow() { return *this; }

private:
    RecordRef(detail::RecordImpl<Storage> const & impl) : base_t(std::move(impl)) {}
    RecordRef(detail::RecordImpl<Storage> && impl) : base_t(std::move(impl)) {}
    friend class RecordRef<S>;
    template <typename T> friend class Record;
};

template <typename S>
class RecordRef : public Record<S> {
    typedef Record<S> base_t;
    using Storage = typename std::remove_const<S>::type;
public:

    RecordRef(RecordRef const &) = default;
    RecordRef(RecordRef &&) = default;

    RecordRef & operator=(RecordRef const &) = delete;
    RecordRef & operator=(RecordRef &&) = delete;

    template <typename Other>
    RecordRef const & operator=(Record<Other> const & other) const;

    Record<S> & shallow() { return *this; }

private:
    RecordRef(detail::RecordImpl<Storage> const & impl) : base_t(std::move(impl)) {}
    RecordRef(detail::RecordImpl<Storage> && impl) : base_t(std::move(impl)) {}
    friend class RecordRef<S const>;
    template <typename T> friend class Record;
};

#ifdef NDARRAY_FAST_CONVERSIONS

template <typename S>
inline RecordRef<S const> const & Record<S const>::operator*() const {
    return *reinterpret_cast<RecordRef<S const>*>(this);
}

template <typename S>
inline RecordRef<S> const & Record<S>::operator*() const {
    return *reinterpret_cast<RecordRef<S>*>(this);
}

#else

template <typename S>
inline RecordRef<S const> Record<S const>::operator*() const {
    return RecordRef<S const>(this->_impl);
}

template <typename S>
inline RecordRef<S> Record<S>::operator*() const {
    return RecordRef<S>(this->_impl);
}

#endif


#ifndef NDARRAY_table_Record_cpp_ACTIVE

extern template class RecordBase<FixedRow>;
extern template class RecordBase<FixedRow const>;
extern template class Record<FixedRow>;
extern template class Record<FixedRow const>;
extern template class RecordRef<FixedRow>;
extern template class RecordRef<FixedRow const>;

extern template class RecordBase<FlexRow>;
extern template class RecordBase<FlexRow const>;
extern template class Record<FlexRow>;
extern template class Record<FlexRow const>;
extern template class RecordRef<FlexRow>;
extern template class RecordRef<FlexRow const>;

extern template class RecordBase<FixedCol>;
extern template class RecordBase<FixedCol const>;
extern template class Record<FixedCol>;
extern template class Record<FixedCol const>;
extern template class RecordRef<FixedCol>;
extern template class RecordRef<FixedCol const>;

extern template class RecordBase<FlexCol>;
extern template class RecordBase<FlexCol const>;
extern template class Record<FlexCol>;
extern template class Record<FlexCol const>;
extern template class RecordRef<FlexCol>;
extern template class RecordRef<FlexCol const>;

#endif

} // ndarray

#endif // !NDARRAY_table_Record_hpp_INCLUDED