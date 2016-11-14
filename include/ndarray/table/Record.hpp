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
#include "ndarray/table/Schema.hpp"
#include "ndarray/table/detail/RecordImpl.hpp"

namespace ndarray {

template <typename S> class RecordRef;

template <typename S>
class RecordBase {
public:

    typedef DType<Schema> dtype_t;

    dtype_t const & dtype() const { return _impl.dtype(); }

    std::shared_ptr<Schema> const & schema() const { return dtype().schema(); }

protected:

    RecordBase() : _impl() {}

    RecordBase(
        byte_t * buffer_,
        dtype_t dtype_,
        std::shared_ptr<Manager> manager_
    ) : _impl(buffer_, std::move(dtype_), std::move(manager_)) {}

    RecordBase(detail::RecordImpl const & impl) : _impl(impl) {}
    RecordBase(detail::RecordImpl && impl) : _impl(std::move(impl)) {}

    RecordBase(RecordBase const &) = default;
    RecordBase(RecordBase &&) = default;

    RecordBase & operator=(RecordBase const &) = default;
    RecordBase & operator=(RecordBase &&) = default;

    detail::RecordImpl _impl;
};


template <typename S>
class Record<S const> : public RecordBase<S const> {
    typedef RecordBase<S const> base_t;
public:
    typedef DType<Schema> dtype_t;

    Record() : base_t() {}

    Record(
        byte_t * buffer_,
        dtype_t dtype_,
        std::shared_ptr<Manager> manager_
    ) : base_t(buffer_, std::move(dtype_), std::move(manager_)) {}

    Record(Record const &) = default;
    Record(Record &&) = default;

    Record & operator=(Record const &) = default;
    Record & operator=(Record &&) = default;

    void swap(Record & other) {
        this->_impl.swap(other._impl);
    }

    template <typename T>
    typename Key<T>::const_reference operator[](Key<T> const & key) const {
        return key.make_const_reference_at(
            this->_impl.buffer,
            this->_impl.manager()
        );
    }

#ifdef NDARRAY_FAST_CONVERSIONS

    RecordRef<S const> const & operator*() const;

#else

    RecordRef<S const> operator*() const;

#endif

protected:
    Record(detail::RecordImpl const & impl) : base_t(impl) {}
    Record(detail::RecordImpl && impl) : base_t(std::move(impl)) {}
private:
    friend class Record<S>;
    template <typename T> friend class RecordRef;
};

template <typename S>
class Record : public RecordBase<S> {
    typedef RecordBase<S> base_t;
public:
    typedef DType<Schema> dtype_t;

    Record() : base_t() {}

    Record(
        byte_t * buffer_,
        dtype_t dtype_,
        std::shared_ptr<Manager> manager_
    ) : base_t(buffer_, std::move(dtype_), std::move(manager_)) {}

    Record(Record const &) = default;
    Record(Record &&) = default;

    Record & operator=(Record const &) = default;
    Record & operator=(Record &&) = default;

    void swap(Record & other) {
        this->_impl.swap(other._impl);
    }

    template <typename T>
    typename Key<T>::reference operator[](Key<T> const & key) const {
        return key.make_reference_at(
            this->_impl.buffer,
            this->_impl.manager()
        );
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
    Record(detail::RecordImpl const & impl) : base_t(impl) {}
    Record(detail::RecordImpl && impl) : base_t(std::move(impl)) {}
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
public:
    typedef DType<Schema> dtype_t;

    RecordRef() : base_t() {}

    RecordRef(
        byte_t * buffer_,
        dtype_t dtype_,
        std::shared_ptr<Manager> manager_
    ) : base_t(buffer_, std::move(dtype_), std::move(manager_)) {}

    RecordRef(RecordRef const &) = default;
    RecordRef(RecordRef &&) = default;

    RecordRef & operator=(RecordRef const &) = delete;
    RecordRef & operator=(RecordRef &&) = delete;

    Record<S const> & shallow() { return *this; }

private:
    RecordRef(detail::RecordImpl const & impl) : base_t(std::move(impl)) {}
    RecordRef(detail::RecordImpl && impl) : base_t(std::move(impl)) {}
    friend class RecordRef<S>;
    template <typename T> friend class Record;
};

template <typename S>
class RecordRef : public Record<S> {
    typedef Record<S> base_t;
public:
    typedef DType<Schema> dtype_t;

    RecordRef() : base_t() {}

    RecordRef(
        byte_t * buffer_,
        dtype_t dtype_,
        std::shared_ptr<Manager> manager_
    ) : base_t(buffer_, std::move(dtype_), std::move(manager_)) {}

    RecordRef(RecordRef const &) = default;
    RecordRef(RecordRef &&) = default;

    RecordRef & operator=(RecordRef const &) = delete;
    RecordRef & operator=(RecordRef &&) = delete;

    RecordRef const & operator=(Record<S const> const & other) const;
    RecordRef const & operator=(Record<S> && other) const;

    Record<S> & shallow() { return *this; }

private:
    RecordRef(detail::RecordImpl const & impl) : base_t(std::move(impl)) {}
    RecordRef(detail::RecordImpl && impl) : base_t(std::move(impl)) {}
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

extern template class RecordBase<Schema>;
extern template class RecordBase<Schema const>;
extern template class Record<Schema>;
extern template class Record<Schema const>;
extern template class RecordRef<Schema>;
extern template class RecordRef<Schema const>;

#endif

} // ndarray

#endif // !NDARRAY_table_Record_hpp_INCLUDED