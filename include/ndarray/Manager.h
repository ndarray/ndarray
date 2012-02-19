// -*- c++ -*-
/*
 * Copyright 2012, Jim Bosch and the LSST Corporation
 * 
 * ndarray is available under two licenses, both of which are described
 * more fully in other files that should be distributed along with
 * the code:
 * 
 *  - A simple BSD-style license (ndarray-bsd-license.txt); under this
 *    license ndarray is broadly compatible with essentially any other
 *    code.
 * 
 *  - As a part of the LSST data management software system, ndarray is
 *    licensed with under the GPL v3 (LsstLicenseStatement.txt).
 * 
 * These files can also be found in the source distribution at:
 * 
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_Manager_h_INCLUDED
#define NDARRAY_Manager_h_INCLUDED

/** 
 *  @file ndarray/Manager.h
 *
 *  @brief Definition of Manager, which manages the ownership of array data.
 */

#include "ndarray_fwd.h"
#include <boost/noncopyable.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/scoped_array.hpp>

namespace ndarray {

class Manager : private boost::noncopyable {
public:

    typedef boost::intrusive_ptr<Manager> Ptr;

    friend inline void intrusive_ptr_add_ref(Manager const * manager) {
        ++manager->_rc;
    }
 
    friend inline void intrusive_ptr_release(Manager const * manager) {
        if ((--manager->_rc)==0) delete manager;
    }

    int getRC() const { return _rc; }

    virtual bool isUnique() const { return false; }

protected:

    virtual ~Manager() {}

    explicit Manager() : _rc(0) {}

private:
    mutable int _rc;
};

template <typename T>
class SimpleManager : public Manager {
    typedef typename boost::remove_const<T>::type U;
public:
    
    static std::pair<Manager::Ptr,T*> allocate(int size) {
        boost::intrusive_ptr<SimpleManager> r(new SimpleManager(size));
        return std::pair<Manager::Ptr,T*>(r, r->_p.get());
    }

    virtual bool isUnique() const { return true; }

private:
    explicit SimpleManager(int size) : _p() {
        if (size > 0) _p.reset(new U[size]);
    }
    boost::scoped_array<U> _p;
};

template <typename T> Manager::Ptr makeManager(T const & owner);

template <typename U>
class ExternalManager : public Manager, private U {
public:
    typedef U Owner;

    template <typename T> friend Manager::Ptr makeManager(T const & owner);

    Owner const & getOwner() const { return *static_cast<Owner const *>(this); }

private:
    explicit ExternalManager(Owner const & owner) : Owner(owner) {}
};

template <typename T>
inline Manager::Ptr makeManager(T const & owner) {
    return Manager::Ptr(new ExternalManager<T>(owner));
}

// A no-op overload for makeManager to avoid unnecessary levels of indirection.
inline Manager::Ptr makeManager(Manager::Ptr const & owner) {
    return owner;
}

} // namespace ndarray

#endif // !NDARRAY_Manager_h_INCLUDED
