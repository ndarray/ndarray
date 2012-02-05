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
#ifndef LSST_NDARRAY_Manager_h_INCLUDED
#define LSST_NDARRAY_Manager_h_INCLUDED

/** 
 *  @file lsst/ndarray/Manager.h
 *
 *  @brief Definition of Manager, which manages the ownership of array data.
 */

#include "lsst/ndarray_fwd.h"
#include <boost/noncopyable.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/scoped_array.hpp>

namespace lsst { namespace ndarray {

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

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_Manager_h_INCLUDED
