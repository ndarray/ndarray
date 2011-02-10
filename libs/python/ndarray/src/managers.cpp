#include "boost/python/ndarray/ndarray.hpp"

namespace bp = boost::python;

namespace ndarray {

static void destroyManagerCObject(void * p) {
    Manager::Ptr * b = reinterpret_cast<Manager::Ptr*>(p);
    delete b;
}

bp::object makePyObject(Manager::Ptr const & x) {
    boost::intrusive_ptr< ExternalManager<bp::object> > y 
        = boost::dynamic_pointer_cast< ExternalManager<bp::object> >(x);
    if (y) {
        return y->getOwner();
    }
    bp::handle<> h(::PyCObject_FromVoidPtr(new Manager::Ptr(x), &destroyManagerCObject));
    return bp::object(h);
}

} // namespace ndarray
