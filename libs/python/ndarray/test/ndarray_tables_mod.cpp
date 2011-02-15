#include <boost/python/ndarray/ndarray.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

static boost::mt19937 engine;
static boost::uniform_int<> random_int(2, 5);

namespace nt = ndarray::tables;

struct Tag {
    typedef boost::fusion::vector< nt::Field<int>, nt::Field<double,2>, nt::Field<float,1> > FieldSequence;

    static const nt::Index<0> a;
    static const nt::Index<1> b;
    static const nt::Index<2> c;
};

BOOST_PYTHON_MODULE(ndarray_mod) {
    boost::python::numpy::initialize();
}
