#include <boost/python/ndarray/tables.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

static boost::mt19937 engine;
static boost::uniform_int<> random_int(2, 5);

namespace nt = ndarray::tables;
namespace bp = boost::python;

struct Tag {
    typedef boost::fusion::vector< nt::Field<int>, nt::Field<double,2>, nt::Field<float,1> > FieldSequence;

    static const nt::Index<0> a;
    static const nt::Index<1> b;
    static const nt::Index<2> c;
};

static nt::Layout<Tag> makeLayout(int b0, int b1, int c0, bool pack) {
    nt::Layout<Tag> layout;
    layout[Tag::a].name = "a";
    layout[Tag::b].name = "b";
    layout[Tag::c].name = "c";
    layout[Tag::b].shape[0] = b0;
    layout[Tag::b].shape[1] = b1;
    layout[Tag::c].shape[0] = c0;
    layout.normalize(pack);
    return layout;
}

static bool compareLayouts(nt::Layout<Tag> const & layout, int b0, int b1, int c0, bool pack) {
    nt::Layout<Tag> other = makeLayout(b0, b1, c0, pack);
    return layout == other;
}

static bool compareTables(
    nt::Table<Tag> const & table, 
    ndarray::Array<int,1,0> const & a, 
    ndarray::Array<double,3,2> const & b, 
    ndarray::Array<float,2,1> const & c
) {
    return (table[Tag::a].shallow() == a) && (table[Tag::b].shallow() == b) && (table[Tag::c].shallow() == c);
}

BOOST_PYTHON_MODULE(ndarray_tables_mod) {
    bp::numpy::initialize();
    bp::def("makeLayout", &makeLayout, (bp::arg("b0"), bp::arg("b1"), bp::arg("c0"), bp::arg("pack")));
    bp::def("compareLayouts", &compareLayouts,
            (bp::arg("layout"), bp::arg("b0"), bp::arg("b1"), bp::arg("c0"), bp::arg("pack")));
    bp::def("makeTable", &nt::Table<Tag>::allocate, (bp::arg("size"), bp::arg("layout")));
    bp::def("compareTables", &compareTables, (bp::arg("tables"), bp::arg("a"), bp::arg("b"), bp::arg("c")));
}
