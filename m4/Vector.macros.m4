define(`VECTOR_ASSIGN',
`
    /// \brief Augmented $1 assignment from another vector.
    template <typename U>
    typename boost::enable_if<boost::is_convertible<U,T>,Vector&>::type
    operator $1 (Vector<U,N> const & other) {
        typename Vector<U,N>::ConstIterator j = other.begin();
        for (Iterator i = begin(); i != end(); ++i, ++j) (*i) $1 (*j);
        return *this;
    }
    /// \brief Augmented $1 assignment from a scalar.
    template <typename U>
    typename boost::enable_if<boost::is_convertible<U,T>,Vector&>::type
    operator $1 (U scalar) {
        for (Iterator i = begin(); i != end(); ++i) (*i) $1 scalar;
        return *this;
    }')dnl
define(`VECTOR_BINARY_OP',
`
    /// \brief Operator overload for Vector $1 Vector.
    template <typename T, typename U, int N>
    Vector<typename Promote<T,U>::Type,N>
    operator $1(Vector<T,N> const & a, Vector<U,N> const & b) {
        Vector<typename Promote<T,U>::Type,N> r(a);
        return r $1= b;
    }
    /** \brief Operator overload for Vector $1 Scalar. */
    template <typename T, typename U, int N>
    Vector<typename Promote<T,U>::Type,N>
    operator $1(Vector<T,N> const & a, U b) {
        Vector<typename Promote<T,U>::Type,N> r(a);
        return r $1= b;
    }
    /** \brief Operator overload for Scalar $1 Vector. */
    template <typename T, typename U, int N>
    Vector<typename Promote<T,U>::Type,N>
    operator $1(U a, Vector<T,N> const & b) {
        Vector<typename Promote<T,U>::Type,N> r(a);
        return r $1= b;
    }')dnl
