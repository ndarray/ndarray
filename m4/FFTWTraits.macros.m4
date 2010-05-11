define(`FFTW_TRAITS',
`
    template <> struct FFTWTraits<$1> {
        BOOST_STATIC_ASSERT((!boost::is_const<$1>::value));
        typedef $2_plan Plan;
        typedef FourierTraits<$1>::ElementX ElementX;
        typedef FourierTraits<$1>::ElementK ElementK;
        typedef boost::shared_ptr<ElementX> OwnerX;
        typedef boost::shared_ptr<ElementK> OwnerK;
        static inline Plan forward(int rank, const int *n, int howmany,	
                                   ElementX *in, const int *inembed, int istride, int idist,
                                   ElementK *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return $2_plan_many_dft_r2c(rank,n,howmany,
                                                in,inembed,istride,idist,
                                                reinterpret_cast<$2_complex*>(out),
                                                onembed,ostride,odist,
                                                flags);			
        }
        static inline Plan inverse(int rank, const int *n, int howmany,
                                   ElementK *in, const int *inembed, int istride, int idist,
                                   ElementX *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return $2_plan_many_dft_c2r(rank,n,howmany,
                                                reinterpret_cast<$2_complex*>(in),
                                                inembed,istride,idist,
                                                out,onembed,ostride,odist,
                                                flags);			
        }
        static inline void destroy(Plan p) { $2_destroy_plan(p); }
        static inline void execute(Plan p) { $2_execute(p); }	
        static inline OwnerX allocateX(int n) {
            return OwnerX(
                reinterpret_cast<ElementX*>(
                    $2_malloc(sizeof(ElementX)*n)
                ),
                $2_free
            );
        }
        static inline OwnerK allocateK(int n) {
            return OwnerK(
                reinterpret_cast<ElementK*>(
                    $2_malloc(sizeof(ElementK)*n)
                ),
                $2_free
            );
        }
    };
    template <> struct FFTWTraits< std::complex<$1> > {
        typedef $2_plan Plan;
        typedef FourierTraits< std::complex<$1> >::ElementX ElementX;
        typedef FourierTraits< std::complex<$1> >::ElementK ElementK;
        typedef boost::shared_ptr<ElementX> OwnerX;
        typedef boost::shared_ptr<ElementK> OwnerK;
        static inline Plan forward(int rank, const int *n, int howmany,	
                                   ElementX *in, const int *inembed, int istride, int idist,
                                   ElementK *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return $2_plan_many_dft(rank,n,howmany,
                                            reinterpret_cast<$2_complex*>(in),
                                            inembed,istride,idist,
                                            reinterpret_cast<$2_complex*>(out),
                                            onembed,ostride,odist,
                                            FFTW_FORWARD,flags);
        }
        static inline Plan inverse(int rank, const int *n, int howmany,
                                   ElementK *in, const int *inembed, int istride, int idist,
                                   ElementX *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return $2_plan_many_dft(rank,n,howmany,
                                            reinterpret_cast<$2_complex*>(in),
                                            inembed,istride,idist,
                                            reinterpret_cast<$2_complex*>(out),
                                            onembed,ostride,odist,
                                            FFTW_BACKWARD,flags);
        }
        static inline void destroy(Plan p) { $2_destroy_plan(p); }
        static inline void execute(Plan p) { $2_execute(p); }	
        static inline OwnerX allocateX(int n) {
            return OwnerX(
                reinterpret_cast<ElementX*>(
                    $2_malloc(sizeof(ElementX)*n)
                ),
                $2_free
            );
        }
        static inline OwnerK allocateK(int n) {
            return OwnerK(
                reinterpret_cast<ElementK*>(
                    $2_malloc(sizeof(ElementK)*n)
                ),
                $2_free
            );
        }
    }')dnl
