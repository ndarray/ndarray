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
#ifndef NDARRAY_errors_hpp_INCLUDED
#define NDARRAY_errors_hpp_INCLUDED

#include <cstdlib>
#include <cstdio>
#include "fmt/format.h"
#include "ndarray/common.hpp"


namespace ndarray {

class Error {
public:

    enum Category { NONCONTIGUOUS, UNINITIALIZED, OUT_OF_BOUNDS, INCOMPATIBLE_ARGUMENTS };

    typedef void (*Handler)(Category, char const * file, int line, std::string const & message);

    class ScopedHandler {
    public:

        explicit ScopedHandler(Handler handler) : _old(get_handler()) {
            set_handler(handler);
        }

        ScopedHandler(ScopedHandler const &) = delete;
        ScopedHandler(ScopedHandler &&) = delete;

        ScopedHandler & operator=(ScopedHandler const &) = delete;
        ScopedHandler & operator=(ScopedHandler &&) = delete;

        ~ScopedHandler() { set_handler(_old); }

    private:
        Handler _old;
    };

    template <typename Exception>
    static void throw_handler(Category category, char const * file, int line, std::string const & message) {
        throw Exception(message);
    }

    static void set_handler(Handler handler=nullptr) {
        get_handler() = handler;
    }

    [[ noreturn ]] static void invoke (
        Category category,
        char const * file,
        int line,
        std::string const & message
    ) {
        Handler handler = get_handler();
        if (handler) {
            handler(category, file, line, message);
        } else {
            fmt::print(stderr, "{:s}:{:d}: {:s}", file, line, message);
        }
        std::abort();
    }

    template <typename ...Args>
    [[ noreturn ]] static void invoke (
        Category category,
        char const * file,
        int line,
        char const * tmpl,
        Args && ...args
    ) {
        invoke(category, file, line, fmt::format(tmpl, std::forward<Args>(args)...));
    }

private:

    static Handler & get_handler() {
        static Handler handler = nullptr;
        return handler;
    }
};


#define NDARRAY_FAIL(CATEGORY, ...) \
    Error::invoke(CATEGORY, __FILE__, __LINE__, __VA_ARGS__)


#ifndef NDARRAY_ASSERT_AUDIT_ENABLED
    #define NDARRAY_ASSERT_AUDIT_ENABLED false
#endif

#if NDARRAY_ASSERT_AUDIT_ENABLED
    #define NDARRAY_ASSERT_CHECK_ENABLED true
#else
    #ifndef NDARRAY_ASSERT_CHECK_ENABLED
        #ifdef NDEBUG
            #define NDARRAY_ASSERT_CHECK_ENABLED false
        #else
            #define NDARRAY_ASSERT_CHECK_ENABLED true
        #endif
    #endif
#endif

#if NDARRAY_ASSERT_AUDIT_ENABLED
    #define NDARRAY_ASSERT_AUDIT(CONDITION, CATEGORY, ...) \
        if (!(CONDITION)) NDARRAY_FAIL(CATEGORY, __VA_ARGS__)
#else
    #define NDARRAY_ASSERT_AUDIT(CONDITION, CATEGORY, ...) ((void)0)
#endif

#if NDARRAY_ASSERT_CHECK_ENABLED
    #define NDARRAY_ASSERT_CHECK(CONDITION, CATEGORY, ...) \
        if (!(CONDITION)) NDARRAY_FAIL(CATEGORY, __VA_ARGS__)
#else
    #define NDARRAY_ASSERT_CHECK(CONDITION, CATEGORY, ...) ((void)0)
#endif

} // ndarray

#endif // !NDARRAY_errors_hpp_INCLUDED
