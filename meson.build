project('ndarray', 'cpp',
  version : '3.0.0',
  license : 'MIT',
  default_options : ['cpp_std=c++14'])

if meson.get_compiler('cpp').get_id() == 'clang'
  add_global_arguments('-Wno-missing-braces', language : 'cpp')
endif

inc = include_directories('include')

subdir('include')
subdir('src')
subdir('tests')
