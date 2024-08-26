cmake_minimum_required(VERSION 3.22)

project(ferroc)

option(FE_DEBUG, "Compile with debug symbols" OFF)
option(FE_TRACK_VALGRIND, "Compile with valgrind support" OFF)
option(FE_FINER_GRAINED, "Increase heap efficiency while decreasing minimal alignment from 16 to 8" OFF)

find_program(
    CARGO cargo
    HINTS ${CARGO_HOME} ${HOME}/.cargo
    PATH_SUFFIXES bin
)

if(NOT CARGO)
    message(
        FATAL_ERROR
        "Could not find cargo. Please download and install the Rust toolchains "
        "via the official website: https://www.rust-lang.org/install"
    )
endif()

exec_program(
    ${CARGO} ARGS +nightly --version
    OUTPUT_VARIABLE CARGO_OUTPUT
    RETURN_VALUE CARGO_RESULT
)
if (NOT CARGO_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "Cargo returned unexpected result ${CARGO_OUTPUT}; "
        "Please make sure that you have a nightly Rust toolchain installed."
    )
endif()

message(STATUS "Compile using ${CARGO_OUTPUT}")

set(arch ${CMAKE_SYSTEM_PROCESSOR})
if(arch STREQUAL "x86")
    set(arch "i686")
endif()
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(TRIPLE ${arch}-unknown-linux-gnu)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(TRIPLE ${arch}-pc-windows-msvc)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(TRIPLE ${arch}-apple-darwin)
endif()
message(STATUS "Target triple for Rust: ${TRIPLE}")

set(FE_BASE_NAME ferroc)
if(FE_TRACK_VALGRIND)
    set(FE_NAME ${FE_BASE_NAME}-valgrind)
else()
    set(FE_NAME ${FE_BASE_NAME})
endif()

if(NOT CMAKE_BUILD_TYPE)
    if ("${CMAKE_BINARY_DIR}" MATCHES ".*(D|d)ebug$" OR FE_DEBUG)
        message(STATUS "No build type selected, default to: Debug")
        set(CMAKE_BUILD_TYPE "Debug")
        set(BUILD_FLAGS "")
        set(BUILD_SUBDIR debug)
    else()
        message(STATUS "No build type selected, default to: Release")
        set(CMAKE_BUILD_TYPE "Release")
        set(BUILD_FLAGS "--release")
        set(BUILD_SUBDIR release)
    endif()
endif()

if(FE_TRACK_VALGRIND)
    set(BUILD_FLAGS ${BUILD_FLAGS} --features track-valgrind)
    message(STATUS "Enabling Valgrind support")
endif()

if(FE_FINER_GRAINED)
    set(BUILD_FLAGS ${BUILD_FLAGS} --features finer-grained)
    message(STATUS "Enabling finer-grained support")
endif()

set(CMAKE_CARGO_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml)
set(CMAKE_CARGO_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/target)
set(CMAKE_CARGO_LIBDIR ${CMAKE_CURRENT_SOURCE_DIR}/target/${TRIPLE}/${BUILD_SUBDIR})

file(GLOB_RECURSE SRC *.rs)
set(STATIC_LIB ${CMAKE_CARGO_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ferroc${CMAKE_STATIC_LIBRARY_SUFFIX})
set(SHARED_LIB ${CMAKE_CARGO_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}ferroc${CMAKE_SHARED_LIBRARY_SUFFIX})
set(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/ferroc.h)
message(STATUS "Building static library: ${STATIC_LIB}")
message(STATUS "Building shared library: ${SHARED_LIB}")

set(RUST_DEBUG_FLAGS "-Cdebuginfo=2")

set(RUSTFLAGS "--cfg sys_alloc -Ztls-model=initial-exec -Cpanic=abort")
if(FE_DEBUG)
    set(RUSTFLAGS "${RUSTFLAGS} ${RUST_DEBUG_FLAGS}")
endif()
message(STATUS "Rustc flags: ${RUSTFLAGS}")

add_custom_command(
    OUTPUT ${STATIC_LIB} ${SHARED_LIB}
    BYPRODUCTS ${HEADERS}
    COMMAND ${CARGO} +nightly build ${BUILD_FLAGS} --features c
        -Zbuild-std=core,std,panic_abort -Zbuild-std-features=
        --target ${TRIPLE}
    DEPENDS ${SRC}
    COMMENT "Building static & shared libraries of ferroc"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
    ferroc-build
    ALL
    DEPENDS ${STATIC_LIB} ${SHARED_LIB}
)

add_library(ferroc SHARED IMPORTED GLOBAL)
add_dependencies(ferroc ferroc-build)
set_target_properties(ferroc PROPERTIES IMPORTED_LOCATION ${SHARED_LIB})

add_library(ferroc-static STATIC IMPORTED GLOBAL)
add_dependencies(ferroc-static ferroc-build)
set_target_properties(ferroc-static PROPERTIES IMPORTED_LOCATION ${STATIC_LIB})

install(FILES ${HEADERS} TYPE INCLUDE COMPONENT headers)
install(FILES ${SHARED_LIB} TYPE LIB COMPONENT libraries)
install(FILES ${STATIC_LIB} TYPE LIB COMPONENT libraries)
