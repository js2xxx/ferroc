cmake_minimum_required(VERSION 3.22)

project(ferroc)

option(FE_TRACK_VALGRIND, "Compile with valgrind support" OFF)
option(FE_FINER_GRAINED, "Increase heap efficiency while decreasing minimal alignment from 16 to 8" OFF)
option(FE_PGO_GATHER, "Enable PGO profiling: add instrumentation" OFF)
option(FE_PGO_USE, "Enable PGO profiling: use pre-compiled profile" OFF)

include(GNUInstallDirs)

find_program(
    CARGO cargo REQUIRED
    HINTS ${CARGO_HOME} ${HOME}/.cargo
    PATH_SUFFIXES bin
)
find_program(
    RUSTC rustc REQUIRED
    HINTS ${CARGO_HOME} ${HOME}/.cargo
    PATH_SUFFIXES bin
)

execute_process(
    COMMAND ${CARGO} +nightly --version
    OUTPUT_VARIABLE CARGO_OUTPUT
    RESULT_VARIABLE CARGO_RESULT
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
    if ("${CMAKE_BINARY_DIR}" MATCHES ".*(D|d)ebug$")
        message(STATUS "No build type selected, default to: Debug")
        set(CMAKE_BUILD_TYPE "Debug")
    else()
        message(STATUS "No build type selected, default to: Release")
        set(CMAKE_BUILD_TYPE "Release")
    endif()
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(BUILD_FLAGS)
set(BUILD_SUBDIR debug)
if(CMAKE_BUILD_TYPE MATCHES ".*Rel.*")
    set(BUILD_FLAGS "--release")
    set(BUILD_SUBDIR release)
endif()
message(STATUS "Using profile: ${BUILD_SUBDIR}")

if(FE_TRACK_VALGRIND)
    set(BUILD_FLAGS ${BUILD_FLAGS} --features track-valgrind)
    message(STATUS "Enabling Valgrind support")
endif()

if(FE_FINER_GRAINED)
    set(BUILD_FLAGS ${BUILD_FLAGS} --features finer-grained)
    message(STATUS "Enabling finer-grained support")
endif()

set(CMAKE_CARGO_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml)

file(GLOB_RECURSE SRC *.rs)
set(STATIC_LIB_NAME ${CMAKE_STATIC_LIBRARY_PREFIX}ferroc${CMAKE_STATIC_LIBRARY_SUFFIX})
set(SHARED_LIB_NAME ${CMAKE_SHARED_LIBRARY_PREFIX}ferroc${CMAKE_SHARED_LIBRARY_SUFFIX})

set(STATIC_LIB ${CMAKE_BINARY_DIR}/${STATIC_LIB_NAME})
set(SHARED_LIB ${CMAKE_BINARY_DIR}/${SHARED_LIB_NAME})
set(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/ferroc.h)
message(STATUS "Building static library: ${STATIC_LIB}")
message(STATUS "Building shared library: ${SHARED_LIB}")

set(RUSTFLAGS "-Ztls-model=initial-exec -Cpanic=abort")
if(CMAKE_BUILD_TYPE MATCHES ".*Deb.*")
    set(RUSTFLAGS "${RUSTFLAGS} -Cdebuginfo=2")
endif()

if(NOT FE_PGO_GATHER)
    set(BUILD_STD_FEATURES compiler-builtins-mem,compiler-builtins-weak-intrinsics)
    if(CMAKE_BUILD_TYPE MATCHES ".*Rel.*")
        set(BUILD_STD_FEATURES ${BUILD_STD_FEATURES},panic_immediate_abort)
    endif()
    set(BUILD_FLAGS ${BUILD_FLAGS}
        -Zbuild-std=core,std,panic_abort
        -Zbuild-std-features=${BUILD_STD_FEATURES}
    )
endif()

set(SRC_PROFDATA)
if(FE_PGO_USE)
    set(SRC_PROFDATA "/tmp/pgo/ferroc.profdata")

    execute_process(
        COMMAND ${RUSTC} --print sysroot
        OUTPUT_VARIABLE RUSTC_SYSROOT
    )
    find_program(
        PROFDATA llvm-profdata
        PATHS ${RUSTC_SYSROOT}/lib/rustlib/${TRIPLE}/bin
        REQUIRED NO_DEFAULT_PATH
    )

    add_custom_command(
        OUTPUT ${SRC_PROFDATA}
        COMMAND ${CMAKE_COMMAND} -E make_directory /tmp/pgo
        COMMAND ${PROFDATA} merge -o ${SRC_PROFDATA} ${CMAKE_BINARY_DIR}/pgo
        COMMENT "Merging PGO profile data"
    )
    message(STATUS "Using PGO profile data at ${SRC_PROFDATA}, program at ${PROFDATA}")
endif()

if(FE_PGO_GATHER)
    set(RUSTFLAGS "${RUSTFLAGS} -Cprofile-generate=${CMAKE_BINARY_DIR}/pgo")
    message(STATUS "Enabling PGO gathering")
elseif(FE_PGO_USE)
    set(RUSTFLAGS "${RUSTFLAGS} -Cprofile-use=\"${SRC_PROFDATA}\"")
    message(STATUS "Enabling PGO usage, please make sure data has been gathered")
endif()

message(STATUS "RUSTC flags: ${RUSTFLAGS}")
message(STATUS "Build flags: ${BUILD_FLAGS}")

add_custom_command(
    OUTPUT ${STATIC_LIB} ${SHARED_LIB}
    BYPRODUCTS ${HEADERS}
    COMMAND ${CMAKE_COMMAND}
        -E env RUSTFLAGS=${RUSTFLAGS}
        ${CARGO} +nightly build ${BUILD_FLAGS} --features c-override --target ${TRIPLE}
            --manifest-path ${CMAKE_CARGO_MANIFEST} --target-dir ${CMAKE_BINARY_DIR}
            --out-dir ${CMAKE_BINARY_DIR} -Zunstable-options
    DEPENDS ${SRC} ${SRC_PROFDATA}
    COMMENT "Building static & shared libraries of ferroc"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
    ferroc-build ALL
    DEPENDS ${STATIC_LIB} ${SHARED_LIB}
    COMMENT "Copying libraries to root target directory"
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
