cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(SphereFittingTest LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(Eigen3 3.3 REQUIRED CONFIG)

if (TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 v${EIGEN3_VERSION_STRING} found in ${EIGEN3_INCLUDE_DIR}")
endif()

set (SPHEREFITTINGTEST_INC ${CMAKE_CURRENT_SOURCE_DIR}/headers)
set (SPHEREFITTINGTEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sources)

file (GLOB HPP_FILES ${SPHEREFITTINGTEST_INC}/*.hpp)
file (GLOB CPP_FILES ${SPHEREFITTINGTEST_SRC}/*.cpp)

option (BUILD_UNIT_TESTS "Build unit tests" ON)
message (STATUS "Build unit tests: ${BUILD_UNIT_TESTS}")

if (BUILD_UNIT_TESTS)
  include(FetchContent)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
  )

  if (NOT googletest_POPULATED)

    FetchContent_Populate(googletest)

    # Prevent GoogleTest from overriding our compiler/linker options with building with Visual Studio
    set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Prevent GoogleTest from using PThreads
    set (gtest_disable_pthreads ON CACHE BOOL "" FORCE)

    # Where Google Test header files can be found.
    set(GTEST_BUILD_INCLUDE_DIRS
      "${gtest_SOURCE_DIR}/include"
      "${gtest_SOURCE_DIR}"
    )

    message (STATUS "googletest_SOURCE_DIR is ${googletest_SOURCE_DIR}")
    message (STATUS "googletest_BINARY_DIR is ${googletest_BINARY_DIR}")

    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

    # Silence std::tr1 warning on MSVC
    if (MSVC)
      foreach (_tgt gtest gtest_main gmock gmock_main)
        target_compile_definitions(${_tgt}
	  PRIVATE
	    "_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING"
	)
      endforeach()
    endif()

  endif()

  add_executable(SphereFittingTest "")

  target_compile_features(SphereFittingTest PRIVATE cxx_std_17)

  if (MSVC)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(SphereFittingTest PRIVATE /W4 /WX- /O2 /MD)
    else ()
      target_compile_options(SphereFittingTest PRIVATE /W4 /WX- /MDd)
    endif ()
  else ()
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(SphereFittingTest PRIVATE -Wall -Wextra -Werror -pedantic -O3)
    else ()
      target_compile_options(SphereFittingTest PRIVATE -Wall -Wextra -Werror -pedantic)
    endif ()
  endif ()

  target_sources(SphereFittingTest PRIVATE ${HPP_FILES} ${CPP_FILES})

  target_include_directories(
    SphereFittingTest
      PRIVATE
      	${EIGEN3_INCLUDE_DIR}
	${GTEST_BUILD_INCLUDE_DIRS}
	${SPHEREFITTINGTEST_INC}
  )

  target_link_libraries(
    SphereFittingTest
      PRIVATE
      gtest_main
  )

  enable_testing()

  add_test(
    NAME google_test
    COMMAND $<TARGET_FILE:SphereFittingTest>
  )

  install(TARGETS SphereFittingTest DESTINATION tst)

endif()
