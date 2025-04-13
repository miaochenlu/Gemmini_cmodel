cmake_minimum_required(VERSION 3.12)
project(gemmini_simulator VERSION 0.1.0)

# Set default build type to Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Configure C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compile options
add_compile_options(-Wall -Wextra)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find Boost library
find_package(Boost 1.74 REQUIRED COMPONENTS timer system program_options serialization)
if(Boost_FOUND)
    message(STATUS "Found Boost library: ${Boost_INCLUDE_DIRS} (version \"${Boost_VERSION}\")")
else()
    message(FATAL_ERROR "Boost library not found. Please install Boost 1.74 or higher.")
endif()

# Find HDF5 library (required for simdb)
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
    message(STATUS "Found HDF5 library: ${HDF5_INCLUDE_DIRS} (version \"${HDF5_VERSION}\")")
else()
    message(FATAL_ERROR "HDF5 library not found. Please install HDF5.")
endif()

# Find YAML-CPP library
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
    message(STATUS "Found YAML-CPP library: ${YAML_CPP_INCLUDE_DIR} (version \"${YAML_CPP_VERSION}\")")
else()
    message(FATAL_ERROR "YAML-CPP library not found. Please install yaml-cpp.")
endif()

# Find SQLite3 library
find_package(SQLite3 REQUIRED)
if(SQLite3_FOUND)
    message(STATUS "Found SQLite3 library: ${SQLite3_INCLUDE_DIRS} (version \"${SQLite3_VERSION}\")")
else()
    message(FATAL_ERROR "SQLite3 library not found. Please install SQLite3.")
endif()

# Find ZLIB
find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "Found ZLIB library: ${ZLIB_INCLUDE_DIRS} (version \"${ZLIB_VERSION_STRING}\")")
else()
    message(FATAL_ERROR "ZLIB library not found. Please install zlib.")
endif()

# Set SPARTA source directory location
set(SPARTA_DIR /Users/cmiao/Tools/map/sparta)
set(SIMDB_DIR ${SPARTA_DIR}/simdb)
set(SIMDB_INCLUDE_DIR ${SIMDB_DIR}/include)

# Set SPARTA and SIMDB library file locations
set(SPARTA_LIB_DIR ${SPARTA_DIR}/release)
set(SIMDB_LIB_DIR ${SPARTA_DIR}/release/simdb)

# Create directories in the build directory to mimic include structure
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/gemmini)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/utils)

# Create symbolic links for header files to support #include "gemmini/..." style includes
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/src/utils/common.hpp 
    ${CMAKE_BINARY_DIR}/include/gemmini/common.hpp
)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/src/gemmini.hpp 
    ${CMAKE_BINARY_DIR}/include/gemmini/gemmini.hpp
)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/src/execute/matrix.hpp 
    ${CMAKE_BINARY_DIR}/include/gemmini/matrix.hpp
)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/src/execute/matrix_multiplier.hpp 
    ${CMAKE_BINARY_DIR}/include/gemmini/matrix_multiplier.hpp
)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/src/execute/pe.hpp 
    ${CMAKE_BINARY_DIR}/include/gemmini/pe.hpp
)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/src/execute/systolic_array.hpp 
    ${CMAKE_BINARY_DIR}/include/gemmini/systolic_array.hpp
)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/src/utils/fifo.hpp 
    ${CMAKE_BINARY_DIR}/include/utils/fifo.hpp
)

# Add include directories - updated to include both src directory and the generated include directory
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/include
    ${SPARTA_DIR}
    ${SIMDB_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
    /opt/homebrew/include  # yaml-cpp header location
)

# Link SPARTA libraries
link_directories(
    ${SPARTA_LIB_DIR}
    ${SIMDB_LIB_DIR}
    /opt/homebrew/lib     # yaml-cpp library location
)

# Find source files for main application (excluding test files)
file(GLOB_RECURSE GEMMINI_SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# Exclude test files from main executable
list(FILTER GEMMINI_SOURCES EXCLUDE REGEX ".*tests/.*\\.cpp$")

# Add Gemmini simulator executable
add_executable(gemmini_simulator ${GEMMINI_SOURCES})

# Make sure symlinks are created before building the executable
add_custom_target(create_symlinks ALL)
add_dependencies(gemmini_simulator create_symlinks)

# Link SPARTA library, simdb library, and other dependencies
target_link_libraries(gemmini_simulator 
    sparta 
    simdb 
    ${Boost_LIBRARIES} 
    ${HDF5_LIBRARIES}
    yaml-cpp
    sqlite3
    ${ZLIB_LIBRARIES}
)

# Setup Google Test - use system-installed GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Enable testing
enable_testing()

# Define common test libraries
set(COMMON_TEST_LIBRARIES
    sparta 
    simdb 
    ${Boost_LIBRARIES} 
    ${HDF5_LIBRARIES}
    yaml-cpp
    sqlite3
    ${ZLIB_LIBRARIES}
    ${GTEST_LIBRARIES}
    pthread
)

# Create PE Google Test executable
set(PE_GTEST_SOURCES
    "${CMAKE_SOURCE_DIR}/src/tests/pe_gtest.cpp"
    "${CMAKE_SOURCE_DIR}/src/execute/pe.cpp"
)

add_executable(pe_gtest ${PE_GTEST_SOURCES})
add_dependencies(pe_gtest create_symlinks)

# Link PE Google Test with required libraries
target_link_libraries(pe_gtest ${COMMON_TEST_LIBRARIES})

# Create Systolic Array Google Test executable
set(SYSTOLIC_ARRAY_GTEST_SOURCES
    "${CMAKE_SOURCE_DIR}/src/tests/systolic_array_gtest.cpp"
    "${CMAKE_SOURCE_DIR}/src/execute/systolic_array.cpp"
    "${CMAKE_SOURCE_DIR}/src/execute/pe.cpp"
)

add_executable(systolic_array_gtest ${SYSTOLIC_ARRAY_GTEST_SOURCES})
add_dependencies(systolic_array_gtest create_symlinks)

# Link Systolic Array Google Test with required libraries
target_link_libraries(systolic_array_gtest ${COMMON_TEST_LIBRARIES})

# Create FIFO Test executable
set(FIFO_TEST_SOURCES
    "${CMAKE_SOURCE_DIR}/src/tests/fifo_test.cpp"
)

add_executable(fifo_test ${FIFO_TEST_SOURCES})
add_dependencies(fifo_test create_symlinks)

# Link FIFO Test with required libraries
target_link_libraries(fifo_test ${COMMON_TEST_LIBRARIES})

# Register Google Test
include(GoogleTest)
gtest_discover_tests(pe_gtest)
gtest_discover_tests(systolic_array_gtest)

# Install targets
install(TARGETS gemmini_simulator pe_gtest systolic_array_gtest fifo_test
    RUNTIME DESTINATION bin
)

# Copy configuration files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/config
     DESTINATION ${CMAKE_BINARY_DIR}/bin
)

# Print configuration information
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "SPARTA Directory: ${SPARTA_DIR}")
message(STATUS "SPARTA Library Directory: ${SPARTA_LIB_DIR}")
message(STATUS "SIMDB Directory: ${SIMDB_DIR}")
message(STATUS "SIMDB Include Directory: ${SIMDB_INCLUDE_DIR}")
message(STATUS "SIMDB Library Directory: ${SIMDB_LIB_DIR}")
message(STATUS "Boost Include Directory: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost Libraries: ${Boost_LIBRARIES}")
message(STATUS "HDF5 Include Directory: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 Libraries: ${HDF5_LIBRARIES}")
message(STATUS "YAML-CPP Include Directory: ${YAML_CPP_INCLUDE_DIR}")
message(STATUS "YAML-CPP Libraries: ${YAML_CPP_LIBRARIES}")
message(STATUS "SQLite3 Include Directory: ${SQLite3_INCLUDE_DIRS}")
message(STATUS "SQLite3 Libraries: ${SQLite3_LIBRARIES}")
message(STATUS "ZLIB Include Directory: ${ZLIB_INCLUDE_DIRS}")
message(STATUS "ZLIB Libraries: ${ZLIB_LIBRARIES}")
message(STATUS "Gemmini Sources: ${GEMMINI_SOURCES}")