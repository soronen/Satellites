CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
project(Satellites)

add_executable(parallel parallel.c)


# Here is an example syntax how to add compiler options to your build process
# See the project work document on compiler flag syntax on Linux and Windows

# target_compile_options(parallel PRIVATE "add-your-option-here")
# target_compile_options(parallel PRIVATE "add-your-second-option-here")

# merely log vectorization reports 
if (MSVC)
    target_compile_options(parallel PRIVATE "/Qvec-report:2")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_compile_options(parallel PRIVATE "-Rpass=loop-vectorize" "-Rpass-missed=loop-vectorize" "-Rpass-analysis=loop-vectorize")
endif()

# enable performance related flags
if (MSVC)
    target_compile_options(parallel PRIVATE 
        "/Qvec-report:2" 
        # "/fp:fast" 
        # "/arch:AVX2"     # Enables AVX2 (FMA3 will be used automatically if appropriate)
        # "/arch:AVX512"  
        #"/Ox"
        #"/GL"
        # "/O2"
    )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_compile_options(parallel PRIVATE 
        "-Rpass=loop-vectorize"
        "-Rpass-missed=loop-vectorize"
        "-Rpass-analysis=loop-vectorize"
        # "-ffast-math"
        # "-march=native"  # This will automatically enable both AVX2 and FMA3 based on the CPU
        # "-mavx2"         # Explicitly enable AVX2
        # "-mfma"          # Explicitly enable FMA3
        #"-mavx512f"      # Enable AVX-512 Foundation
        # "-Ofast"
        # "-mpreferred-stack-boundary=2"
        # "-fno-stack-protector" 
        # "-fomit-frame-pointer"
        # "-fno-strict-aliasing"
        # "-funsafe-loop-optimizations"
    )
endif()


# UNCOMMENT THESE TO ENABLE OPENMP
#
# find_package(OpenMP REQUIRED)
# target_link_libraries(parallel OpenMP::OpenMP_C)


# UNCOMMENT THESE TO ENABLE OPENCL
# This will also copy the kernel source file parallel.cl (created by you) to the build directory
# The copying command is unfortunately not perfect, as it doesn't redo the copy if you only edit
# the parallel.cl, but leave the parallel.c untouched.
# Because of this, you might need to force 'Rebuild All' to ensure kernel code updates propagate
# to the build directory. If you know enough CMake magic, feel free to fix this and let me know.
#
# find_package(OpenCL REQUIRED)
# target_include_directories(parallel PRIVATE ${OpenCL_INCLUDE_DIRS})
# target_link_libraries(parallel ${OpenCL_LIBRARIES})
# add_custom_command(
#     TARGET parallel POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${CMAKE_SOURCE_DIR}/parallel.cl"
#     $<TARGET_FILE_DIR:parallel>
#     VERBATIM)


# Find and link SDL2
if (WIN32)
    set(SDL2_DIR "${CMAKE_SOURCE_DIR}/SDL2-2.30.7/cmake")
endif()
find_package(SDL2 REQUIRED)
target_include_directories(parallel PRIVATE ${SDL2_INCLUDE_DIR})
target_link_libraries(parallel ${SDL2_LIBRARIES})

if (WIN32)
# copy the sdl2 .dll file to the same folder as the executable
add_custom_command(
    TARGET parallel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL2::SDL2>
    $<TARGET_FILE_DIR:parallel>
    VERBATIM)
else()
    # Math library shouldn't be linked on Windows, but must be linked on Linux
    target_link_libraries(parallel m)
endif()
