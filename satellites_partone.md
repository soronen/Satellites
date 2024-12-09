# COMP.CE.350 Multicore and GPU Programming Project Work

### Eetu Soronen - 152830694

## 6.1

I did all of the exercises on my a Windows PC with AMD Ryzen 7800X3D CPU and Nvidia RTX 4070 GPU. I ran the program and did benchmarks with Visual Studio 2022 enterprise. Profiles and commands for running the program with various flags should be found in CMakeSettings.json. I downloaded Clang (clang_cl_x86_64) through Visual Studio UI. Likewise i compiled the code with visual studio ui

Compiler version info:

```cmd
C:\Program Files\Microsoft Visual Studio\2022\Enterprise>clang --version
clang version 17.0.3
Target: i686-pc-windows-msvc
Thread model: posix
InstalledDir: C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm\bin

C:\Program Files\Microsoft Visual Studio\2022\Enterprise>cl
Microsoft (R) C/C++ Optimizing Compiler Version 19.41.34120 for x86
Copyright (C) Microsoft Corporation.  All rights reserved.

usage: cl [ option... ] filename... [ /link linkoption... ]

C:\Program Files\Microsoft Visual Studio\2022\Enterprise>
```

### Ex 6.1.1 & 6.2

Unmodified code run with MSVC and Clang compilers on Visual Studio (Windows) with Release and Debug modes.

| Column 1          | MSVC Debug | MSVC Release | Clang Debug | Clang Release |
| ----------------- | ---------- | ------------ | ----------- | ------------- |
| Average frametime | 1603ms     | 859ms        | 2056ms      | 702ms         |

MSVC is predictably slower than Clang, except in debug mode which was surprising. Average frametime was taken around 1 minute after starting the program for each test.

### Ex 6.1.3

#### MSVC

- On MSVC, for ParallelPhysicsEngine the output is:

```bash
  --- Analyzing function: parallelPhysicsEngine
  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(126) :
  info C5002: loop not vectorized due to reason '1300'

  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(139) :
  info C5002: loop not vectorized due to reason '1203'

  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(135) :
  info C5002: loop not vectorized due to reason '1106'

  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(174) :
  info C5002: loop not vectorized due to reason '1300'
```

- Reason code 1300 means “too little computation”, 1106 “outer loop not vectorized”, and 1203 “Loop body includes non-contiguous accesses into an array”. Loop on line 139 might be optimizable by modifying the array, the rest not so much.

```bash
  --- Analyzing function: parallelGraphicsEngine
  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(223) :
  info C5002: loop not vectorized due to reason '500'

  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(248) :
  info C5002: loop not vectorized due to reason '1104'

  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(193) :
  info C5002: loop not vectorized due to reason '1106'
```

- For parallelGraphicsEngine, the first loop is not vectorized because it has multiple exits, and the second is passed because it modifies scalar variables (likely renderColor).

#### Clang

- Looking at the same function as above, the logs show that Clang is able to vectorize one loop, which is the one containing the non-contiguous access into an array. Otherwise the results are similiar.

```c++
    # parallelPhysicsEngine
    C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(139,7):
    remark: the cost-model indicates that interleaving is not beneficial
     [-Rpass-analysis=loop-vectorize]
	139 |   	for(i = 0; i < SATELLITE_COUNT; ++i){
    	|   	^
    C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(139,7):
    remark: vectorized loop
    (vectorization width: 2, interleaved count: 1) [-Rpass=loop-vectorize]


  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(126,4):
  remark: the cost-model indicates that vectorization is not beneficial
  [-Rpass-missed=loop-vectorize]
	126 |	for (idx = 0; idx < SATELLITE_COUNT; ++idx) {
    	|	^

  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(126,4):
  remark: the cost-model indicates that interleaving is not beneficial
  [-Rpass-missed=loop-vectorize]

  C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(174,4):
  remark: loop not vectorized [-Rpass-missed=loop-vectorize]
    174 |    for (idx2 = 0; idx2 < SATELLITE_COUNT; ++idx2) {
        |    ^
```

```c++
  # parallelGraphicsEngine
    C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(223,7):
    remark: loop not vectorized:
    value that could not be identified as reduction is used outside the loop
    [-Rpass-analysis=loop-vectorize]
    223 |       for(j = 0; j < SATELLITE_COUNT; ++j){
        |       ^

    C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(223,7):
    remark: loop not vectorized: could not determine number of loop iterations
    [-Rpass-analysis=loop-vectorize]

    C:\Users\Eetu\Projects\ylilipasto\Satellites\parallel.c(248,10):
    remark: loop not vectorized [-Rpass-missed=loop-vectorize]
    248 |          for(k = 0; k < SATELLITE_COUNT; ++k){
        |          ^
```

- Clang was not able to vectorize anything in parallelGraphicsEngine either

### Ex 6.1.4

Looking at my CPU, it does apparently support many extensions up to AVX512, so I'll try those on top of fastmath

| Instructions        | MSVC Release | Clang Release |
| ------------------- | ------------ | ------------- |
| AVX512 with rest    | 506ms        | 611ms         |
| rest without AVX512 | 531ms        | 622ms         |
| fastmath only       | 486ms        | 522ms         |
| -0fast or /02       | 484ms        | 554ms         |

#### Portion from my CMake containing the performance flags i tried:

```bash
# enable performance related flags
if (MSVC)
    target_compile_options(parallel PRIVATE
        "/Qvec-report:2"
        "/fp:fast"
        "/arch:AVX2"
        "/arch:AVX512"   # enable or disable
    )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_compile_options(parallel PRIVATE
        "-Rpass=loop-vectorize"
        "-Rpass-missed=loop-vectorize"
        "-Rpass-analysis=loop-vectorize"
        "-ffast-math"
        "-march=native"
        "-mavx2"
        "-mfma"
        "-mavx512f"      # enable or disable
    )
endif()
```

### Ex 6.1.5

- Running with fastmath only or 0fast gave biggest performance boost for both Clang and MSVC. MSVC was also suprisingly faster than Clang in all cases on my environment.

- Fast math works by making assumptions such as x==x, which may not be true in all cases. This flag breaks the IEEE floating point standard and can cause bugs, but also gives a performance boost.

- -0fast on Clang or /02 on MSVC are collections performance flags, including fastmath and probably some instruction extensions. For testing these I disabled all other flags, and similarly for testing other rows I disabled these.

- AVX512 and other similar flags I enabled allow the compiler to use special SIMD instructions that may not be available in all x86_64 CPUs, which is why they are not enabled by default. Some processors like mine in the case of AVX512 also emulate those instructions without dedicated hardware, so performance may not be increased by using them.

### Ex 6.1.6

On Clang, I found online some flags like -funsafe-optimizations that may have done so, but that is seemingly not supported by Visual Studio Clang. Same for many others like -fno-strict-aliasing, -fomit-frame-pointer, "-fno-stack-protector, or -mpreferred-stack-boundary=2

On MSVC, enabling flags like /02 /0x did cause objects in the program to move a little differently but they didn't crash either. I also tried /GL which is whole program optimization flag that supposedly doesn't work with incompatible modules. Worked fine on my machine though.

And every flag i tried compiled without issues.

### Ex 6.2

In parallelGraphicsEngine() it's possible to get rid of all square roots by instead squaring the other value in comparisons and such. Some of these squares can also be calculated once instead of every loop. With this, MSVC with no optimzation flags on release mode goes from average 867ms -> 592ms.

In phyicsEngine() moving DELTATIME / PHYSICSUPDATESPERFRAME outside loop and using 1/sqrt() instead of sqrt to normalize the direction brought the average speed to 560ms.

### Ex 6.3

ParallelPhysicsEngine:

- Physics iteration loop can't be run in multiple threads as is because tmpVelocity and tmpPosition accumulate values.

- Physics satellite loop should be able to run in multiple threads because satellites don't affect each other. Though the loop is so small it may not be beneficial.

ParallelGraphicsEngine:

- Graphics pixel loop can be run in multiple threads. drawing one pixel doesn't affect the other.

- Graphics satellite loop accumulates weight value. It might be able to be parallelized but the benefit may not be great because it uses reduction and the outer loop will be parallelized anyway.

---

**Can you transform the code in some way (change the code without affecting the end results) which either allows parallelization of a loop which originally was not parallelizable, or makes a loop which originally was not initially beneficial to parallelize with OpenMP beneficial to parallelize with OpenMP?**

**If yes, explain your code transformation?**

**Does your code transformation have any effect on vectorization performed by the compiler?**

- In parallelphysicsengine, changing the tmpVel and tmpPos from doublevector structs into normal variables allows Clang to vectorize physics satellite loop which was not done previously. It also speeds up little from 399ms -> 390ms. MSVC sees no difference. It may or may not affect its parallelism aspirations.

```c++
    //So changing
   for (idx = 0; idx < SATELLITE_COUNT; ++idx) {
       tmpPosition[idx].x = satellites[idx].position.x;
       tmpPosition[idx].y = satellites[idx].position.y;
       tmpVelocity[idx].x = satellites[idx].velocity.x;
       tmpVelocity[idx].y = satellites[idx].velocity.y;
   }

   // into:
   for (idx = 0; idx < SATELLITE_COUNT; ++idx) {
       tmpPosX[idx] = satellites[idx].position.x;
       tmpPosY[idx] = satellites[idx].position.y;
       tmpVelX[idx] = satellites[idx].velocity.x;
       tmpVelY[idx] = satellites[idx].velocity.y;
   }
```

- Surefire and simple way to make parallelizing any shorter loop beneficial would be to make the loops larger. So for example changing the satellite count to 1000.

### Ex 6.4

#### benchmarks before and after adding openmp

| Optimizations                      | MSVC Release | Clang Release |
| ---------------------------------- | ------------ | ------------- |
| no optimization flags              | 560ms        | 394ms         |
| "/fp:fast" or "-ffast-math"        | 371ms        | 269ms         |
| adding openmp pragamas & ffastmath | 120ms        | 48ms          |

MSVC also crashed because i had put some for loop incrementor declarations inside the for loop. (like for(int i = 0) instead of int i; for(i = 0)) So i had to revert those.
Clang had no issues with it though.

I parallelized in parallelGraphicsEngine the pixel loop and in parallelPhysics engine the loops loading and saving my temp variables before and after the main loop. I did also try parallelizing the main physics loop, but it actually hurt performance. trying more complex pragmas like scheduling(static) or dynamic didn't make a difference.

Since I only have 8 cores to work with, using those for graphicsEngine is more beneficial since drawing pixels can be parallelized embarassingly, while physics calculations are more complex and involve accumulators and waiting for previous loop results. My CPU usage is already at 100% so any more parallelization even if it would work, has no resources to use and is therefore not beneficial.

Did my performance scale with the amount of cores? 269 / 48 = 5.6, and my core count is 8. So the parallelization did have some overhead, but overall it's a big improvement. There can be a lot of reasons for the overhead, creating new threads always has a cost, reading data from memory and cache misses become more common with more threads. Synchronizing data and waiting for threads also surely happens at least a little, though my cpu usage stays consistently at 100% for all threads. And there are also big parts still done serially (like the physics calculation), which mean that the graphics would theoretically have to wait for those to finish if the drawing was fast enough.

It took me maybe 10-15 hours to complete this
