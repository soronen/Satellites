#pragma once

// These are used to decide the window size
#define WINDOW_WIDTH  1920
#define WINDOW_HEIGHT 1080
#define WINDOW_SIZE WINDOW_WIDTH*WINDOW_HEIGHT

// The number of satellites can be changed to see how it affects performance.
// Benchmarks must be run with the original number of satellites
#define SATELLITE_COUNT 64

// These are used to control the satellite movement
#define SATELLITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f
#define DELTATIME 32
#define PHYSICSUPDATESPERFRAME 100000
#define BLACK_HOLE_RADIUS 4.5f

// def work group size
#define WORK_GROUP_SIZE {16, 8}

# define WORK_GROUP_SIZE_PHYSICS 64

//  physics kernel even without writing back to the cpu is 50ms compared to 10ms when running it on the cpu
#define USE_PHYSICS_KERNEL 0 // 0 (cpu ) or 1 (gpu)