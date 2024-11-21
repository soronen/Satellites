/* COMP.CE.350 Parallelization Excercise 2024
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
					  Heikki Kultala heikki.kultala@tut.fi
					  Topi Leppanen  topi.leppanen@tuni.fi

VERSION 1.1 - updated to not have stuck satellites so easily
VERSION 1.2 - updated to not have stuck satellites hopefully at all.
VERSION 19.0 - make all satellites affect the color with weighted average.
			   add physic correctness check.
VERSION 20.0 - relax physic correctness check
VERSION 24.0 - port to SDL2
*/


#ifdef _WIN32
#include "SDL.h"
#else
#include "SDL2/SDL.h"
#endif

#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define CL_TARGET_OPENCL_VERSION 300

#ifndef __APPLE__
#include <CL/cl.h>
#else
#include <OpenCL/cl.h>
#endif

int mousePosX;
int mousePosY;


#include "constants.h"



// Stores 2D data like the coordinates
typedef struct {
	float x;
	float y;
} floatvector;

// Stores 2D data like the coordinates
typedef struct {
	double x;
	double y;
} doublevector;

// Each float may vary from 0.0f ... 1.0f
typedef struct {
	float blue;
	float green;
	float red;
} color_f32;

// Stores rendered colors. Each value may vary from 0 ... 255
typedef struct {
	uint8_t blue;
	uint8_t green;
	uint8_t red;
	uint8_t reserved;
} color_u8;

// Stores the satellite data, which fly around black hole in the space
typedef struct {
	color_f32 identifier;
	floatvector position;
	floatvector velocity;
} satellite;

// Pixel buffer which is rendered to the screen
color_u8* pixels;

// Pixel buffer which is used for error checking
color_u8* correctPixels;

// Buffer for all satellites in the space
satellite* satellites;
satellite* backupSatelites;




//
//// ## You may add your own variables here ##
const double DELTATIME_PER_PHYSICSUPDATESPERFRAME = (double)DELTATIME / PHYSICSUPDATESPERFRAME;

// set work group size
size_t localWorkSize[2] = WORK_GROUP_SIZE;

// global work size must be multiple of local work size and also depends on the window size....
size_t globalWorkSize[2];

// physics work size 
size_t localWorkSizePhysics = WORK_GROUP_SIZE_PHYSICS;
size_t globalWorkSizePhysics;


int satelliteCount = SATELLITE_COUNT;
int windowWidth = WINDOW_WIDTH;
int windowHeight = WINDOW_HEIGHT;
float blackHoleRadius = BLACK_HOLE_RADIUS;
float satelliteRadius = SATELLITE_RADIUS;

float blackHoleRadiusSquared = BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS;
float satelliteRadiusSquared = SATELLITE_RADIUS * SATELLITE_RADIUS;


cl_context context;
cl_command_queue commandQueue;
cl_program program;
cl_kernel graphicsKernel;
cl_kernel physicsKernel;

cl_mem pixelBuffer;
cl_mem satelliteBuffer;

const int PLATFORM_INDEX = 0;
const int DEVICE_INDEX = 0;

const char* openclErrors[] = {
	"Success!",
	"Device not found.",
	"Device not available",
	"Compiler not available",
	"Memory object allocation failure",
	"Out of resources",
	"Out of host memory",
	"Profiling information not available",
	"Memory copy overlap",
	"Image format mismatch",
	"Image format not supported",
	"Program build failure",
	"Map failure",
	"Invalid value",
	"Invalid device type",
	"Invalid platform",
	"Invalid device",
	"Invalid context",
	"Invalid queue properties",
	"Invalid command queue",
	"Invalid host pointer",
	"Invalid memory object",
	"Invalid image format descriptor",
	"Invalid image size",
	"Invalid sampler",
	"Invalid binary",
	"Invalid build options",
	"Invalid program",
	"Invalid program executable",
	"Invalid kernel name",
	"Invalid kernel definition",
	"Invalid kernel",
	"Invalid argument index",
	"Invalid argument value",
	"Invalid argument size",
	"Invalid kernel arguments",
	"Invalid work dimension",
	"Invalid work group size",
	"Invalid work item size",
	"Invalid global offset",
	"Invalid event wait list",
	"Invalid event",
	"Invalid operation",
	"Invalid OpenGL object",
	"Invalid buffer size",
	"Invalid mip-map level",
	"Unknown",
};


const char* clErrorString(cl_int e)
{
	switch (e) {
	case CL_SUCCESS:                            return openclErrors[0];
	case CL_DEVICE_NOT_FOUND:                   return openclErrors[1];
	case CL_DEVICE_NOT_AVAILABLE:               return openclErrors[2];
	case CL_COMPILER_NOT_AVAILABLE:             return openclErrors[3];
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return openclErrors[4];
	case CL_OUT_OF_RESOURCES:                   return openclErrors[5];
	case CL_OUT_OF_HOST_MEMORY:                 return openclErrors[6];
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return openclErrors[7];
	case CL_MEM_COPY_OVERLAP:                   return openclErrors[8];
	case CL_IMAGE_FORMAT_MISMATCH:              return openclErrors[9];
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return openclErrors[10];
	case CL_BUILD_PROGRAM_FAILURE:              return openclErrors[11];
	case CL_MAP_FAILURE:                        return openclErrors[12];
	case CL_INVALID_VALUE:                      return openclErrors[13];
	case CL_INVALID_DEVICE_TYPE:                return openclErrors[14];
	case CL_INVALID_PLATFORM:                   return openclErrors[15];
	case CL_INVALID_DEVICE:                     return openclErrors[16];
	case CL_INVALID_CONTEXT:                    return openclErrors[17];
	case CL_INVALID_QUEUE_PROPERTIES:           return openclErrors[18];
	case CL_INVALID_COMMAND_QUEUE:              return openclErrors[19];
	case CL_INVALID_HOST_PTR:                   return openclErrors[20];
	case CL_INVALID_MEM_OBJECT:                 return openclErrors[21];
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return openclErrors[22];
	case CL_INVALID_IMAGE_SIZE:                 return openclErrors[23];
	case CL_INVALID_SAMPLER:                    return openclErrors[24];
	case CL_INVALID_BINARY:                     return openclErrors[25];
	case CL_INVALID_BUILD_OPTIONS:              return openclErrors[26];
	case CL_INVALID_PROGRAM:                    return openclErrors[27];
	case CL_INVALID_PROGRAM_EXECUTABLE:         return openclErrors[28];
	case CL_INVALID_KERNEL_NAME:                return openclErrors[29];
	case CL_INVALID_KERNEL_DEFINITION:          return openclErrors[30];
	case CL_INVALID_KERNEL:                     return openclErrors[31];
	case CL_INVALID_ARG_INDEX:                  return openclErrors[32];
	case CL_INVALID_ARG_VALUE:                  return openclErrors[33];
	case CL_INVALID_ARG_SIZE:                   return openclErrors[34];
	case CL_INVALID_KERNEL_ARGS:                return openclErrors[35];
	case CL_INVALID_WORK_DIMENSION:             return openclErrors[36];
	case CL_INVALID_WORK_GROUP_SIZE:            return openclErrors[37];
	case CL_INVALID_WORK_ITEM_SIZE:             return openclErrors[38];
	case CL_INVALID_GLOBAL_OFFSET:              return openclErrors[39];
	case CL_INVALID_EVENT_WAIT_LIST:            return openclErrors[40];
	case CL_INVALID_EVENT:                      return openclErrors[41];
	case CL_INVALID_OPERATION:                  return openclErrors[42];
	case CL_INVALID_GL_OBJECT:                  return openclErrors[43];
	case CL_INVALID_BUFFER_SIZE:                return openclErrors[44];
	case CL_INVALID_MIP_LEVEL:                  return openclErrors[45];
	default:                                    return openclErrors[46];
	}
}
// This function reads in a text file and stores it as a char pointer
char*
readSource(char* kernelPath) {
	cl_int status;
	FILE* fp;
	char* source;
	long int size;
	printf("Program file is: %s\n", kernelPath);

	//fp = fopen(kernelPath, "rb");

	errno_t err = fopen_s(&fp, kernelPath, "rb");

	if (err != 0 || fp == NULL) {
		fprintf(stderr, "Error: Failed to open file %s\n", kernelPath);
		exit(EXIT_FAILURE);
	}


	if (!fp) {
		printf("Could not open kernel file\n");
		exit(-1);
	}
	status = fseek(fp, 0, SEEK_END);
	if (status != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}
	size = ftell(fp);
	if (size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}
	rewind(fp);
	source = (char*)malloc(size + 1);
	if (source == NULL) {
		printf("Error allocating space for the kernel source\n");
		exit(-1);
	}
	size_t readBytes = fread(source, 1, size, fp);
	if ((long int)readBytes != size) {
		printf("Error reading the kernel file\n");
		exit(-1);
	}
	source[size] = '\0';
	fclose(fp);
	return source;
}

// Informational printing
void
printPlatformInfo(cl_platform_id* platformId, size_t ret_num_platforms) {
	size_t infoLength = 0;
	char* infoStr = NULL;
	cl_int status;
	for (unsigned int r = 0; r < (unsigned int)ret_num_platforms; ++r) {
		printf("Platform %d information:\n", r);
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_PROFILE, 0, NULL, &infoLength);
		if (status != CL_SUCCESS) {
			printf("Platform profile length error: %s\n", clErrorString(status));
		}
		infoStr = malloc((infoLength) * sizeof(char));
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_PROFILE, infoLength, infoStr, NULL);
		if (status != CL_SUCCESS) {
			printf("Platform profile info error: %s\n", clErrorString(status));
		}
		printf("\tProfile: %s\n", infoStr);
		free(infoStr);
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_VERSION, 0, NULL, &infoLength);
		if (status != CL_SUCCESS) {
			printf("Platform version length error: %s\n", clErrorString(status));
		}
		infoStr = malloc((infoLength) * sizeof(char));
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_VERSION, infoLength, infoStr, NULL);
		if (status != CL_SUCCESS) {
			printf("Platform version info error: %s\n", clErrorString(status));
		}
		printf("\tVersion: %s\n", infoStr);
		free(infoStr);
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_NAME, 0, NULL, &infoLength);
		if (status != CL_SUCCESS) {
			printf("Platform name length error: %s\n", clErrorString(status));
		}
		infoStr = malloc((infoLength) * sizeof(char));
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_NAME, infoLength, infoStr, NULL);
		if (status != CL_SUCCESS) {
			printf("Platform name info error: %s\n", clErrorString(status));
		}
		printf("\tName: %s\n", infoStr);
		free(infoStr);
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_VENDOR, 0, NULL, &infoLength);
		if (status != CL_SUCCESS) {
			printf("Platform vendor info length error: %s\n", clErrorString(status));
		}
		infoStr = malloc((infoLength) * sizeof(char));
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_VENDOR, infoLength, infoStr, NULL);
		if (status != CL_SUCCESS) {
			printf("Platform vendor info error: %s\n", clErrorString(status));
		}
		printf("\tVendor: %s\n", infoStr);
		free(infoStr);
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_EXTENSIONS, 0, NULL, &infoLength);
		if (status != CL_SUCCESS) {
			printf("Platform extensions info length error: %s\n", clErrorString(status));
		}
		infoStr = malloc((infoLength) * sizeof(char));
		status = clGetPlatformInfo(platformId[r], CL_PLATFORM_EXTENSIONS, infoLength, infoStr, NULL);
		if (status != CL_SUCCESS) {
			printf("Platform extensions info error: %s\n", clErrorString(status));
		}
		printf("\tExtensions: %s\n", infoStr);
		free(infoStr);
	}
	printf("\nUsing Platform %d.\n", PLATFORM_INDEX);
}

// Informational printing
void
printDeviceInfo(cl_device_id* deviceIds, size_t ret_num_devices) {
	// Print info about the devices
	size_t infoLength = 0;
	char* infoStr = NULL;
	cl_int status;

	for (unsigned int r = 0; r < ret_num_devices; ++r) {
		printf("Device %d indormation:\n", r);
		status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_VENDOR, 0, NULL, &infoLength);
		if (status != CL_SUCCESS) {
			printf("Device Vendor info length error: %s\n", clErrorString(status));
		}
		infoStr = malloc((infoLength) * sizeof(char));
		status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_VENDOR, infoLength, infoStr, NULL);
		if (status != CL_SUCCESS) {
			printf("Device Vendor info error: %s\n", clErrorString(status));
		}
		printf("\tVendor: %s\n", infoStr);
		free(infoStr);
		status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_NAME, 0, NULL, &infoLength);
		if (status != CL_SUCCESS) {
			printf("Device name info length error: %s\n", clErrorString(status));
		}
		infoStr = malloc((infoLength) * sizeof(char));
		status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_NAME, infoLength, infoStr, NULL);
		if (status != CL_SUCCESS) {
			printf("Device name info error: %s\n", clErrorString(status));
		}
		printf("\tName: %s\n", infoStr);
		free(infoStr);
		status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_VERSION, 0, NULL, &infoLength);
		if (status != CL_SUCCESS) {
			printf("Device version info length error: %s\n", clErrorString(status));
		}
		infoStr = malloc((infoLength) * sizeof(char));
		status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_VERSION, infoLength, infoStr, NULL);
		if (status != CL_SUCCESS) {
			printf("Device version info error: %s\n", clErrorString(status));
		}
		printf("\tVersion: %s\n", infoStr);
		free(infoStr);
	}
	printf("\nUsing Device %d.\n", DEVICE_INDEX);
}




// ## You may add your own initialization routines here ##
void init() {

	// Start the OpenCL initialization
	cl_int status;  // Use this to check the output of each API call

	// Get available OpenCL platforms
	cl_uint ret_num_platforms;
	status = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	if (status != CL_SUCCESS) {
		printf("Error getting the number of platforms: %s", clErrorString(status));
	}
	cl_platform_id* platformId = malloc(sizeof(cl_platform_id) * ret_num_platforms);
	status = clGetPlatformIDs(ret_num_platforms, platformId, NULL);
	if (status != CL_SUCCESS) {
		printf("Error getting the platforms: %s", clErrorString(status));
	}

	// Print info about the platform. Not needed for functionality,
	// but nice to see in order to confirm your OpenCL installation
	printPlatformInfo(platformId, ret_num_platforms);

	// Get available devices
	cl_uint ret_num_devices = 0;
	status = clGetDeviceIDs(
		platformId[PLATFORM_INDEX], CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
	if (status != CL_SUCCESS) {
		printf("Error getting the number of devices: %s", clErrorString(status));
	}
	cl_device_id* deviceIds = malloc((ret_num_devices) * sizeof(cl_device_id));
	status = clGetDeviceIDs(
		platformId[PLATFORM_INDEX], CL_DEVICE_TYPE_ALL, ret_num_devices, deviceIds, &ret_num_devices);
	if (status != CL_SUCCESS) {
		printf("Error getting device ids: %s", clErrorString(status));
	}

	// Again, this only prints nice-to-know information
	printDeviceInfo(deviceIds, ret_num_devices);

	context = clCreateContext(NULL, 1, &(deviceIds[DEVICE_INDEX]), NULL, NULL, &status);
	if (status != CL_SUCCESS) {
		printf("Context creation error: %s\n", clErrorString(status));
	}

	// In order command queue
	// Using the 1.2 clCreateCommandQueue API since it's bit simpler,
	// this was later deprecated in OpenCL 2.0
	//commandQueue = clCreateCommandQueue(context, deviceIds[DEVICE_INDEX], 0, &status);

	cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
	commandQueue = clCreateCommandQueueWithProperties(context, deviceIds[DEVICE_INDEX], properties, &status);

	if (status != CL_SUCCESS) {
		printf("Command queue creation error: %s", clErrorString(status));
	}

	// Make kernel string into a program
	const char* programSource = readSource("parallel.cl");
	program = clCreateProgramWithSource(context, 1, &programSource, NULL, &status);
	if (status != CL_SUCCESS) {
		printf("Program creation error: %s", clErrorString(status));
	}

	// Program compiling
	status = clBuildProgram(program, 1, &deviceIds[DEVICE_INDEX], NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		printf("OpenCL build error: %s\n", clErrorString(status));
		// Fetch build errors if there were some.
		if (status == CL_BUILD_PROGRAM_FAILURE) {
			size_t infoLength = 0;
			cl_int cl_build_status = clGetProgramBuildInfo(
				program, deviceIds[DEVICE_INDEX], CL_PROGRAM_BUILD_LOG, 0, 0, &infoLength);
			if (cl_build_status != CL_SUCCESS) {
				printf("Build log length fetch error: %s\n", clErrorString(cl_build_status));
			}
			char* infoStr = malloc(infoLength * sizeof(char));
			cl_build_status = clGetProgramBuildInfo(
				program, deviceIds[DEVICE_INDEX], CL_PROGRAM_BUILD_LOG, infoLength, infoStr, 0);
			if (cl_build_status != CL_SUCCESS) {
				printf("Build log fetch error: %s\n", clErrorString(cl_build_status));
			}

			printf("OpenCL build log:\n %s", infoStr);
			free(infoStr);
		}
		abort();
	}

	// Create the graphicsKernel
	graphicsKernel = clCreateKernel(program, "graphicsKernel", &status);
	if (status != CL_SUCCESS) {
		printf("Kernel creation error: %s\n", clErrorString(status));
	}

	// create physics kernel
	physicsKernel = clCreateKernel(program, "physicsKernel", &status);
	if (status != CL_SUCCESS) {
		printf("Kernel creation error: %s\n", clErrorString(status));
	}

	// create buffers
	satelliteBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(satellite) * SATELLITE_COUNT, NULL, &status);
	if (status != CL_SUCCESS) {
		printf("Error creating satellite buffer: %s\n", clErrorString(status));
		abort();
	}

#if USE_PHYSICS_KERNEL == 1
	// Write the initial satellite data to the GPU. if parallel physics engine is used, this is not needed
	status = clEnqueueWriteBuffer(commandQueue, satelliteBuffer, CL_TRUE, 0,
		sizeof(satellite) * SATELLITE_COUNT, satellites, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		printf("Error writing initial satellite data: %s\n", clErrorString(status));
		abort();
	}
#endif

	pixelBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(color_u8) * WINDOW_SIZE, NULL, &status);
	if (status != CL_SUCCESS) {
		printf("Error creating pixel buffer: %s\n", clErrorString(status));
		abort();
	}

	// get max work group size, calculate how many threads will be out of bounds
	size_t maxWorkGroupSize;
	status = clGetDeviceInfo(deviceIds[DEVICE_INDEX], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
	if (status != CL_SUCCESS) {
		printf("Error getting max work group size: %s\n", clErrorString(status));
		abort();
	}
	printf("Max work group size: %d\n", maxWorkGroupSize);

	if (localWorkSize[0] * localWorkSize[1] > maxWorkGroupSize) {
		printf("Error: Work group size exceeds the maximum supported size of %zu\n", maxWorkGroupSize);
		abort();
	}

	// pad the global work size to be multiple of local work size
	globalWorkSize[0] = ((WINDOW_WIDTH + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
	globalWorkSize[1] = ((WINDOW_HEIGHT + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
	size_t totalWorkItems = globalWorkSize[0] * globalWorkSize[1];
	printf("Total work items: %d\n", totalWorkItems);

	size_t totalValidPixels = windowWidth * windowHeight;
	printf("Total valid pixels: %d\n", totalValidPixels);

	size_t outOfBoundsThreads = totalWorkItems - totalValidPixels;
	printf("Out of bounds threads: %d\n", outOfBoundsThreads);

	// same for physics work sizes
	globalWorkSizePhysics = ((SATELLITE_COUNT + localWorkSizePhysics - 1) / localWorkSizePhysics) * localWorkSizePhysics;
	printf("Total work items physics: %d\n", globalWorkSizePhysics);
	size_t outOfBoundsThreadsPhysics = globalWorkSizePhysics - SATELLITE_COUNT;
	printf("Out of bounds threads physics: %d\n", outOfBoundsThreadsPhysics);


	free((void*)programSource);
	free(platformId);
	free(deviceIds);
}



void parallelPhyicsEngineGPU() {
	cl_int status;

	// Set kernel arguments
	status = clSetKernelArg(physicsKernel, 0, sizeof(cl_mem), &satelliteBuffer);
	status |= clSetKernelArg(physicsKernel, 1, sizeof(int), &mousePosX);
	status |= clSetKernelArg(physicsKernel, 2, sizeof(int), &mousePosY);

	if (status != CL_SUCCESS) {
		printf("Error setting kernel arguments: %s\n", clErrorString(status));
		abort();
	}

	status = clEnqueueNDRangeKernel(commandQueue, physicsKernel, 1, NULL,
		&globalWorkSizePhysics, &localWorkSizePhysics, 0, NULL, NULL);

	if (status != CL_SUCCESS) {
		printf("Error executing kernel: %s\n", clErrorString(status));
		abort();
	}

	// Read back results from the pixel buffer after kernel execution
	//status = clEnqueueReadBuffer(commandQueue, satelliteBuffer, CL_TRUE, 0,
	//	sizeof(satellite) * satelliteCount, satellites, 0, NULL, NULL);
	//if (status != CL_SUCCESS) {
	//	printf("Error reading results: %s\n", clErrorString(status));
	//	abort();
	//}

	// write it back only for error checking in first 2 frames. after that no need
	extern unsigned int frameNumber;
	if (frameNumber < 2) {
		// Ensure the kernel has finished executing before reading back data
		clFinish(commandQueue);

		status = clEnqueueReadBuffer(commandQueue, satelliteBuffer, CL_TRUE, 0,
			sizeof(satellite) * satelliteCount, satellites, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("Error reading satellite data: %s\n", clErrorString(status));
			abort();
		}
	}
}


// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satellites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once
void parallelPhysicsEngine() {
	if (USE_PHYSICS_KERNEL) {
		parallelPhyicsEngineGPU();
		return;
	}

	int tmpMousePosX = mousePosX;
	int tmpMousePosY = mousePosY;

	double tmpPosX[SATELLITE_COUNT];
	double tmpPosY[SATELLITE_COUNT];
	double tmpVelX[SATELLITE_COUNT];
	double tmpVelY[SATELLITE_COUNT];

	int idx;
#pragma omp parallel for
	for (idx = 0; idx < SATELLITE_COUNT; ++idx) {
		tmpPosX[idx] = satellites[idx].position.x;
		tmpPosY[idx] = satellites[idx].position.y;
		tmpVelX[idx] = satellites[idx].velocity.x;
		tmpVelY[idx] = satellites[idx].velocity.y;
	}
	// Physics iteration loop
	int physicsUpdateIndex;
	for (physicsUpdateIndex = 0;
		physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
		++physicsUpdateIndex) {
		// Physics satellite loop
		for (int i = 0; i < SATELLITE_COUNT; ++i) {

			// Distance to the blackhole (bit ugly code because C-struct cannot have member functions)
			doublevector positionToBlackHole = { .x = tmpPosX[i] -
			   tmpMousePosX, .y = tmpPosY[i] - tmpMousePosY };
			double distToBlackHoleSquared =
				positionToBlackHole.x * positionToBlackHole.x +
				positionToBlackHole.y * positionToBlackHole.y;
			//double distToBlackHole = sqrt(distToBlackHoleSquared);

			double invDistToBlackHole = 1.0 / sqrt(distToBlackHoleSquared); // apparently faster

			// Gravity force
			doublevector normalizedDirection = {
			   .x = positionToBlackHole.x * invDistToBlackHole,
			   .y = positionToBlackHole.y * invDistToBlackHole };

			double accumulation = GRAVITY / distToBlackHoleSquared;

			// Delta time is used to make velocity same despite different FPS
			// Update velocity based on force
			tmpVelX[i] -= accumulation * normalizedDirection.x *
				DELTATIME_PER_PHYSICSUPDATESPERFRAME;
			tmpVelY[i] -= accumulation * normalizedDirection.y *
				DELTATIME_PER_PHYSICSUPDATESPERFRAME;

			// Update position based on velocity
			tmpPosX[i] +=
				tmpVelX[i] * DELTATIME_PER_PHYSICSUPDATESPERFRAME;
			tmpPosY[i] +=
				tmpVelY[i] * DELTATIME_PER_PHYSICSUPDATESPERFRAME;
		}
	}

	// double precision required for accumulation inside this routine,
	// but float storage is ok outside these loops.
	// copy back the float storage.
	int idx2;
#pragma omp parallel for
	for (idx2 = 0; idx2 < SATELLITE_COUNT; ++idx2) {
		satellites[idx2].position.x = (float)tmpPosX[idx2];
		satellites[idx2].position.y = (float)tmpPosY[idx2];
		satellites[idx2].velocity.x = (float)tmpVelX[idx2];
		satellites[idx2].velocity.y = (float)tmpVelY[idx2];
	}

}


// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.
void parallelGraphicsEngine() {
	cl_int status;

	//Write satellite data to GPU. Comment out if you don't need to update satellite data (if you are using the physics kernel)
#if USE_PHYSICS_KERNEL == 0
	status = clEnqueueWriteBuffer(commandQueue, satelliteBuffer, CL_TRUE, 0,
		sizeof(satellite) * SATELLITE_COUNT, satellites, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		printf("Error writing satellite data: %s\n", clErrorString(status));
		abort();
	}
#endif

	// Set kernel arguments
	status = clSetKernelArg(graphicsKernel, 0, sizeof(cl_mem), &pixelBuffer);
	status |= clSetKernelArg(graphicsKernel, 1, sizeof(cl_mem), &satelliteBuffer);
	status |= clSetKernelArg(graphicsKernel, 2, sizeof(int), &mousePosX);
	status |= clSetKernelArg(graphicsKernel, 3, sizeof(int), &mousePosY);

	if (status != CL_SUCCESS) {
		printf("Error setting kernel arguments: %s\n", clErrorString(status));
		abort();
	}


	// Execute graphicskernel
	status = clEnqueueNDRangeKernel(commandQueue, graphicsKernel, 2, NULL,
		globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (status != CL_SUCCESS) {
		printf("Error executing kernel: %s\n", clErrorString(status));
		abort();
	}

	// Read back results from the pixel buffer after kernel execution
	status = clEnqueueReadBuffer(commandQueue, pixelBuffer, CL_TRUE, 0,
		sizeof(color_u8) * WINDOW_SIZE, pixels, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		printf("Error reading results: %s\n", clErrorString(status));
		abort();
	}
}


// ## You may add your own destrcution routines here ##
void destroy() {
	clReleaseMemObject(pixelBuffer);
	clReleaseMemObject(satelliteBuffer);
	clReleaseKernel(graphicsKernel);
	clReleaseKernel(physicsKernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);
}


////////////////////////////////////////////////
// ¤¤ TO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)
SDL_Window* win;
SDL_Surface* surf;
// Is used to find out frame times
int totalTimeAcc, satelliteMovementAcc, pixelColoringAcc, frameCount;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Sequential rendering loop used for finding errors
void sequentialGraphicsEngine() {
	// Graphics pixel loop
	for (int i = 0; i < WINDOW_SIZE; ++i) {

		// Row wise ordering
		floatvector pixel = { .x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH };

		// Draw the black hole
		floatvector positionToBlackHole = { .x = pixel.x -
		   HORIZONTAL_CENTER, .y = pixel.y - VERTICAL_CENTER };
		float distToBlackHoleSquared =
			positionToBlackHole.x * positionToBlackHole.x +
			positionToBlackHole.y * positionToBlackHole.y;
		float distToBlackHole = sqrt(distToBlackHoleSquared);
		if (distToBlackHole < BLACK_HOLE_RADIUS) {
			correctPixels[i].red = 0;
			correctPixels[i].green = 0;
			correctPixels[i].blue = 0;
			continue; // Black hole drawing done
		}

		// This color is used for coloring the pixel
		color_f32 renderColor = { .red = 0.f, .green = 0.f, .blue = 0.f };

		// Find closest satellite
		float shortestDistance = INFINITY;

		float weights = 0.f;
		int hitsSatellite = 0;

		// First Graphics satellite loop: Find the closest satellite.
		for (int j = 0; j < SATELLITE_COUNT; ++j) {
			floatvector difference = { .x = pixel.x - satellites[j].position.x,
									  .y = pixel.y - satellites[j].position.y };
			float distance = sqrt(difference.x * difference.x +
				difference.y * difference.y);

			if (distance < SATELLITE_RADIUS) {
				renderColor.red = 1.0f;
				renderColor.green = 1.0f;
				renderColor.blue = 1.0f;
				hitsSatellite = 1;
				break;
			}
			else {
				float weight = 1.0f / (distance * distance * distance * distance);
				weights += weight;
				if (distance < shortestDistance) {
					shortestDistance = distance;
					renderColor = satellites[j].identifier;
				}
			}
		}

		// Second graphics loop: Calculate the color based on distance to every satellite.
		if (!hitsSatellite) {
			for (int j = 0; j < SATELLITE_COUNT; ++j) {
				floatvector difference = { .x = pixel.x - satellites[j].position.x,
										  .y = pixel.y - satellites[j].position.y };
				float dist2 = (difference.x * difference.x +
					difference.y * difference.y);
				float weight = 1.0f / (dist2 * dist2);

				renderColor.red += (satellites[j].identifier.red *
					weight / weights) * 3.0f;

				renderColor.green += (satellites[j].identifier.green *
					weight / weights) * 3.0f;

				renderColor.blue += (satellites[j].identifier.blue *
					weight / weights) * 3.0f;
			}
		}
		correctPixels[i].red = (uint8_t)(renderColor.red * 255.0f);
		correctPixels[i].green = (uint8_t)(renderColor.green * 255.0f);
		correctPixels[i].blue = (uint8_t)(renderColor.blue * 255.0f);
	}
}

void sequentialPhysicsEngine(satellite* s) {

	// double precision required for accumulation inside this routine,
	// but float storage is ok outside these loops.
	doublevector tmpPosition[SATELLITE_COUNT];
	doublevector tmpVelocity[SATELLITE_COUNT];

	for (int i = 0; i < SATELLITE_COUNT; ++i) {
		tmpPosition[i].x = s[i].position.x;
		tmpPosition[i].y = s[i].position.y;
		tmpVelocity[i].x = s[i].velocity.x;
		tmpVelocity[i].y = s[i].velocity.y;
	}

	// Physics iteration loop
	for (int physicsUpdateIndex = 0;
		physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
		++physicsUpdateIndex) {

		// Physics satellite loop
		for (int i = 0; i < SATELLITE_COUNT; ++i) {

			// Distance to the blackhole
			// (bit ugly code because C-struct cannot have member functions)
			doublevector positionToBlackHole = { .x = tmpPosition[i].x -
			   HORIZONTAL_CENTER, .y = tmpPosition[i].y - VERTICAL_CENTER };
			double distToBlackHoleSquared =
				positionToBlackHole.x * positionToBlackHole.x +
				positionToBlackHole.y * positionToBlackHole.y;
			double distToBlackHole = sqrt(distToBlackHoleSquared);

			// Gravity force
			doublevector normalizedDirection = {
			   .x = positionToBlackHole.x / distToBlackHole,
			   .y = positionToBlackHole.y / distToBlackHole };
			double accumulation = GRAVITY / distToBlackHoleSquared;

			// Delta time is used to make velocity same despite different FPS
			// Update velocity based on force
			tmpVelocity[i].x -= accumulation * normalizedDirection.x *
				DELTATIME / PHYSICSUPDATESPERFRAME;
			tmpVelocity[i].y -= accumulation * normalizedDirection.y *
				DELTATIME / PHYSICSUPDATESPERFRAME;

			// Update position based on velocity
			tmpPosition[i].x +=
				tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
			tmpPosition[i].y +=
				tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
		}
	}

	// double precision required for accumulation inside this routine,
	// but float storage is ok outside these loops.
	// copy back the float storage.
	for (int i = 0; i < SATELLITE_COUNT; ++i) {
		s[i].position.x = tmpPosition[i].x;
		s[i].position.y = tmpPosition[i].y;
		s[i].velocity.x = tmpVelocity[i].x;
		s[i].velocity.y = tmpVelocity[i].y;
	}
}

// Just some value that barely passes for OpenCL example program
#define ALLOWED_ERROR 10
#define ALLOWED_NUMBER_OF_ERRORS 10
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void errorCheck() {
	int countErrors = 0;
	for (unsigned int i = 0; i < WINDOW_SIZE; ++i) {
		if (abs(correctPixels[i].red - pixels[i].red) > ALLOWED_ERROR ||
			abs(correctPixels[i].green - pixels[i].green) > ALLOWED_ERROR ||
			abs(correctPixels[i].blue - pixels[i].blue) > ALLOWED_ERROR) {
			printf("Pixel x=%d y=%d value: %d, %d, %d. Should have been: %d, %d, %d\n",
				i % WINDOW_WIDTH, i / WINDOW_WIDTH,
				pixels[i].red, pixels[i].green, pixels[i].blue,
				correctPixels[i].red, correctPixels[i].green, correctPixels[i].blue);
			countErrors++;
			if (countErrors > ALLOWED_NUMBER_OF_ERRORS) {
				printf("Too many errors (%d) in frame %d, Press enter to continue.\n", countErrors, frameNumber);
				getchar();
				return;
			}
		}
	}
	printf("Error check passed with acceptable number of wrong pixels: %d\n", countErrors);
}


// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void compute(void) {
	int timeSinceStart = SDL_GetTicks();

	// Error check during first frames
	if (frameNumber < 2) {
		memcpy(backupSatelites, satellites, sizeof(satellite) * SATELLITE_COUNT);
		sequentialPhysicsEngine(backupSatelites);
		mousePosX = HORIZONTAL_CENTER;
		mousePosY = VERTICAL_CENTER;
	}
	else {
		SDL_GetMouseState(&mousePosX, &mousePosY);
		if ((mousePosX == 0) && (mousePosY == 0)) {
			mousePosX = HORIZONTAL_CENTER;
			mousePosY = VERTICAL_CENTER;
		}
	}
	parallelPhysicsEngine();
	if (frameNumber < 2) {
		for (int i = 0; i < SATELLITE_COUNT; i++) {
			if (memcmp(&satellites[i], &backupSatelites[i], sizeof(satellite))) {
				printf("Incorrect satellite data of satellite: %d\n", i);
				getchar();
			}
		}
	}

	int satelliteMovementMoment = SDL_GetTicks();
	int satelliteMovementTime = satelliteMovementMoment - timeSinceStart;

	// Decides the colors for the pixels
	parallelGraphicsEngine();

	int pixelColoringMoment = SDL_GetTicks();
	int pixelColoringTime = pixelColoringMoment - satelliteMovementMoment;

	int finishTime = SDL_GetTicks();
	// Sequential code is used to check possible errors in the parallel version
	if (frameNumber < 2) {
		sequentialGraphicsEngine();
		errorCheck();
	}
	else if (frameNumber == 2) {
		previousFinishTime = finishTime;
		printf("Time spent on moving satellites + Time spent on space coloring : Total time in milliseconds between frames (might not equal the sum of the left-hand expression)\n");
	}
	else if (frameNumber > 2) {
		// Print timings
		int totalTime = finishTime - previousFinishTime;
		previousFinishTime = finishTime;

		printf("Latency of this frame %i + %i : %ims \n",
			satelliteMovementTime, pixelColoringTime, totalTime);

		frameCount++;
		totalTimeAcc += totalTime;
		satelliteMovementAcc += satelliteMovementTime;
		pixelColoringAcc += pixelColoringTime;
		printf("Averaged over all frames: %i + %i : %ims.\n",
			satelliteMovementAcc / frameCount, pixelColoringAcc / frameCount, totalTimeAcc / frameCount);

	}
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Probably not the best random number generator
float randomNumber(float min, float max) {
	return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed) {

	if (seed != 0) {
		srand(seed);
	}

	// Init pixel buffer which is rendered to the widow
	pixels = (color_u8*)malloc(sizeof(color_u8) * WINDOW_SIZE);

	// Init pixel buffer which is used for error checking
	correctPixels = (color_u8*)malloc(sizeof(color_u8) * WINDOW_SIZE);

	backupSatelites = (satellite*)malloc(sizeof(satellite) * SATELLITE_COUNT);


	// Init satellites buffer which are moving in the space
	satellites = (satellite*)malloc(sizeof(satellite) * SATELLITE_COUNT);

	// Create random satellites
	for (int i = 0; i < SATELLITE_COUNT; ++i) {

		// Random reddish color
		color_f32 id = { .red = randomNumber(0.f, 0.15f) + 0.1f,
					.green = randomNumber(0.f, 0.14f) + 0.0f,
					.blue = randomNumber(0.f, 0.16f) + 0.0f };

		// Random position with margins to borders
		floatvector initialPosition = { .x = HORIZONTAL_CENTER - randomNumber(50, 320),
								.y = VERTICAL_CENTER - randomNumber(50, 320) };
		initialPosition.x = (i / 2 % 2 == 0) ?
			initialPosition.x : WINDOW_WIDTH - initialPosition.x;
		initialPosition.y = (i < SATELLITE_COUNT / 2) ?
			initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

		// Randomize velocity tangential to the balck hole
		floatvector positionToBlackHole = { .x = initialPosition.x - HORIZONTAL_CENTER,
									  .y = initialPosition.y - VERTICAL_CENTER };
		float distance = (0.06 + randomNumber(-0.01f, 0.01f)) /
			sqrt(positionToBlackHole.x * positionToBlackHole.x +
				positionToBlackHole.y * positionToBlackHole.y);
		floatvector initialVelocity = { .x = distance * -positionToBlackHole.y,
								  .y = distance * positionToBlackHole.x };

		// Every other orbits clockwise
		if (i % 2 == 0) {
			initialVelocity.x = -initialVelocity.x;
			initialVelocity.y = -initialVelocity.y;
		}

		satellite tmpSatelite = { .identifier = id, .position = initialPosition,
								.velocity = initialVelocity };
		satellites[i] = tmpSatelite;
	}
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void fixedDestroy(void) {
	destroy();

	free(pixels);
	free(correctPixels);
	free(satellites);

	if (seed != 0) {
		printf("Used seed: %i\n", seed);
	}
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Renders pixels-buffer to the window 
void render(void) {
	SDL_LockSurface(surf);
	memcpy(surf->pixels, pixels, WINDOW_WIDTH * WINDOW_HEIGHT * 4);
	SDL_UnlockSurface(surf);

	SDL_UpdateWindowSurface(win);
	frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits render window and starts mainloop
int main(int argc, char** argv) {

	if (argc > 1) {
		seed = atoi(argv[1]);
		printf("Using seed: %i\n", seed);
	}

	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);
	win = SDL_CreateWindow(
		"Satellites",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		WINDOW_WIDTH, WINDOW_HEIGHT,
		0
	);
	surf = SDL_GetWindowSurface(win);

	fixedInit(seed);
	init();

	SDL_Event event;
	int running = 1;
	while (running) {
		while (SDL_PollEvent(&event)) switch (event.type) {
		case SDL_QUIT:
			printf("Quit called\n");
			running = 0;
			break;
		}
		compute();
		render();
	}
	SDL_Quit();
	fixedDestroy();
	return 0;
}
