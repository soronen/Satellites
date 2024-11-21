#include "constants.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable // this is ´probably enabled by default

// general and graphics vars
const int windowWidth = WINDOW_WIDTH;
const int windowHeight = WINDOW_HEIGHT;
const int satelliteCount = SATELLITE_COUNT;

float blackHoleRadiusSquared = BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS;
float satelliteRadiusSquared = SATELLITE_RADIUS * SATELLITE_RADIUS;

typedef struct {
	float blue;
	float green;
	float red;
} color_f32;

typedef struct {
	uchar blue;
	uchar green;
	uchar red;
	uchar reserved;
} color_u8;

typedef struct {
	float x;
	float y;
} floatvector;

typedef struct {
	color_f32 identifier;
	floatvector position;
	floatvector velocity;
} satellite;


__kernel void graphicsKernel(
	__global color_u8* pixels,  // in the buffer pixels are uchar values, but in kernel they are treated as floats
	__global satellite* satellites,
	const int mousePosX,
	const int mousePosY
) {
	// which pixel are we drawing to
	int x = get_global_id(0);
	int y = get_global_id(1);
	int i = y * windowWidth + x;
	floatvector pixel = (floatvector){ x, y };

	// return in case pixel is outside the window bounds
	if (x >= windowWidth || y >= windowHeight) return;


	// Are inside the black hole? If so, the pixel is black
	floatvector positionToBlackHole = (floatvector){
		pixel.x - mousePosX,
		pixel.y - mousePosY
	};
	float distToBlackHoleSquared =
		positionToBlackHole.x * positionToBlackHole.x +
		positionToBlackHole.y * positionToBlackHole.y;

	if (distToBlackHoleSquared < blackHoleRadiusSquared) {
		pixels[i] = (color_u8){ 0, 0, 0, 0 };
		return;
	}

	// First pass - collect weights and check hits
	float weights = 0.0f;
	float shortestDistanceSquared = INFINITY;
	color_f32 renderColor = (color_f32){ 0.0f, 0.0f, 0.0f };
	int hitsSatellite = 0;

	for (int j = 0; j < satelliteCount; j++) {
		floatvector difference = (floatvector){
			pixel.x - satellites[j].position.x,
			pixel.y - satellites[j].position.y
		};
		float distanceSquared = difference.x * difference.x + difference.y * difference.y;

		if (distanceSquared < satelliteRadiusSquared) {
			// if pixel is inside a satellite, color is white
			pixels[i] = (color_u8){ 255, 255, 255, 0 };
			hitsSatellite = 1;
			break;
		}
		else {
			float weight = 1.0f / (distanceSquared * distanceSquared);
			weights += weight;
			if (distanceSquared < shortestDistanceSquared) {
				shortestDistanceSquared = distanceSquared;
				renderColor = satellites[j].identifier;
			}
		}
	}

	if (!hitsSatellite) {
		// Second pass - calculate final color
		for (int j = 0; j < satelliteCount; j++) {
			floatvector difference = (floatvector){
				pixel.x - satellites[j].position.x,
				pixel.y - satellites[j].position.y
			};
			float distanceSquared = difference.x * difference.x + difference.y * difference.y;
			float weight = 1.0f / (distanceSquared * distanceSquared);

			renderColor.red += (satellites[j].identifier.red * weight / weights) * 3.0f;
			renderColor.green += (satellites[j].identifier.green * weight / weights) * 3.0f;
			renderColor.blue += (satellites[j].identifier.blue * weight / weights) * 3.0f;
		}

		pixels[i] = (color_u8){
			(uchar)(renderColor.blue * 255.0f),
			(uchar)(renderColor.green * 255.0f),
			(uchar)(renderColor.red * 255.0f),
			0
		};
	}
}


// physics vars
typedef struct {
	double x;
	double y;
} doublevector;

// freeze if not cast as double
const double DELTATIME_PER_PHYSICSUPDATESPERFRAME = (double)DELTATIME / PHYSICSUPDATESPERFRAME;

__kernel void physicsKernel(
	__global satellite* satellites,
	const int mousePosX,
	const int mousePosY
) {
	const int satelliteIndex = get_global_id(0);
	if (satelliteIndex >= SATELLITE_COUNT) return;


	double posX = satellites[satelliteIndex].position.x;
	double posY = satellites[satelliteIndex].position.y;
	double velX = satellites[satelliteIndex].velocity.x;
	double velY = satellites[satelliteIndex].velocity.y;

	// Physics iteration loop
	for (int physicsUpdateIndex = 0; physicsUpdateIndex < PHYSICSUPDATESPERFRAME; ++physicsUpdateIndex) {

		// Distance to the blackhole (bit ugly code because C-struct cannot have member functions)
		doublevector positionToBlackHole = {
			.x = posX - mousePosX,
			.y = posY - mousePosY };

		double distToBlackHoleSquared =
			positionToBlackHole.x * positionToBlackHole.x +
			positionToBlackHole.y * positionToBlackHole.y;

		double invDistToBlackHole = 1.0 / sqrt(distToBlackHoleSquared); // apparently faster than square root by itself

		// Gravity force
		doublevector normalizedDirection = {
		   .x = positionToBlackHole.x * invDistToBlackHole,
		   .y = positionToBlackHole.y * invDistToBlackHole };

		double accumulation = GRAVITY / distToBlackHoleSquared;

		// Delta time is used to make velocity same despite different FPS
		// Update velocity based on force
		velX -= accumulation * normalizedDirection.x * DELTATIME_PER_PHYSICSUPDATESPERFRAME;
		velY -= accumulation * normalizedDirection.y * DELTATIME_PER_PHYSICSUPDATESPERFRAME;

		// Update position based on velocity
		posX += velX * DELTATIME_PER_PHYSICSUPDATESPERFRAME;
		posY += velY * DELTATIME_PER_PHYSICSUPDATESPERFRAME;

	}

	// double precision required for accumulation inside this routine,
	// but float storage is ok outside these loops.
	// copy back the float storage.
	satellites[satelliteIndex].position.x = (float)posX;
	satellites[satelliteIndex].position.y = (float)posY;
	satellites[satelliteIndex].velocity.x = (float)velX;
	satellites[satelliteIndex].velocity.y = (float)velY;
}