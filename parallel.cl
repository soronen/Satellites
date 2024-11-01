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
	__global color_u8* pixels,  // in the buffer pixels are uchar values, but in kernel they are treated as floats for calculations
    __global satellite* satellites,
    int mousePosX,
    int mousePosY,
    float blackHoleRadiusSquared,
    float satelliteRadiusSquared,
    int satelliteCount,
	const int windowWidth,
	const int windowHeight
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

