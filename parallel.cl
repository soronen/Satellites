typedef uchar uint8_t;

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
  uint8 blue;
  uint8 green;
  uint8 red;
  uint8 reserved;
} color_u8;

// Stores the satellite data, which fly around black hole in the space
typedef struct {
  color_f32 identifier;
  floatvector position;
  floatvector velocity;
} satellite;

__kernel void
graphicsKernel(__global color_u8 *pixels,            // Output pixel buffer
               __global const satellite *satellites, // Input satellites array
               const int SATELLITE_COUNT,            // Number of satellites
               const int WINDOW_WIDTH,               // Window width
               const float BLACK_HOLE_RADIUS,        // Black hole radius
               const float SATELLITE_RADIUS,         // Satellite radius
               const int mousePosX,                  // Mouse X position
               const int mousePosY                   // Mouse Y position
) {
  // int tmpMousePosX = mousePosX;
  // int tmpMousePosY = mousePosY;

  const float BLACK_HOLE_RADIUS_SQUARED = BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS;
  const float SATELLITE_RADIUS_SQUARED = SATELLITE_RADIUS * SATELLITE_RADIUS;

  int i = get_global_id(0);

  // Row wise ordering
  floatvector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

  // Draw the black hole
  floatvector positionToBlackHole = {.x = pixel.x - mousePosX,
                                     .y = pixel.y - mousePosY};
  float distToBlackHoleSquared = positionToBlackHole.x * positionToBlackHole.x +
                                 positionToBlackHole.y * positionToBlackHole.y;
  // float distToBlackHole = sqrt(distToBlackHoleSquared);
  if (distToBlackHoleSquared < BLACK_HOLE_RADIUS_SQUARED) {
    pixels[i].red = 0;
    pixels[i].green = 0;
    pixels[i].blue = 0;
    return; // Black hole drawing done
  }

  // This color is used for coloring the pixel
  color_f32 renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

  // Find closest satellite
  float shortestDistanceSquared = INFINITY;

  float weights = 0.f;
  int hitsSatellite = 0;

  // First Graphics satellite loop: Find the closest satellite.
  for (int j = 0; j < SATELLITE_COUNT; ++j) {
    floatvector difference = {.x = pixel.x - satellites[j].position.x,
                              .y = pixel.y - satellites[j].position.y};
    float distanceSquared =
        (difference.x * difference.x + difference.y * difference.y);

    if (distanceSquared < SATELLITE_RADIUS_SQUARED) {
      renderColor.red = 1.0f;
      renderColor.green = 1.0f;
      renderColor.blue = 1.0f;
      hitsSatellite = 1;
      break;
    } else {
      float weight = 1.0f / (distanceSquared * distanceSquared);
      weights += weight;
      if (distanceSquared < shortestDistanceSquared) {
        shortestDistanceSquared = distanceSquared;
        renderColor = satellites[j].identifier;
      }
    }
  }

  // Second graphics loop: Calculate the color based on distance to every
  // satellite.
  if (!hitsSatellite) {
    for (int k = 0; k < SATELLITE_COUNT; ++k) {
      floatvector difference = {.x = pixel.x - satellites[k].position.x,
                                .y = pixel.y - satellites[k].position.y};
      float dist2 = (difference.x * difference.x + difference.y * difference.y);
      float weight = 1.0f / (dist2 * dist2);

      renderColor.red +=
          (satellites[k].identifier.red * weight / weights) * 3.0f;

      renderColor.green +=
          (satellites[k].identifier.green * weight / weights) * 3.0f;

      renderColor.blue +=
          (satellites[k].identifier.blue * weight / weights) * 3.0f;
    }
  }
  pixels[i].red = (uint8_t)(renderColor.red * 255.0f);
  pixels[i].green = (uint8_t)(renderColor.green * 255.0f);
  pixels[i].blue = (uint8_t)(renderColor.blue * 255.0f);
}