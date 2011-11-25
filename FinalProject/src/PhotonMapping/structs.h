#ifndef STRUCTS_H_
#define STRUCTS_H_

#include <optixu/optixu_math_namespace.h>

// from progressivePhotonMap/ppm.h
// begin
#define PPM_X (1 << 0)
#define PPM_Y (1 << 1)
#define PPM_Z (1 << 2)
#define PPM_LEAF (1 << 3)
#define PPM_NULL (1 << 4)
// end

#define IN_SHADOW (1 << 5)
#define EXCEPTION (1 << 6)
#define HIT (1 << 7)

struct HitRecord {
  optix::uint flags;
  optix::float3 attenuated_Kd;
  optix::float3 position;
  optix::float3 normal;
  optix::float3 outgoing;
};

struct RTViewingRayPayload {
  optix::float3 attenuation;
  optix::uint depth;
};

struct PhotonRecord {
  optix::float3 position;
  optix::float3 normal;
  optix::float3 incoming;
  optix::float3 power;

  optix::uint axis;
};

struct PTPhotonRayPayload {
  optix::float3 power;
  optix::uint index;
  optix::uint num_deposits;
  optix::uint depth;
  optix::uint seed;
};

struct GTShadowRayPayload {
  bool blocked;
};

struct ParallelogramLight {
  optix::float3 corner;
  optix::float3 v1;
  optix::float3 v2;
  optix::float3 normal;
  optix::float3 power;
  float area;
  optix::uint sqrt_num_samples;
  optix::float3 emitted;
};

#endif  // STRUCTS_H_
