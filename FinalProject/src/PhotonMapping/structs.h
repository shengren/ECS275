#ifndef STRUCTS_H_
#define STRUCTS_H_

#include <optixu/optixu_math_namespace.h>

#define HIT (1 << 5)
#define HIT_LIGHT (1 << 6)
#define HIT_BACKGROUND (1 << 7)
#define EXCEPTION (1 << 8)

struct RTViewingRayPayload {
  optix::float3 attenuation;
  optix::float3 radiance;
  optix::uint depth;
  optix::uint seed;
  bool inside;
};

struct RTShadowRayPayload {
  bool blocked;
};

struct PhotonRecord {
  optix::float3 power;
  optix::float3 position;
  optix::float3 normal;
  optix::float3 incoming;
  int axis;  // used in kdtree
};

struct PTPhotonRayPayload {
  optix::float3 power;
  optix::uint index;
  optix::uint num_deposits;
  optix::uint depth;
  optix::uint seed;
  bool inside;
};

struct ParallelogramLight {
  optix::float3 corner;
  optix::float3 v1;
  optix::float3 v2;
  optix::float3 normal;
  float area;
  optix::float3 power;  // for photon tracing
  optix::uint sqrt_num_samples;
  optix::float3 emitted;  // for direct illumination, to-do: Le has different values
};

struct Neighbor {
  float dist2;
  int idx;
};

/*
struct Sphere {  // for caustic photon tracing
  optix::float3 center;
  float radius;
};
*/

#endif  // STRUCTS_H_
