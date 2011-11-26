#ifndef STRUCTS_H_
#define STRUCTS_H_

#include <optixu/optixu_math_namespace.h>

#define EXCEPTION (1 << 5)
#define HIT_LIGHT (1 << 6)
#define HIT (1 << 7)
#define HIT_BACKGROUND (1 << 8)
//#define IN_SHADOW

struct HitRecord {
  optix::uint flags;
  optix::float3 attenuation;
  optix::float3 position;
  optix::float3 normal;
  optix::float3 outgoing;
  optix::float3 Rho_d;
};

struct RTViewingRayPayload {
  optix::float3 attenuation;
  optix::uint depth;
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
};

struct GTShadowRayPayload {
  bool blocked;
};

struct ParallelogramLight {
  optix::float3 corner;
  optix::float3 v1;
  optix::float3 v2;
  optix::float3 normal;
  float area;
  optix::float3 power;
  optix::uint sqrt_num_samples;
  optix::float3 emitted;
};

#endif  // STRUCTS_H_
