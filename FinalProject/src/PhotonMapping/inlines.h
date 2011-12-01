#ifndef INLINES_H_
#define INLINES_H_

#include <cmath>  // to-do: there are sinf, cosf, sqrtf, etc. may be faster

#include <optixu/optixu_math_namespace.h>  // to-do: this header has many utilities!

#include "random.h"

__device__ __inline__ optix::float3 getSpecularBRDF(optix::float3 in,
                                                    optix::float3 n,
                                                    optix::float3 out,
                                                    optix::float3 rho_s,
                                                    float p) {
  optix::float3 s = optix::reflect(-in, n);  // inversed incoming
  float cos = optix::dot(out, s);
  cos = (cos > 1.0f) ? 1.0f : cos;
  return rho_s * (p + 2) * pow(cos, p) / (2.0f * M_PI);
}

__device__ __inline__ optix::float3 getDiffuseBRDF(optix::float3 rho_d) {
  return rho_d / M_PI;
}

// from progressivePhotonMap/path_tracer.h
// Create ONB from normal. Resulting W is parallel to normal
__device__ __inline__ void createONB( const optix::float3& n,
                                      optix::float3& U,
                                      optix::float3& V,
                                      optix::float3& W )
{
  using namespace optix;

  W = normalize( n );
  U = cross( W, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( fabsf( U.x ) < 0.001f && fabsf( U.y ) < 0.001f && fabsf( U.z ) < 0.001f )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( W, U );
}

// modified from progressivePhotonMap/path_tracer.h
__device__ __inline__ optix::float3 sampleUnitHemisphereCosine(
    optix::uint& seed,
    const optix::float3& normal) {
  optix::float3 U, V, W;
  createONB(normal, U, V, W);

  float phi = 2.0f * M_PI * rnd(seed);
  float r = sqrt(rnd(seed));
  float x = r * cos(phi);
  float y = r * sin(phi);
  float z = 1.0f - x * x - y * y;
  z = z > 0.0f ? sqrt(z) : 0.0f;

  return x * U + y * V + z * W;
}

__device__ __inline__ void generatePhoton(const ParallelogramLight& light,
                                          optix::uint& seed,
                                          optix::float3& sample_position,
                                          optix::float3& sample_direction,
                                          optix::float3& sample_power) {
  sample_position = light.corner + light.v1 * rnd(seed) + light.v2 * rnd(seed);

  sample_direction = sampleUnitHemisphereCosine(seed, light.normal);

  sample_power = light.power;
}

__device__ __inline__ void generateCausticPhoton(const Sphere& sphere,
                                                 const ParallelogramLight& light,
                                                 optix::uint& seed,
                                                 optix::float3& sample_position,
                                                 optix::float3& sample_direction,
                                                 optix::float3& sample_power) {
  sample_position = light.corner + light.v1 * rnd(seed) + light.v2 * rnd(seed);

  sample_power = light.power;

  // uniformly sampling on the caustic sphere
  float phi = 2.0f * M_PI * rnd(seed);
  float cos_theta = 1.0f - 2.0f * rnd(seed);
  float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
  float3 sdir = make_float3(cos(phi) * sin_theta,
                            sin(phi) * sin_theta,
                            cos_theta);
  float3 sp = sphere.center + sdir * sphere.radius;

  sample_direction = optix::normalize(sp - sample_position);
}

__device__ __inline__ float getGeometry(const optix::float3 ns,
                                        const optix::float3 nl,
                                        const optix::float3 dir,
                                        const float dist) {
  return optix::dot(ns, dir) * optix::dot(nl, -dir) / (dist * dist);
}

// modified from progressivePhotonMap/ppm.cpp
// Finds the smallest power of 2 greater or equal to x.
inline unsigned int pow2roundup(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x+1;
}

#endif  // INLINES_H_
