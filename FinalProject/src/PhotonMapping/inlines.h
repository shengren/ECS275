#ifndef INLINES_H_
#define INLINES_H_

#include <optixu/optixu_math_namespace.h>

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
  if ( fabsf( U.x) < 0.001f && fabsf( U.y ) < 0.001f && fabsf( U.z ) < 0.001f  )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( W, U );
}

__device__ __inline__ float getDiffuseBRDF() {
  return 0.9f / M_PI;  // to-do: Kd is defined at where?
}

__device__ __inline__ float getGeometry(optix::float3 ns,
                                        optix::float3 nl,
                                        optix::float3 dir,
                                        float dist) {
  return optix::dot(ns, dir) * optix::dot(nl, -dir) / (dist * dist);
}

// modified from progressivePhotonMap/ppm.cpp
// begin

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
// end

#endif  // INLINES_H_
