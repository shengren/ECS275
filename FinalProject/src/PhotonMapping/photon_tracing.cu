#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"  // tea<>, rnd
#include "structs.h"
#include "inlines.h"

using namespace optix;

// modified from progressivePhotonMap/path_tracer.h
__device__ __inline__ optix::float3 sampleUnitHemisphereCosine(
    optix::uint& seed,
    const optix::float3& normal) {
  optix::float3 U, V, W;
  createONB(normal, U, V, W);

  float phi = 2.0f * M_PIf * rnd(seed);
  float r = sqrt(rnd(seed));
  float x = r * cos(phi);
  float y = r * sin(phi);
  float z = 1.0f - x * x - y * y;
  z = z > 0.0f ? sqrt(z) : 0.0f;

  return x * U + y * V + z * W;
}

__device__ __inline__ void generatePhoton(const ParallelogramLight& light,
                                          uint& seed,
                                          float3& sample_position,
                                          float3& sample_direction,
                                          float3& sample_power) {
  sample_position = light.corner + light.v1 * rnd(seed) + light.v2 * rnd(seed);

  sample_direction = sampleUnitHemisphereCosine(seed, light.normal);

  sample_power = light.power;
}

// variables used in multiple programs
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint, pt_photon_ray_type, , );
rtDeclareVariable(uint, max_num_deposits, , );
rtBuffer<PhotonRecord> photon_record_buffer;  // 1D by default

// photon tracing, ray generation
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(uint, frame_number, , );
rtBuffer<ParallelogramLight> lights;  // to-do: only have parallelogram lights

RT_PROGRAM void pt_ray_generation() {
  uint index = launch_index.y * launch_dim.x + launch_index.x;  // to-do: Will 1D launch be enough?
  uint seed = tea<16>(index, frame_number);
  float3 sample_position;
  float3 sample_direction;
  float3 sample_power;

  // to-do: better way?
  for (int i = 0; i < max_num_deposits; ++i)
    photon_record_buffer[index + i].power = make_float3(0.0f);

  // to-do: one parallelogram light only
  generatePhoton(lights[0], seed, sample_position, sample_direction, sample_power);

  Ray ray(sample_position,
          sample_direction,
          pt_photon_ray_type,
          1e-10f);

  PTPhotonRayPayload payload;
  payload.power = sample_power;
  payload.index = index;
  payload.seed = seed;
  payload.num_deposits = 0;
  payload.depth = 0;

  rtTrace(top_object, ray, payload);
}

// photon tracing, exception
RT_PROGRAM void pt_exception() {
  rtPrintExceptionDetails();  // to-do:
}

// photon tracing, photon ray, closest hit, default material
rtDeclareVariable(Ray, pt_photon_ray, rtCurrentRay, );
rtDeclareVariable(float, hit_t, rtIntersectionDistance, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, Le, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(PTPhotonRayPayload, pt_photon_ray_payload, rtPayload, );
rtDeclareVariable(uint, max_depth, , );

RT_PROGRAM void pt_photon_ray_closest_hit() {
  if (fmaxf(Le) > 0.0f) {  // light source
    return;
  }

  float3 hit_point = pt_photon_ray.origin + hit_t * pt_photon_ray.direction;
  float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal = faceforward(world_shading_normal, -pt_photon_ray.direction, world_geometric_normal);

  // record when hit diffuse surfaces and bounced at least once (avoid doubling direct illumination)
  // to-do: test-only
  //if (fmaxf(Kd) > 0.0f && pt_photon_ray_payload.depth > 0) {
  if (fmaxf(Kd) > 0.0f) {
    PhotonRecord& pr = photon_record_buffer[pt_photon_ray_payload.index +
                                            pt_photon_ray_payload.num_deposits];
    pr.position = hit_point;
    pr.normal = ffnormal;
    pr.incoming = -pt_photon_ray.direction;  // hit_point is the origin
    pr.power = pt_photon_ray_payload.power;

    pt_photon_ray_payload.num_deposits++;
  }

  if (pt_photon_ray_payload.num_deposits >= max_num_deposits ||
      pt_photon_ray_payload.depth >= max_depth) {
    return;
  }

  pt_photon_ray_payload.depth++;
  float3 next_direction;
  if (fmaxf(Kd) > 0.0f) {  // diffuse
    pt_photon_ray_payload.power *= Kd;  // to-do: BRDF shouldn't be a constant
    next_direction = sampleUnitHemisphereCosine(pt_photon_ray_payload.seed,
                                                ffnormal);
  } else {  // specular
    pt_photon_ray_payload.power *= Ks;  // to-do: BRDF shouldn't be a constant
    next_direction = reflect(pt_photon_ray.direction, ffnormal);  // inversed incoming
  }

  Ray ray(hit_point,
          next_direction,
          pt_photon_ray_type,
          1e-10f);
  rtTrace(top_object, ray, pt_photon_ray_payload);
}

// photon tracing, photon, miss, default material
// do nothing
