#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"  // tea<>, rnd
#include "structs.h"
#include "inlines.h"

using namespace optix;

// variables used in multiple programs
rtBuffer<PhotonRecord, 1> photon_record_buffer;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint, pt_photon_ray_type, , );
rtDeclareVariable(uint, max_num_deposits, , );

// photon tracing, ray generation
rtBuffer<ParallelogramLight> lights;  // to-do: only have parallelogram lights
//rtBuffer<Sphere> caustics;  // to-do: only caustic spheres
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(uint, frame_number, , );
rtDeclareVariable(float, total_emitted, , );

// since we reuse the photon_record_buffer, make sure it is clean before generating photons
// now it is cleaned before executing this program.
RT_PROGRAM void pt_ray_generation() {
  uint index = launch_index.y * launch_dim.x + launch_index.x;  // to-do: is 1D launch enough?
  uint seed = tea<16>(index, frame_number);
  float3 sample_position;
  float3 sample_direction;
  float3 sample_power;

  generatePhoton(lights[0], seed,
      sample_position, sample_direction, sample_power);  // to-do: only one parallelogram light now

  Ray ray(sample_position,
          sample_direction,
          pt_photon_ray_type,
          1e-2f);

  PTPhotonRayPayload payload;
  payload.power = sample_power / total_emitted;  // to-do: real power per photon?
  payload.index = index;
  payload.num_deposits = 0;
  payload.depth = 1;
  payload.seed = seed;
  payload.inside = false;

  rtTrace(top_object, ray, payload);
}

// photon tracing, exception
RT_PROGRAM void pt_exception() {
  rtPrintExceptionDetails();  // to-do: for debugging
}

// photon tracing, photon ray, closest hit, default material
rtDeclareVariable(Ray, pt_photon_ray, rtCurrentRay, );
rtDeclareVariable(float, hit_t, rtIntersectionDistance, );
rtDeclareVariable(PTPhotonRayPayload, pt_photon_ray_payload, rtPayload, );
rtDeclareVariable(uint, min_depth, , );  // started from 1, record bounces in [min_depth, max_depth]
rtDeclareVariable(uint, max_depth, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, Le, , );
rtDeclareVariable(float3, Rho_d, , );
rtDeclareVariable(float3, Rho_s, , );
rtDeclareVariable(float, shininess, , );  // unused now
rtDeclareVariable(float, index_of_refraction, , );  // non-zero indiates a refraction surface, Rho_s is needed as well

RT_PROGRAM void pt_photon_ray_closest_hit() {
  if (pt_photon_ray_payload.num_deposits >= max_num_deposits ||
      pt_photon_ray_payload.depth > max_depth) {
    return;
  }

  if (fmaxf(Le) > 0.0f) {  // light source
    return;
  }

  float3 hit_point = pt_photon_ray.origin + hit_t * pt_photon_ray.direction;
  float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal = faceforward(world_shading_normal, -pt_photon_ray.direction, world_geometric_normal);

  // record when hit diffuse surfaces and bounced at least once (avoid doubling direct illumination)
  // min_depth = 1, record from the first bounce for test, = 2, regular case
  if (fmaxf(Rho_d) > 0.0f && pt_photon_ray_payload.depth >= min_depth) {
    PhotonRecord& pr = photon_record_buffer[pt_photon_ray_payload.index +
                                            pt_photon_ray_payload.num_deposits];
    pr.power = pt_photon_ray_payload.power;
    pr.position = hit_point;
    pr.normal = ffnormal;
    pr.incoming = -pt_photon_ray.direction;  // hit_point is the origin
    // pr.axis is used in kdtree

    pt_photon_ray_payload.num_deposits++;
  }

  pt_photon_ray_payload.depth++;

  if (fmaxf(Rho_d) > 0.0f) {  // diffuse
    float3 next_direction = sampleUnitHemisphereCosine(pt_photon_ray_payload.seed,
                                                       ffnormal);
    pt_photon_ray_payload.power *= getDiffuseBRDF(Rho_d);
    Ray ray(hit_point,
            next_direction,
            pt_photon_ray_type,
            1e-2f);
    rtTrace(top_object, ray, pt_photon_ray_payload);
    return;
  }

  // specular

  float3 reflection_direction = reflect(pt_photon_ray.direction, ffnormal);  // inversed incoming
  float reflection_ratio = 1.0f;
  float3 refraction_direction;
  float refraction_ratio = 0.0f;
  bool has_refraction = false;
  PTPhotonRayPayload refraction_payload = pt_photon_ray_payload;
  if (index_of_refraction > 0.0f) {
    float iof = (pt_photon_ray_payload.inside) ?
                (1.0f / index_of_refraction) : index_of_refraction;
    refract(refraction_direction, pt_photon_ray.direction, ffnormal, iof);
    float cos_i = dot(-pt_photon_ray.direction, ffnormal);
    float cos2_t = 1.0f - ((1.0f - (cos_i * cos_i)) / (iof * iof));
    if (cos2_t >= 0) {
      has_refraction = true;
      float a = index_of_refraction - 1.0f;
      float b = index_of_refraction + 1.0f;
      float R0 = a * a / (b * b);
      float c = 1.0f - (pt_photon_ray_payload.inside ?
                        dot(refraction_direction, -ffnormal) :
                        cos_i);
      reflection_ratio = R0 + (1.0f - R0) * c * c * c * c * c;
      refraction_ratio = 1.0f - reflection_ratio;
    }
  }

  // refraction
  if (has_refraction) {
    refraction_payload.power *= Rho_s * refraction_ratio;
    refraction_payload.inside = !refraction_payload.inside;
    Ray ray(hit_point,
            refraction_direction,
            pt_photon_ray_type,
            1e-2f);
    rtTrace(top_object, ray, refraction_payload);
    // update num_deposits is important. Otherwise, the photons generated by
    // refraction will be overwritten.
    pt_photon_ray_payload.num_deposits = refraction_payload.num_deposits;
    pt_photon_ray_payload.seed = refraction_payload.seed;
  }

  // reflection
  // to-do: no internal reflection
  if (pt_photon_ray_payload.inside)
    return;

  pt_photon_ray_payload.power *= Rho_s * reflection_ratio;  // perfect reflection
  Ray ray(hit_point,
          reflection_direction,
          pt_photon_ray_type,
          1e-2f);
  rtTrace(top_object, ray, pt_photon_ray_payload);
}

// photon tracing, photon, miss, default material
// do nothing
