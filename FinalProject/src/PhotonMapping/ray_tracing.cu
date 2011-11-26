#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"  // tea<>, rnd()
#include "structs.h"
#include "inlines.h"

using namespace optix;

// variables used in multiple programs
rtBuffer<HitRecord, 2> hit_record_buffer;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint, rt_viewing_ray_type, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(RTViewingRayPayload, rt_viewing_ray_payload, rtPayload, );

// ray tracing, ray generation
rtDeclareVariable(uint, frame_number, , );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(float3, camera_position, , );
rtDeclareVariable(float3, camera_u, , );
rtDeclareVariable(float3, camera_v, , );
rtDeclareVariable(float3, camera_w, , );

RT_PROGRAM void rt_ray_generation() {
  uint seed = tea<16>(launch_index.y * launch_dim.x + launch_index.x,
                      frame_number);
  float2 offset = (make_float2(launch_index)
                  + make_float2(rnd(seed), rnd(seed)))
                  / make_float2(launch_dim) * 2.0f - 1.0f;

  Ray ray(camera_position,  // origin
          normalize(offset.x * camera_u + offset.y * camera_v + camera_w),  // direction
          rt_viewing_ray_type,  // type
          1e-10f);  // tmin; tmax uses default

  RTViewingRayPayload payload;
  payload.attenuation = make_float3(1.0f);
  payload.depth = 1;  // to-do: unused now

  rtTrace(top_object, ray, payload);
}

// ray tracing, exception
rtDeclareVariable(float3, bad_color, , );  // green

RT_PROGRAM void rt_exception() {
  HitRecord& hr = hit_record_buffer[launch_index];
  hr.flags = EXCEPTION;
  hr.attenuation = bad_color;
  hr.position = hr.normal = hr.outgoing = hr.Rho_d = make_float3(0.0f);

  rtPrintExceptionDetails();  // to-do: for debugging
}

// ray tracing, viewing ray, closest hit, default material
rtDeclareVariable(Ray, rt_viewing_ray, rtCurrentRay, );
rtDeclareVariable(float, hit_t, rtIntersectionDistance, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, Le, , );
rtDeclareVariable(float3, Rho_d, , );
rtDeclareVariable(float3, Rho_s, , );
rtDeclareVariable(float, shininess, , );

RT_PROGRAM void rt_viewing_ray_closest_hit() {
  if (fmaxf(Le) > 0.0f) {  // light source?
    HitRecord& hr = hit_record_buffer[launch_index];
    hr.flags = HIT_LIGHT;
    hr.attenuation = rt_viewing_ray_payload.attenuation * Le;
    hr.position = hr.normal = hr.outgoing = hr.Rho_d = make_float3(0.0f);
    return;
  }

  float3 hit_point = rt_viewing_ray.origin + hit_t * rt_viewing_ray.direction;
  float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal = faceforward(world_shading_normal, -rt_viewing_ray.direction, world_geometric_normal);

  if (fmaxf(Rho_d) > 0.0f) {  // diffuse surface?
    HitRecord& hr = hit_record_buffer[launch_index];
    hr.flags = HIT;
    // since we don't know the incoming directions, here we don't apply the
    // BRDF and cosine term. They should be computed in the gathering pass.
    // i.e. attenuation only includes all previous hits' computations on
    // specular surfaces.
    hr.attenuation = rt_viewing_ray_payload.attenuation;
    hr.position = hit_point;
    hr.normal = ffnormal;
    hr.outgoing = -rt_viewing_ray.direction;
    hr.Rho_d = Rho_d;
    return;
  }

  // specular surface, recursion
  rt_viewing_ray_payload.depth++;  // to-do: unused now
  float3 reflection_direction = reflect(rt_viewing_ray.direction, ffnormal);  // inversed incoming
  rt_viewing_ray_payload.attenuation *= getSpecularBRDF(reflection_direction,  // incoming
                                                        ffnormal,  // normal
                                                        -rt_viewing_ray.direction,  // outgoing
                                                        Rho_s,  // not Ks but for computing Ks
                                                        shininess);  // the power factor
  rt_viewing_ray_payload.attenuation *= dot(reflection_direction, ffnormal);  // cosine term
  Ray ray(hit_point,
          reflection_direction,
          rt_viewing_ray_type,
          1e-10f);
  rtTrace(top_object, ray, rt_viewing_ray_payload);
}

// ray tracing, viewing ray, miss, default material
rtDeclareVariable(float3, bg_color, , );  // black

RT_PROGRAM void rt_viewing_ray_miss() {
  HitRecord& hr = hit_record_buffer[launch_index];
  hr.flags = HIT_BACKGROUND;
  hr.attenuation = rt_viewing_ray_payload.attenuation * bg_color;
  hr.position = hr.normal = hr.outgoing = hr.Rho_d = make_float3(0.0f);
}
