#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"  // tea<>, rnd
#include "structs.h"

using namespace optix;

// variables used in multiple programs
rtBuffer<HitRecord, 2> hit_record_buffer;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, rt_viewing_ray_type, , );
rtDeclareVariable(RTViewingRayPayload, rt_viewing_ray_payload, rtPayload, );

// ray tracing, ray generation
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(uint, frame_number, , );
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
  payload.depth = 0;

  rtTrace(top_object, ray, payload);
}

// ray tracing, exception
rtDeclareVariable(float3, bad_color, , );

RT_PROGRAM void rt_exception() {
  HitRecord& hr = hit_record_buffer[launch_index];
  hr.flags = EXCEPTION;
  hr.attenuated_Kd = bad_color;

  rtPrintExceptionDetails();  // to-do:
}

// ray tracing, viewing ray, closest hit, default material
rtDeclareVariable(Ray, rt_viewing_ray, rtCurrentRay, );
rtDeclareVariable(float, hit_t, rtIntersectionDistance, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, Le, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, Ks, , );

RT_PROGRAM void rt_viewing_ray_closest_hit() {
  if (fmaxf(Le) > 0.0f) {  // light source?
    HitRecord& hr = hit_record_buffer[launch_index];
    hr.flags = 0;
    hr.attenuated_Kd = rt_viewing_ray_payload.attenuation * Le;
    return;
  }

  float3 hit_point = rt_viewing_ray.origin + hit_t * rt_viewing_ray.direction;
  float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal = faceforward(world_shading_normal, -rt_viewing_ray.direction, world_geometric_normal);

  if (fmaxf(Kd) > 0.0f) {  // diffuse surface?
    HitRecord& hr = hit_record_buffer[launch_index];
    hr.flags = HIT;
    hr.attenuated_Kd = rt_viewing_ray_payload.attenuation * Kd;  // to-do: BRDF shouldn't be a constant
    hr.position = hit_point;
    hr.normal = ffnormal;
    hr.outgoing = -rt_viewing_ray.direction;
    return;
  }

  // specular surface, recursion
  rt_viewing_ray_payload.attenuation *= Ks;  // to-do: BRDF shouldn't be a constant
  rt_viewing_ray_payload.depth++;  // to-do: unused
  float3 reflection_direction = reflect(rt_viewing_ray.direction, ffnormal);  // inversed incoming
  Ray ray(hit_point,
          reflection_direction,
          rt_viewing_ray_type,
          1e-10f);
  rtTrace(top_object, ray, rt_viewing_ray_payload);
}

// ray tracing, viewing ray, miss, default material
rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void rt_viewing_ray_miss() {
  HitRecord& hr = hit_record_buffer[launch_index];
  hr.flags = 0;
  hr.attenuated_Kd = rt_viewing_ray_payload.attenuation * bg_color;
}
