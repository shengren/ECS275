#include <cfloat>

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"  // tea<>, rnd()
#include "kdtree.h"
#include "structs.h"
#include "inlines.h"

using namespace optix;

// variables used in multiple programs
rtBuffer<float4, 2> output_buffer;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint, rt_viewing_ray_type, , );
rtDeclareVariable(RTViewingRayPayload, rt_viewing_ray_payload, rtPayload, );

// ray tracing, ray generation
rtDeclareVariable(uint, frame_number, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(float3, camera_position, , );
rtDeclareVariable(float3, camera_u, , );
rtDeclareVariable(float3, camera_v, , );
rtDeclareVariable(float3, camera_w, , );
rtDeclareVariable(uint, sqrt_num_subpixels, , );

RT_PROGRAM void rt_ray_generation() {
  uint seed = tea<16>(launch_index.y * launch_dim.x + launch_index.x,
                      frame_number);
  float2 base = make_float2(launch_index.x * sqrt_num_subpixels,
                            launch_index.y * sqrt_num_subpixels);
  float2 resolution = make_float2(launch_dim.x * sqrt_num_subpixels,
                                  launch_dim.y * sqrt_num_subpixels);
  float3 result = make_float3(0.0f);
  for (int i = 0; i < sqrt_num_subpixels; ++i)
    for (int j = 0; j < sqrt_num_subpixels; ++j) {  // to-do: this anti-aliasing needs a larger stack size
      float2 offset = (base + make_float2(i + rnd(seed), j + rnd(seed)))
                      / resolution * 2.0f - 1.0f;

      Ray ray(camera_position,  // origin
              normalize(offset.x * camera_u + offset.y * camera_v + camera_w),  // direction
              rt_viewing_ray_type,  // type
              1e-2f);  // tmin; tmax uses default

      RTViewingRayPayload payload;
      payload.attenuation = make_float3(1.0f);
      payload.radiance = make_float3(0.0f);
      payload.depth = 1;
      payload.seed = seed;
      payload.inside = false;

      rtTrace(top_object, ray, payload);

      result += payload.radiance;
      seed = payload.seed;
    }
  result *= 1.0f / (float)(sqrt_num_subpixels * sqrt_num_subpixels);

  if (frame_number == 1) {
    output_buffer[launch_index] = make_float4(result, 0.0f);
  } else {
    float a = 1.0f / (float)frame_number;
    float b = ((float)frame_number - 1.0f) * a;
    float3 old_color = make_float3(output_buffer[launch_index]);
    output_buffer[launch_index] = make_float4(a * result + b * old_color, 0.0f);
  }
}

// ray tracing, exception
rtDeclareVariable(float3, bad_color, , );  // blue

RT_PROGRAM void rt_exception() {
  output_buffer[launch_index] = make_float4(bad_color, 0.0f);
  rtPrintExceptionDetails();  // to-do: for debugging
}

// ray tracing, viewing ray, closest hit, default material
rtBuffer<PhotonRecord, 1> photon_map;  // 1D
rtBuffer<ParallelogramLight> lights;  // to-do: only have parallelogram lights
rtDeclareVariable(Ray, rt_viewing_ray, rtCurrentRay, );
rtDeclareVariable(float, hit_t, rtIntersectionDistance, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, Le, , );
rtDeclareVariable(float3, Rho_d, , );
rtDeclareVariable(float3, Rho_s, , );
rtDeclareVariable(float, shininess, , );  // unused now
rtDeclareVariable(float, index_of_refraction, , );  // non-zero indiates a refraction surface, Rho_s is needed as well
rtDeclareVariable(uint, viewing_ray_max_depth, , );
rtDeclareVariable(uint, rt_shadow_ray_type, , );
rtDeclareVariable(float, radius2, , );

// to-do: photon_map (rtBuffer) cannot be passed through function parameters
// hence this function is not included in 'inlines.h'
// output_buffer is used as a debug buffer in some places
// modified from gather() in progressivePhotonMap/ppm_gather.cu
#define MAX_DEPTH 24  // to-do: 2^24-1 is the maximal size of the photon map

__device__ __inline__ void estimateRadiance(const float3 position,
                                            const float3 normal,
                                            const float3 Rho_d,
                                            const float radius2,
                                            float3& total_flux,
                                            int& num_photons,
                                            float& max_radius2) {
  total_flux = make_float3(0.0f, 0.0f, 0.0f);
  num_photons = 0;  // to-do: unused now
  max_radius2 = 0.0f;

  /*
  const int max_heap_size = (1 << 6) - 1;
  Neighbor max_heap[max_heap_size];
  for (int i = 0; i < max_heap_size; ++i) {
    max_heap[i].dist2 = FLT_MAX;
    max_heap[i].idx = -1;
  }
  */

  unsigned int stack[MAX_DEPTH];
  unsigned int stack_current = 0;
  unsigned int node = 0;  // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

  push_node(0);

  int photon_map_size = photon_map.size();  // for debugging

  do {
    // debugging assertion
    if (!(node < photon_map_size)) {
      //output_buffer[rt_viewing_ray_payload.index] = make_float4(1.0f, 1.0f, 0.0f, 0.0f);
      rtPrintf("overflow case 1\n");
      return;
    }

    const PhotonRecord& pr = photon_map[node];

    if (!(pr.axis & PPM_NULL)) {
      float3 diff = position - pr.position;
      float distance2 = dot(diff, diff);

      // accumulate photons
      if (distance2 <= radius2) {
        //if (dot(normal, pr.normal) > 1e-3f) {  // on the same plane?
        if (dot(normal, pr.incoming) > 1e-3f) {  // to-do: better way?
          total_flux += pr.power * getDiffuseBRDF(Rho_d);  // with BRDF
          num_photons++;
          if (distance2 > max_radius2)
            max_radius2 = distance2;
        }
        /*
        if (dot(normal, pr.normal) > 1e-3f &&
            distance2 < max_heap[0].dist2) {  // heap insertion
          max_heap[0].dist2 = distance2;
          max_heap[0].idx = node;
          int p = 0;
          while (p * 2 + 2 < max_heap_size) {
            if (max_heap[p * 2 + 1].dist2 > max_heap[p * 2 + 2].dist2) {
              if (max_heap[p * 2 + 1].dist2 > max_heap[p].dist2) {
                Neighbor t = max_heap[p * 2 + 1];
                max_heap[p * 2 + 1] = max_heap[p];
                max_heap[p] = t;
                p = p * 2 + 1;
              } else {
                break;
              }
            } else {
              if (max_heap[p * 2 + 2].dist2 > max_heap[p].dist2) {
                Neighbor t = max_heap[p * 2 + 2];
                max_heap[p * 2 + 2] = max_heap[p];
                max_heap[p] = t;
                p = p * 2 + 2;
              } else {
                break;
              }
            }
          }  // while
        }
        */
      }

      // Recurse
      if (!(pr.axis & PPM_LEAF)) {
        float d;
        if (pr.axis & PPM_X) d = diff.x;
        else if (pr.axis & PPM_Y) d = diff.y;
        else d = diff.z;  // PPM_Z

        // Calculate the next child selector. 0 is left, 1 is right.
        int selector = d < 0.0f ? 0 : 1;
        if (d * d < radius2) {
          // debugging assertion
          if (!(stack_current + 1 < MAX_DEPTH)) {
            //output_buffer[rt_viewing_ray_payload.index] = make_float4(1.0f, 1.0f, 0.0f, 0.0f);
            rtPrintf("overflow case 2\n");
            return;
          }

          push_node((node << 1) + 2 - selector);
        }

        // debugging assertion
        if (!(stack_current + 1 < MAX_DEPTH)) {
          //output_buffer[rt_viewing_ray_payload.index] = make_float4(1.0f, 1.0f, 0.0f, 0.0f);
          rtPrintf("overflow case 3\n");
          return;
        }

        node = (node << 1) + 1 + selector;
      } else {
        node = pop_node();
      }
    } else {
      node = pop_node();
    }
  } while (node);

  /*
  for (int i = 0; i < max_heap_size; ++i) {
    if (max_heap[i].idx != -1) {
      total_flux += photon_map[max_heap[i].idx].power * getDiffuseBRDF(Rho_d);  // with BRDF
      num_photons++;
      if (max_heap[i].dist2 > max_radius2)
        max_radius2 = max_heap[i].dist2;
    }
  }
  */
}

// to-do: make this function separate in order to make the code clean
// launch_index, launch_dim, frame_number, lights are local variables
__device__ __inline__ float3 directIllumination(const float3 position,
                                                const float3 normal,
                                                const float3 Rho_d,
                                                uint& seed) {
  // uniformly choose one of the area lights
  int i = (int)((float)lights.size() * rnd(seed));
  const ParallelogramLight& light = lights[i];  // to-do: only one type of light now

  float3 jitter_scale_v1 = light.v1 / (float)light.sqrt_num_samples;
  float3 jitter_scale_v2 = light.v2 / (float)light.sqrt_num_samples;
  int num_samples = light.sqrt_num_samples * light.sqrt_num_samples;
  float ratio = 0.0;
  for (int x = 0; x < light.sqrt_num_samples; ++x)
    for (int y = 0; y < light.sqrt_num_samples; ++y) {
      float3 sample_on_light = light.corner +
                               jitter_scale_v1 * (x + rnd(seed)) +
                               jitter_scale_v2 * (y + rnd(seed));
      float distance_to_light = length(sample_on_light - position);
      float3 direction_to_light = normalize(sample_on_light - position);

      if (dot(normal, direction_to_light) > 1e-2f &&
          dot(light.normal, -direction_to_light) > 1e-2f) {  // trace shadow ray
        RTShadowRayPayload payload;
        payload.blocked = false;

        Ray ray(position,
                direction_to_light,
                rt_shadow_ray_type,
                1e-2f,
                distance_to_light - 1e-2f);

        rtTrace(top_object, ray, payload);  // to-do: hitting the light source doesn't count

        if (!payload.blocked) {
          float geom = getGeometry(normal,
                                   light.normal,
                                   direction_to_light,
                                   distance_to_light);
          ratio += geom;
        }
      }
    }
  ratio *= light.area;  // probability of sampling on this light source
  ratio *= (float)lights.size();  // probability of sampling among light sources
  ratio /= (float)num_samples;

  return light.emitted * getDiffuseBRDF(Rho_d) * ratio;
}

__device__ __inline__ float3 shade(const float3 position,
                                   const float3 normal,
                                   const float3 Rho_d,
                                   uint& seed) {
  // indirect illumination
  float3 total_flux = make_float3(0.0f);
  int num_photons = 0;
  float max_radius2 = 0.0f;

  estimateRadiance(position, normal, Rho_d, radius2, total_flux, num_photons, max_radius2);

  float3 indirect = total_flux / (M_PI * max_radius2);

  // direct illumination
  float3 direct = directIllumination(position, normal, Rho_d, seed);

  float3 ret = direct + indirect;
  //float3 ret = indirect;
  //float3 ret = direct;

  return ret;
}

RT_PROGRAM void rt_viewing_ray_closest_hit() {
  if (fmaxf(Le) > 0.0f) {  // light source?
    rt_viewing_ray_payload.radiance = rt_viewing_ray_payload.attenuation * Le;
    return;
  }

  float3 hit_point = rt_viewing_ray.origin + hit_t * rt_viewing_ray.direction;
  float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal = faceforward(world_shading_normal, -rt_viewing_ray.direction, world_geometric_normal);

  if (fmaxf(Rho_d) > 0.0f) {  // diffuse surface?
    rt_viewing_ray_payload.radiance = rt_viewing_ray_payload.attenuation *
        shade(hit_point, ffnormal, Rho_d, rt_viewing_ray_payload.seed);
    return;
  }

  if (rt_viewing_ray_payload.depth > viewing_ray_max_depth)
    return;  // stop recursion

  rt_viewing_ray_payload.depth++;

  float3 reflection_direction = reflect(rt_viewing_ray.direction, ffnormal);  // inversed incoming
  float3 refraction_direction;
  float reflection_ratio = 1.0f;
  float refraction_ratio = 0.0f;
  RTViewingRayPayload reflection_payload = rt_viewing_ray_payload;
  RTViewingRayPayload refraction_payload = rt_viewing_ray_payload;
  bool has_refraction = false;
  if (index_of_refraction > 0.0f) {
    float iof = (rt_viewing_ray_payload.inside) ?
                (1.0f / index_of_refraction) : index_of_refraction;
    refract(refraction_direction, rt_viewing_ray.direction, ffnormal, iof);
    float cos_i = dot(-rt_viewing_ray.direction, ffnormal);
    float cos2_t = 1.0f - ((1.0f - (cos_i * cos_i)) / (iof * iof));
    if (cos2_t >= 0) {
      has_refraction = true;
      float a = index_of_refraction - 1.0f;
      float b = index_of_refraction + 1.0f;
      float R0 = a * a / (b * b);
      float c = 1.0f - (rt_viewing_ray_payload.inside ?
                        dot(refraction_direction, -ffnormal) :
                        cos_i);
      reflection_ratio = R0 + (1.0f - R0) * c * c * c * c * c;
      refraction_ratio = 1.0f - reflection_ratio;
    }
  }

  // refraction
  if (has_refraction) {
    refraction_payload.attenuation *= Rho_s * refraction_ratio;
    refraction_payload.inside = !refraction_payload.inside;
    Ray ray(hit_point,
            refraction_direction,
            rt_viewing_ray_type,
            1e-2f);
    rtTrace(top_object, ray, refraction_payload);
    rt_viewing_ray_payload.radiance += refraction_payload.radiance;  // recursively return
    // update the seed
    reflection_payload.seed = refraction_payload.seed;
  }

  // reflection
  reflection_payload.attenuation *= Rho_s * reflection_ratio;  // perfect reflection
  Ray ray(hit_point,
          reflection_direction,
          rt_viewing_ray_type,
          1e-2f);
  rtTrace(top_object, ray, reflection_payload);
  rt_viewing_ray_payload.radiance += reflection_payload.radiance;  // recursively return
  // update the seed
  rt_viewing_ray_payload.seed = reflection_payload.seed;
}

// ray tracing, viewing ray, miss, default material
rtDeclareVariable(float3, bg_color, , );  // black

RT_PROGRAM void rt_viewing_ray_miss() {
  rt_viewing_ray_payload.radiance = rt_viewing_ray_payload.attenuation * bg_color;
}

// ray tracing, direct illumination, shadow ray, any hit
rtDeclareVariable(RTShadowRayPayload, rt_shadow_ray_payload, rtPayload, );

RT_PROGRAM void rt_shadow_ray_any_hit() {
  rt_shadow_ray_payload.blocked = true;
  rtTerminateRay();  // to-do: what's the difference between this function and 'return'
}
