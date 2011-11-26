#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"
#include "kdtree.h"
#include "structs.h"
#include "inlines.h"

using namespace optix;

// gathering, ray generation
rtBuffer<HitRecord, 2> hit_record_buffer;
rtBuffer<float4, 2> output_buffer;
rtBuffer<PhotonRecord, 1> photon_map;  // 1D
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(uint, frame_number, , );
rtDeclareVariable(float, total_emitted, , );
rtDeclareVariable(uint, gt_shadow_ray_type, , );
rtBuffer<ParallelogramLight> lights;  // to-do: only have parallelogram lights
rtDeclareVariable(rtObject, top_object, , );

// to-do:
#define MAX_DEPTH 20

RT_PROGRAM void gt_ray_generation() {
  HitRecord hr = hit_record_buffer[launch_index];

  if ((hr.flags & EXCEPTION) || !(hr.flags & HIT)) {
    output_buffer[launch_index] = make_float4(hr.attenuated_Kd);
    return;
  }

  float3 total_flux = make_float3(0.0f);
  int num_photons = 0;
  //float radius2 = 0.25f;  // to-do: input parameter!!!
  float radius2 = 5.0f;  // to-do: input parameter!!!

  // modified from gather() in ppm_gather.cu
  // begin
  unsigned int stack[MAX_DEPTH];
  unsigned int stack_current = 0;
  unsigned int node = 0;

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

  push_node(0);

  int photon_map_size = photon_map.size();
  if (launch_index.x < 2 && launch_index.y < 2) {
    output_buffer[launch_index] = make_float4((float)photon_map_size, 0.0, 0.0, 0.0);
    return;
  }

  do {
    // check
    if (!(node < photon_map_size)) {
      output_buffer[launch_index] = make_float4(1.0, 1.0, 0.0, 0.0);
      return;
    }

    PhotonRecord& pr = photon_map[node];

    if (!(pr.axis & PPM_NULL)) {
      float3 diff = hr.position - pr.position;
      float distance2 = dot(diff, diff);

      // accumulate photon
      if (distance2 <= radius2) {
        /*
        if (dot(pr.normal, hr.normal) > 1e-5) {
          //total_flux += pr.power * hr.attenuated_Kd;
          total_flux += pr.power;
          num_photons++;
        }
        */
        total_flux += pr.power;
        num_photons++;
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
          // check
          if (!(stack_current + 1 < MAX_DEPTH)) {
            output_buffer[launch_index] = make_float4(0.0, 1.0, 0.0, 0.0);
            return;
          }

          push_node((node << 1) + 2 - selector);
        }

        // check
        if (!(stack_current + 1 < MAX_DEPTH)) {
          output_buffer[launch_index] = make_float4(0.0, 1.0, 1.0, 0.0);
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
  // end

  // indirect
  // to-do:
  //float3 indirect = total_flux / (M_PI * radius2) / total_emitted;
  float3 indirect;
  if (num_photons > 0)
    indirect = make_float3(1.0f);
  else
    indirect = make_float3(0.0f);

  // direct
  float3 direct = make_float3(0.0f);
  /*
  uint seed = tea<16>(launch_index.y * launch_dim.x + launch_index.x,
                      frame_number);
  // uniformly choose one of the area lights
  int i = (int)((float)lights.size() * rnd(seed));
  ParallelogramLight light = lights[i];  // to-do: only one type of light now

  float3 jitter_scale_v1 = light.v1 / (float)light.sqrt_num_samples;
  float3 jitter_scale_v2 = light.v2 / (float)light.sqrt_num_samples;
  int num_samples = light.sqrt_num_samples * light.sqrt_num_samples;
  float ratio = 0.0;
  for (int x = 0; x < light.sqrt_num_samples; ++x)
    for (int y = 0; y < light.sqrt_num_samples; ++y) {
      float3 sample_on_light =
          light.corner +
          jitter_scale_v1 * (x + rnd(seed)) +
          jitter_scale_v2 * (y + rnd(seed));
      float distance_to_light = length(sample_on_light - hr.position);
      float3 direction_to_light = normalize(sample_on_light - hr.position);

      if (dot(hr.normal, direction_to_light) > 0.0f &&
          dot(light.normal, -direction_to_light) > 0.0f) {  // trace shadow ray
        GTShadowRayPayload payload;
        payload.blocked = false;

        Ray ray(hr.position,
                direction_to_light,
                gt_shadow_ray_type,
                1e-10f,
                distance_to_light - 1e-10f);

        rtTrace(top_object, ray, payload);  // to-do: hitting the light source doesn't count

        if (!payload.blocked) {
          float BRDF = getDiffuseBRDF();
          float geom = getGeometry(hr.normal,
                                   light.normal,
                                   direction_to_light,
                                   distance_to_light);
          ratio += BRDF * geom;
        }
      }
    }
  ratio *= light.area / (float)num_samples;
  ratio *= (float)lights.size();
  direct = light.emitted * ratio;
  */

  // output
  //output_buffer[launch_index] = make_float4((indirect + direct) * hr.attenuated_Kd,
  //                                          0.0f);
  output_buffer[launch_index] = make_float4((indirect + direct) * hr.Rho_d * hr.attenuated_Kd,
                                            0.0f);
}

// gathering, exception
rtDeclareVariable(float3, bad_color, , );

RT_PROGRAM void gt_exception() {
  output_buffer[launch_index] = make_float4(bad_color);
}

// gathering, direct illumination, shadow ray, any hit
rtDeclareVariable(GTShadowRayPayload, gt_shadow_ray_payload, rtPayload, );

RT_PROGRAM void gt_shadow_ray_any_hit() {
  gt_shadow_ray_payload.blocked = true;
  rtTerminateRay();
}
