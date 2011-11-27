#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"
#include "kdtree.h"
#include "structs.h"
#include "inlines.h"

using namespace optix;

// variables used in multiple programs
//rtBuffer<float4, 2> output_buffer;
rtBuffer<float3, 2> subpixel_accumulator;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

// gathering, ray generation
rtBuffer<HitRecord, 2> hit_record_buffer;
rtBuffer<PhotonRecord, 1> photon_map;  // 1D
rtBuffer<ParallelogramLight> lights;  // to-do: only have parallelogram lights
rtDeclareVariable(uint, frame_number, , );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(uint, gt_shadow_ray_type, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(float, radius2, , );

// to-do: photon_map (rtBuffer) cannot be passed through function parameters
// hence this function is not included in 'inlines.h'
// output_buffer is used as a debug buffer in some places
// modified from gather() in progressivePhotonMap/ppm_gather.cu
#define MAX_DEPTH 20  // one MILLION photons

__device__ __inline__ void estimateRadiance(const HitRecord& hr,
                                            const float& radius2,
                                            float3& total_flux,
                                            int& num_photons) {
  total_flux = make_float3(0.0f, 0.0f, 0.0f);
  num_photons = 0;  // to-do: unused now

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
      //output_buffer[launch_index] = make_float4(1.0f, 1.0f, 0.0f, 0.0f);
      subpixel_accumulator[launch_index] = make_float3(1.0f, 1.0f, 0.0f);
      return;
    }

    const PhotonRecord& pr = photon_map[node];

    if (!(pr.axis & PPM_NULL)) {
      float3 diff = hr.position - pr.position;
      float distance2 = dot(diff, diff);

      // accumulate photons
      if (distance2 <= radius2) {
        if (dot(hr.normal, pr.normal) > 1e-2f) {  // on the same plane?
          total_flux += pr.power * getDiffuseBRDF(hr.Rho_d);  // with BRDF
          num_photons++;
        }
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
            //output_buffer[launch_index] = make_float4(1.0f, 1.0f, 0.0f, 0.0f);
            subpixel_accumulator[launch_index] = make_float3(1.0f, 1.0f, 0.0f);
            return;
          }

          push_node((node << 1) + 2 - selector);
        }

        // debugging assertion
        if (!(stack_current + 1 < MAX_DEPTH)) {
          //output_buffer[launch_index] = make_float4(1.0f, 1.0f, 0.0f, 0.0f);
          subpixel_accumulator[launch_index] = make_float3(1.0f, 1.0f, 0.0f);
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
}

// to-do: make this function separate in order to make the code clean
// launch_index, launch_dim, frame_number, lights are local variables
__device__ __inline__ float3 directIllumination(const HitRecord& hr) {
  uint seed = tea<16>(launch_index.y * launch_dim.x + launch_index.x,
                      frame_number);

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
      float distance_to_light = length(sample_on_light - hr.position);
      float3 direction_to_light = normalize(sample_on_light - hr.position);

      if (dot(hr.normal, direction_to_light) > 1e-5f &&
          dot(light.normal, -direction_to_light) > 1e-5f) {  // trace shadow ray
        GTShadowRayPayload payload;
        payload.blocked = false;

        Ray ray(hr.position,
                direction_to_light,
                gt_shadow_ray_type,
                1e-10f,
                distance_to_light - 1e-10f);

        rtTrace(top_object, ray, payload);  // to-do: hitting the light source doesn't count

        if (!payload.blocked) {
          float geom = getGeometry(hr.normal,
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

  return light.emitted * getDiffuseBRDF(hr.Rho_d) * ratio;
}

RT_PROGRAM void gt_ray_generation() {
  // clean the output buffer
  if (frame_number == 1) {
    //output_buffer[launch_index] = make_float4(0.0f);
    subpixel_accumulator[launch_index] = make_float3(0.0f);
  }

  const HitRecord& hr = hit_record_buffer[launch_index];

  if (!(hr.flags & HIT)) {
    //output_buffer[launch_index] = make_float4(hr.attenuation);
    subpixel_accumulator[launch_index] += hr.attenuation;
    return;
  }

  // indirect illumination
  float3 total_flux = make_float3(0.0f);
  int num_photons = 0;  // to-do: unused now

  estimateRadiance(hr, radius2, total_flux, num_photons);

  float3 indirect = total_flux / (M_PI * radius2);

  // direct illumination
  float3 direct = directIllumination(hr);

  // output
  //float3 result = direct * hr.attenuation;
  //float3 result = indirect * hr.attenuation;
  float3 result = (direct + indirect) * hr.attenuation;

  /*
  if (frame_number == 1) {
    output_buffer[launch_index] = make_float4(result, 0.0f);
  } else {
    float a = 1.0f / (float)frame_number;
    float b = ((float)frame_number - 1.0f) * a;
    float3 old_result = make_float3(output_buffer[launch_index]);
    output_buffer[launch_index] = make_float4(a * result + b * old_result, 0.0f);
  }
  */
  subpixel_accumulator[launch_index] += result;
}

// gathering, exception
rtDeclareVariable(float3, bad_color, , );

RT_PROGRAM void gt_exception() {
  //output_buffer[launch_index] = make_float4(bad_color);
  subpixel_accumulator[launch_index] = bad_color;

  rtPrintExceptionDetails();  // to-do: for debugging
}

// gathering, direct illumination, shadow ray, any hit
rtDeclareVariable(GTShadowRayPayload, gt_shadow_ray_payload, rtPayload, );

RT_PROGRAM void gt_shadow_ray_any_hit() {
  gt_shadow_ray_payload.blocked = true;
  rtTerminateRay();  // to-do: what's the difference between this function and 'return'
}
