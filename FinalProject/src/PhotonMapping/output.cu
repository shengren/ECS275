#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

// output, ray generation
rtBuffer<float3, 2> subpixel_accumulator;
rtBuffer<float4, 2> output_buffer;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, frame_number, , );
rtDeclareVariable(uint, sqrt_num_subpixels, , );

RT_PROGRAM void ot_ray_generation() {
  float3 result = make_float3(0.0f);
  for (int i = 0; i < sqrt_num_subpixels; ++i)
    for (int j = 0; j < sqrt_num_subpixels; ++j) {
      uint2 index = make_uint2(launch_index.x * sqrt_num_subpixels + i,
                               launch_index.y * sqrt_num_subpixels + j);
      result += subpixel_accumulator[index];
    }
  result /= (float)(sqrt_num_subpixels * sqrt_num_subpixels);
  result /= (float)frame_number;
  output_buffer[launch_index] = make_float4(result, 0.0f);
}
