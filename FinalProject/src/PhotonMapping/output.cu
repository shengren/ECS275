#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

// output, ray generation
rtBuffer<float3, 2> accumulator;
rtBuffer<float4, 2> output_buffer;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, frame_number, , );
rtDeclareVariable(uint, sqrt_num_subpixels, , );

RT_PROGRAM void ot_ray_generation() {
  float3 result = accumulator[launch_index] / (float)frame_number;
  output_buffer[launch_index] = make_float4(result, 0.0f);
}
