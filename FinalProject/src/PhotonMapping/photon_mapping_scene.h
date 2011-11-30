#ifndef PHOTON_MAPPING_SCENE_H_
#define PHOTON_MAPPING_SCENE_H_

#include <string>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include <GLUTDisplay.h>

class PhotonMappingScene : public SampleScene {
 public:
  PhotonMappingScene();

  // defined in the base class
  virtual void initScene(InitialCameraData& camera_data);
  virtual void trace(const RayGenCameraData& camera_data);
  virtual optix::Buffer getOutputBuffer();

  // all function names follow the convention in the base class
  void setDisplayResolution(const optix::uint w, const optix::uint h) {
    width = w;
    height = h;
  }

 private:
  std::string getPTXPath(const std::string& cu_file_name) {
    return std::string(sutilSamplesPtxDir()) +
           "/" +
           "PhotonMapping" +  // executable name
           "_generated_" +
           cu_file_name +  // source file name
           ".ptx";
  }
  void createScene(InitialCameraData& camera_data);
  void createCornellBox(InitialCameraData& camera_data);
  optix::GeometryInstance createParallelogram(
      const float3 anchor,
      const float3 offset1,
      const float3 offset2,
      const optix::Program& intersection,
      const optix::Program& bounding_box,
      const optix::Material& material);
  optix::GeometryInstance createSphere(
      const float3 center,
      const float radius,
      const optix::Program& intersection,
      const optix::Program& bounding_box,
      const optix::Material& material);
  void createPhotonMap();

  enum ProgramEntryPoint {
    pt,
    rt,
    ot,
    num_programs
  };

  enum RayType {
    rt_viewing_ray_type,
    rt_shadow_ray_type,
    pt_photon_ray_type,
    num_ray_types
  };

  optix::Context& context;  // refer to _context from SampleScene
  optix::uint width;
  optix::uint height;
  optix::uint sqrt_num_subpixels;
  optix::uint frame_number;  // to-do: enable progressive rendering
  optix::uint pt_width;
  optix::uint pt_height;
  optix::uint max_num_deposits;
  optix::uint min_depth;
  optix::uint max_depth;
  optix::uint photon_map_size;
  float radius2;
  optix::Buffer photon_record_buffer;
  optix::Buffer photon_map;
  optix::Buffer accumulator;
};

#endif  // PHOTON_MAPPING_SCENE_H_
