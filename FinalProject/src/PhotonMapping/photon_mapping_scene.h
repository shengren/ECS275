#ifndef PHOTON_MAPPING_SCENE_H_
#define PHOTON_MAPPING_SCENE_H_

#include <string>

#include <optixu/optixpp_namespace.h>
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
  void setDisplayResolution(const unsigned int w, const unsigned int h) {
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
      const float3& anchor,
      const float3& offset1,
      const float3& offset2,
      const optix::Program& intersection,
      const optix::Program& bounding_box,
      const optix::Material& material);
  void createPhotonMap();

  enum ProgramEntryPoint {
    rt,
    pt,
    gt,
    num_programs
  };
  enum RayType {
    rt_viewing_ray_type,
    pt_photon_ray_type,
    gt_shadow_ray_type,
    num_ray_types
  };
  optix::Context& context;  // refer to _context from SampleScene
  unsigned int width;
  unsigned int height;
  unsigned int frame_number;  // to-do: enable progressive rendering
  unsigned int pt_width;
  unsigned int pt_height;
  unsigned int max_num_deposits;
  unsigned int min_depth;
  unsigned int max_depth;
  unsigned int photon_map_size;
  optix::Buffer hit_record_buffer;
  optix::Buffer photon_record_buffer;
  optix::Buffer photon_map;
};

#endif  // PHOTON_MAPPING_SCENE_H_
