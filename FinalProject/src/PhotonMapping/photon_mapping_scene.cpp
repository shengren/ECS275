#include "photon_mapping_scene.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include <memory.h>  // memcpy

#include <optixu/optixu_math_namespace.h>

#include "kdtree.h"
#include "structs.h"
#include "inlines.h"

using namespace std;
using namespace optix;

PhotonMappingScene::PhotonMappingScene()
    : context(_context),
      width(512),
      height(512),
      frame_number(0),
      pt_width(100),
      pt_height(100),
      max_num_deposits(2)
{}

void PhotonMappingScene::initScene(InitialCameraData& camera_data) {
  context->setEntryPointCount(num_programs);  // rt, pt, gt = 3
  context->setStackSize(2000);  // to-do: tuning

  context["bad_color"]->setFloat(make_float3(0.0f, 1.0f, 0.0f));  // green
  context["bg_color"]->setFloat(make_float3(0.0f));  // black
  context["max_depth"]->setUint(3);  // to-do: input parameters?
  context["max_num_deposits"]->setUint(max_num_deposits);

  context->setRayTypeCount(num_ray_types);
  context["rt_viewing_ray_type"]->setUint(rt_viewing_ray_type);
  context["pt_photon_ray_type"]->setUint(pt_photon_ray_type);
  context["gt_shadow_ray_type"]->setUint(gt_shadow_ray_type);

  // ray tracing

  // camera ray generation parameters
  // camera_data is handled by GLUT for updating camera parameters.
  // these parameters here are used in kernels and will be set before tracing.
  context["camera_position"]->setFloat(make_float3(0.0f));
  context["camera_u"]->setFloat(make_float3(0.0f));
  context["camera_v"]->setFloat(make_float3(0.0f));
  context["camera_w"]->setFloat(make_float3(0.0f));
  context["frame_number"]->setUint(0);  // for progressive rendering

  hit_record_buffer = context->createBuffer(RT_BUFFER_OUTPUT);
  hit_record_buffer->setFormat(RT_FORMAT_USER);
  hit_record_buffer->setElementSize(sizeof(HitRecord));
  hit_record_buffer->setSize(width, height);
  context["hit_record_buffer"]->set(hit_record_buffer);

  context->setRayGenerationProgram(
      rt,
      context->createProgramFromPTXFile(getPTXPath("ray_tracing.cu"),
                                        "rt_ray_generation"));
  context->setExceptionProgram(
      rt,
      context->createProgramFromPTXFile(getPTXPath("ray_tracing.cu"),
                                        "rt_exception"));
  context->setMissProgram(
      rt_viewing_ray_type,
      context->createProgramFromPTXFile(getPTXPath("ray_tracing.cu"),
                                                   "rt_viewing_ray_miss"));

  // photon tracing

  photon_record_buffer = context->createBuffer(RT_BUFFER_OUTPUT);
  photon_record_buffer->setFormat(RT_FORMAT_USER);
  photon_record_buffer->setElementSize(sizeof(PhotonRecord));
  photon_record_buffer->setSize(pt_width * pt_height * max_num_deposits);  // to-do: should be a parameter!!!
  context["photon_record_buffer"]->set(photon_record_buffer);

  context->setRayGenerationProgram(
      pt,
      context->createProgramFromPTXFile(getPTXPath("photon_tracing.cu"),
                                        "pt_ray_generation"));
  context->setExceptionProgram(
      pt,
      context->createProgramFromPTXFile(getPTXPath("photon_tracing.cu"),
                                        "pt_exception"));

  // knn

  photon_map_size = pow2roundup(pt_width * pt_height * max_num_deposits) - 1;  // to-do: should be a parameter!!!
  photon_map = context->createBuffer(RT_BUFFER_INPUT);
  photon_map->setFormat(RT_FORMAT_USER);
  photon_map->setElementSize(sizeof(PhotonRecord));
  photon_map->setSize(photon_map_size);
  context["photon_map"]->set(photon_map);

  // gather
  
  context["total_emitted"]->setFloat(pt_width * pt_height);

  context["output_buffer"]->set(
      createOutputBuffer(RT_FORMAT_FLOAT4, width, height));  // to-do: why FLOAT4?

  context->setRayGenerationProgram(
      gt,
      context->createProgramFromPTXFile(getPTXPath("gather.cu"),
                                        "gt_ray_generation"));
  context->setExceptionProgram(
      gt,
      context->createProgramFromPTXFile(getPTXPath("gather.cu"),
                                        "gt_exception"));

  // scene

  createScene(camera_data);

  context->validate();
  context->compile();
}

void PhotonMappingScene::trace(const RayGenCameraData& camera_data) {
  // to-do: this part enables progressive rendering
  if (_camera_changed) {
    _camera_changed = false;
    frame_number = 0;
  }

  // to-do: for test only
  // render only one frame, but, actually, 'trace' is called twice.
  // on Mac, can't see output if called 'trace' only once.
  // guess, it is related to camera information updates.
  if (frame_number > 0)
    return;

  context["frame_number"]->setUint(frame_number++);

  // set the current camera parameters
  context["camera_position"]->setFloat(camera_data.eye);
  context["camera_u"]->setFloat(camera_data.U);
  context["camera_v"]->setFloat(camera_data.V);
  context["camera_w"]->setFloat(camera_data.W);

  // window size may be changed
  Buffer buffer = getOutputBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize(buffer_width, buffer_height);

  cout << "initialization" << endl;

  // ray tracing
  context->launch(rt,
                  buffer_width,
                  buffer_height);

  cout << "ray tracing" << endl;

  // photon_tracing
  context->launch(pt,
                  pt_width,
                  pt_height);

  cout << "photon tracing" << endl;

  // build photon map

  createPhotonMap();

  cout << "create photon map" << endl;

  // gather
  context->launch(gt,
                  buffer_width,
                  buffer_height);

  cout << "gather" << endl;
}

Buffer PhotonMappingScene::getOutputBuffer() {
  return context["output_buffer"]->getBuffer();
}

GeometryInstance PhotonMappingScene::createParallelogram(const float3& anchor,
                                                         const float3& offset1,
                                                         const float3& offset2,
                                                         const Program& intersection,
                                                         const Program& bounding_box,
                                                         const Material& material) {
  Geometry g = context->createGeometry();
  g->setPrimitiveCount(1);
  g->setIntersectionProgram(intersection);
  g->setBoundingBoxProgram(bounding_box);

  float3 normal = normalize(cross(offset1, offset2));
  float d = dot(normal, anchor);
  float4 plane = make_float4(normal, d);
  float3 v1 = offset1 / dot(offset1, offset1);  // to-do: normalize?
  float3 v2 = offset2 / dot(offset2, offset2);

  g["plane"]->setFloat(plane);
  g["anchor"]->setFloat(anchor);
  g["v1"]->setFloat(v1);
  g["v2"]->setFloat(v2);

  GeometryInstance gi = context->createGeometryInstance();
  gi->setGeometry(g);
  gi->addMaterial(material);
  gi["Kd"]->setFloat(0.0f, 0.0f, 0.0f);
  gi["Ks"]->setFloat(0.0f, 0.0f, 0.0f);
  gi["Le"]->setFloat(0.0f, 0.0f, 0.0f);

  return gi;
}

void PhotonMappingScene::createScene(InitialCameraData& camera_data) {
  // to-do: support other test scenes?
  createCornellBox(camera_data);
}

void PhotonMappingScene::createCornellBox(InitialCameraData& camera_data) {
  camera_data = InitialCameraData(  // to-do: load from scene description file?
      make_float3(278.0f, 273.0f, -800.0f),  // position
      make_float3(278.0f, 273.0f, 0.0f),  // shoot at
      make_float3(0.0f, 1.0f, 0.0f),  // up
      35.0f);  // vfov

  // create a parallelogram light
  ParallelogramLight light;  // to-do: load from scene description file?
  light.corner = make_float3(343.0f, 548.6f, 227.0f);
  light.v1 = make_float3(0.0f, 0.0f, 105.0f);
  light.v2 = make_float3(-130.0f, 0.0, 0.0f);
  light.normal = normalize(cross(light.v1, light.v2));
  light.area = length(cross(light.v1, light.v2));
  light.power = make_float3(0.5e6f, 0.5e6f, 0.5e6f);  // to-do: power?
  light.sqrt_num_samples = 1;
  light.emitted = make_float3(75.0f, 75.0f, 75.0f);
  // add this light to the engine
  Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(ParallelogramLight));
  light_buffer->setSize(1);
  memcpy(light_buffer->map(), &light, sizeof(light));
  light_buffer->unmap();
  context["lights"]->setBuffer(light_buffer);

  // create geometry instances

  // load programs for geometry parallelogram
  Program para_intersection = context->createProgramFromPTXFile(
      getPTXPath("parallelogram.cu"), "intersect");
  Program para_bounding_box = context->createProgramFromPTXFile(
      getPTXPath("parallelogram.cu"), "bounds");

  // create materials
  Material material = context->createMaterial();
  material->setClosestHitProgram(
      rt_viewing_ray_type,
      context->createProgramFromPTXFile(getPTXPath("ray_tracing.cu"),
                                                   "rt_viewing_ray_closest_hit"));
  material->setClosestHitProgram(
      pt_photon_ray_type,
      context->createProgramFromPTXFile(getPTXPath("photon_tracing.cu"),
                                                   "pt_photon_ray_closest_hit"));
  material->setAnyHitProgram(
      gt_shadow_ray_type,
      context->createProgramFromPTXFile(getPTXPath("gather.cu"),
                                        "gt_shadow_ray_any_hit"));

  const float3 white = make_float3(0.8f, 0.8f, 0.8f);
  const float3 green = make_float3(0.05f, 0.8f, 0.05f);
  const float3 red = make_float3(0.8f, 0.05f, 0.05f);
  const float3 black = make_float3(0.0f);

  vector<GeometryInstance> gis;

  // Floor
  gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
                                    make_float3(0.0f, 0.0f, 559.2f),
                                    make_float3(556.0f, 0.0f, 0.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);  // to-do: find a better way to do this
  // Ceiling
  gis.push_back(createParallelogram(make_float3(0.0f, 548.8f, 0.0f),
                                    make_float3(556.0f, 0.0f, 0.0f),
                                    make_float3(0.0f, 0.0f, 559.2f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  // Back wall
  gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 559.2f),
                                    make_float3(0.0f, 548.8f, 0.0f),
                                    make_float3(556.0f, 0.0f, 0.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  // Right wall
  gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
                                    make_float3(0.0f, 548.8f, 0.0f),
                                    make_float3(0.0f, 0.0f, 559.2f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(green);
  // Left wall
  gis.push_back(createParallelogram(make_float3(556.0f, 0.0f, 0.0f),
                                    make_float3(0.0f, 0.0f, 559.2f),
                                    make_float3(0.0f, 548.8f, 0.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(red);
  // Short block
  gis.push_back(createParallelogram(make_float3(130.0f, 165.0f, 65.0f),
                                    make_float3(-48.0f, 0.0f, 160.0f),
                                    make_float3(160.0f, 0.0f, 49.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  gis.push_back(createParallelogram(make_float3(290.0f, 0.0f, 114.0f),
                                    make_float3(0.0f, 165.0f, 0.0f),
                                    make_float3(-50.0f, 0.0f, 158.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  gis.push_back(createParallelogram(make_float3(130.0f, 0.0f, 65.0f),
                                    make_float3(0.0f, 165.0f, 0.0f),
                                    make_float3(160.0f, 0.0f, 49.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  gis.push_back(createParallelogram(make_float3(82.0f, 0.0f, 225.0f),
                                    make_float3(0.0f, 165.0f, 0.0f),
                                    make_float3(48.0f, 0.0f, -160.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  gis.push_back(createParallelogram(make_float3(240.0f, 0.0f, 272.0f),
                                    make_float3(0.0f, 165.0f, 0.0f),
                                    make_float3(-158.0f, 0.0f, -47.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  // Tall block
  gis.push_back(createParallelogram(make_float3(423.0f, 330.0f, 247.0f),
                                    make_float3(-158.0f, 0.0f, 49.0f),
                                    make_float3(49.0f, 0.0f, 159.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  gis.push_back(createParallelogram(make_float3(423.0f, 0.0f, 247.0f),
                                    make_float3(0.0f, 330.0f, 0.0f),
                                    make_float3(49.0f, 0.0f, 159.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  gis.push_back(createParallelogram(make_float3(472.0f, 0.0f, 406.0f),
                                    make_float3(0.0f, 330.0f, 0.0f),
                                    make_float3(-158.0f, 0.0f, 50.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  gis.push_back(createParallelogram(make_float3(314.0f, 0.0f, 456.0f),
                                    make_float3(0.0f, 330.0f, 0.0f),
                                    make_float3(-49.0f, 0.0f, -160.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);
  gis.push_back(createParallelogram(make_float3(265.0f, 0.0f, 296.0f),
                                    make_float3(0.0f, 330.0f, 0.0f),
                                    make_float3(158.0f, 0.0f, -49.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Kd"]->setFloat(white);

  // Parallelogram light, appearing in both the light buffer and geometry objects
  // make sure these two are identical in geometry, e.g. having the same normal vector
  gis.push_back(createParallelogram(make_float3(343.0f, 548.6f, 227.0f),
                                    make_float3(0.0f, 0.0f, 105.0f),
                                    make_float3(-130.0f, 0.0f, 0.0f),
                                    para_intersection,
                                    para_bounding_box,
                                    material));
  gis.back()["Le"]->setFloat(15.0f, 15.0f, 15.0f);  // to-do: power and/or radiance for light source

  GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
  geometry_group->setAcceleration(context->createAcceleration("Bvh", "Bvh"));
  context["top_object"]->set(geometry_group);
}

void PhotonMappingScene::createPhotonMap()
{
  int NUM_PHOTONS = pt_width * pt_height * max_num_deposits;  // to-do: temporarily
  int _photon_map_size = NUM_PHOTONS;  // to-do: temporarily
  SplitChoice _split_choice = LongestDim;

  PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( photon_record_buffer->map() );
  PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>( photon_map->map() );

  for( unsigned int i = 0; i < _photon_map_size; ++i ) {
    photon_map_data[i].power = make_float3( 0.0f );
  }

  // Push all valid photons to front of list
  unsigned int valid_photons = 0;
  PhotonRecord** temp_photons = new PhotonRecord*[NUM_PHOTONS];
  for( unsigned int i = 0; i < NUM_PHOTONS; ++i ) {
    if( fmaxf( photons_data[i].power ) > 0.0f ) {
      temp_photons[valid_photons++] = &photons_data[i];
    }
  }

  //std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
  //          << valid_photons<<"/"<<NUM_PHOTONS
  //          <<" ("<<valid_photons/static_cast<float>(NUM_PHOTONS)<<")\n";

  // Make sure we arent at most 1 less than power of 2
  valid_photons = valid_photons >= _photon_map_size ? _photon_map_size : valid_photons;

  float3 bbmin = make_float3(0.0f);
  float3 bbmax = make_float3(0.0f);
  if( _split_choice == LongestDim ) {
    bbmin = make_float3(  std::numeric_limits<float>::max() );
    bbmax = make_float3( -std::numeric_limits<float>::max() );
    // Compute the bounds of the photons
    for(unsigned int i = 0; i < valid_photons; ++i) {
      float3 position = (*temp_photons[i]).position;
      bbmin = fminf(bbmin, position);
      bbmax = fmaxf(bbmax, position);
    }
  }

  // Now build KD tree
  buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, _split_choice, bbmin, bbmax );

  delete[] temp_photons;
  photon_map->unmap();
  photon_record_buffer->unmap();
}
