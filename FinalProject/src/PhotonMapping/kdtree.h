#ifndef KDTREE_H_
#define KDTREE_H_

#include <iostream>

#include <optixu/optixu_math_namespace.h>

#include "select.h"
#include "structs.h"

using namespace optix;

// from progressivePhotonMap/ppm.h
#define PPM_X (1 << 0)
#define PPM_Y (1 << 1)
#define PPM_Z (1 << 2)
#define PPM_LEAF (1 << 3)
#define PPM_NULL (1 << 4)

// modified from progressivePhotonMap/ppm.cpp

enum SplitChoice {
  RoundRobin,
  HighestVariance,
  LongestDim
};

inline RT_HOSTDEVICE int max_component(float3 a)
{
  if(a.x > a.y) {
    if(a.x > a.z) {
      return 0;
    } else {
      return 2;
    }
  } else {
    if(a.y > a.z) {
      return 1;
    } else {
      return 2;
    }
  }
}

void buildKDTree( PhotonRecord** photons, int start, int end, int depth, PhotonRecord* kd_tree, int current_root,
                  SplitChoice split_choice, float3 bbmin, float3 bbmax)
{
  // If we have zero photons, this is a NULL node
  if( end - start == 0 ) {
    kd_tree[current_root].axis = PPM_NULL;
    kd_tree[current_root].power = make_float3( 0.0f );
    return;
  }

  // If we have a single photon
  if( end - start == 1 ) {
    photons[start]->axis = PPM_LEAF;
    kd_tree[current_root] = *(photons[start]);
    return;
  }

  // Choose axis to split on
  int axis;
  switch(split_choice) {
  case RoundRobin:
    {
      axis = depth%3;
    }
    break;
  case HighestVariance:
    {
      float3 mean  = make_float3( 0.0f ); 
      float3 diff2 = make_float3( 0.0f );
      for(int i = start; i < end; ++i) {
        float3 x     = photons[i]->position;
        float3 delta = x - mean;
        float3 n_inv = make_float3( 1.0f / ( static_cast<float>( i - start ) + 1.0f ) );
        mean = mean + delta * n_inv;
        diff2 += delta*( x - mean );
      }
      float3 n_inv = make_float3( 1.0f / ( static_cast<float>(end-start) - 1.0f ) );
      float3 variance = diff2 * n_inv;
      axis = max_component(variance);
    }
    break;
  case LongestDim:
    {
      float3 diag = bbmax-bbmin;
      axis = max_component(diag);
    }
    break;
  default:
    axis = -1;
    std::cerr << "Unknown SplitChoice " << split_choice << " at "<<__FILE__<<":"<<__LINE__<<"\n";
    exit(2);
    break;
  }

  int median = (start+end) / 2;
  PhotonRecord** start_addr = &(photons[start]);
#if 0
  switch( axis ) {
  case 0:
    std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpX );
    photons[median]->axis = PPM_X;
    break;
  case 1:
    std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpY );
    photons[median]->axis = PPM_Y;
    break;
  case 2:
    std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpZ );
    photons[median]->axis = PPM_Z;
    break;
  }
#else
  switch( axis ) {
  case 0:
    select<PhotonRecord*, 0>( start_addr, 0, end-start-1, median-start );
    photons[median]->axis = PPM_X;
    break;
  case 1:
    select<PhotonRecord*, 1>( start_addr, 0, end-start-1, median-start );
    photons[median]->axis = PPM_Y;
    break;
  case 2:
    select<PhotonRecord*, 2>( start_addr, 0, end-start-1, median-start );
    photons[median]->axis = PPM_Z;
    break;
  }
#endif
  float3 rightMin = bbmin;
  float3 leftMax  = bbmax;
  if(split_choice == LongestDim) {
    float3 midPoint = (*photons[median]).position;
    switch( axis ) {
      case 0:
        rightMin.x = midPoint.x;
        leftMax.x  = midPoint.x;
        break;
      case 1:
        rightMin.y = midPoint.y;
        leftMax.y  = midPoint.y;
        break;
      case 2:
        rightMin.z = midPoint.z;
        leftMax.z  = midPoint.z;
        break;
    }
  }

  kd_tree[current_root] = *(photons[median]);
  buildKDTree( photons, start, median, depth+1, kd_tree, 2*current_root+1, split_choice, bbmin,  leftMax );
  buildKDTree( photons, median+1, end, depth+1, kd_tree, 2*current_root+2, split_choice, rightMin, bbmax );
}

#endif  // KDTREE_H_
