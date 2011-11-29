
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <PPMLoader.h>
#include <optixu/optixu_math_namespace.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace optix;

//-----------------------------------------------------------------------------
//  
//  PPMLoader class definition
//
//-----------------------------------------------------------------------------

PPMLoader::PPMLoader( const std::string& filename, const bool vflip )
  : _nx( 0u ), _ny( 0u ), _max_val( 0u ), _raster( 0 ), _is_ascii(false)
{
  if ( filename.empty() ) return;
  
  std::string extension = filename.substr( filename.find_last_of( '.' ) );
  if (extension != ".ppm" ) {
    std::cerr << "PPMLoader( '" << filename << "' ) non-ppm file extension given '" << extension << "'" << std::endl;
    return;
  }

  // Open file
  try {
    std::ifstream file_in( filename.c_str(), std::ifstream::in | std::ifstream::binary );
    if ( !file_in ) {
      std::cerr << "PPMLoader( '" << filename << "' ) failed to open file."
                << std::endl;
      return;
    }

    // Check magic number to make sure we have an ascii or binary PPM
    std::string line, magic_number;
    getLine( file_in, line );
    std::istringstream iss1( line );
    iss1 >> magic_number;
    if ( magic_number != "P6" && magic_number != "P3") {
      std::cerr << "PPMLoader( '" << filename << "' ) unknown magic number: "
                << magic_number << ".  Only P3 and P6 supported." << std::endl;
      return;
    }
    if ( magic_number == "P3" ) {
      _is_ascii = true;
    }

    // width, height
    getLine( file_in, line );
    std::istringstream iss2( line );
    iss2 >> _nx >> _ny;

    // max channel value
    getLine( file_in, line );
    std::istringstream iss3( line );
    iss3 >> _max_val;

    _raster = new(std::nothrow)  unsigned char[ _nx*_ny*3 ];
    if(_is_ascii) {
      unsigned int num_elements = _nx*_ny*3;
      unsigned int count = 0;

      while(count < num_elements) {
        getLine( file_in, line );
        std::istringstream iss(line);

        while(iss.good()) {
          unsigned int c;
          iss >> c;
          _raster[count++] = static_cast<unsigned char>(c);
        }
      }

    } else {
      file_in.read( reinterpret_cast<char*>( _raster ), _nx*_ny*3 );
    }

    if(vflip) {
      unsigned char *_raster2 = new(std::nothrow)  unsigned char[ _nx*_ny*3 ];
      for(unsigned int y2=_ny-1, y=0; y<_ny; y2--, y++) {
        for(unsigned int x=0; x<_nx*3; x++)
          _raster2[y2*_nx*3+x] = _raster[y*_nx*3+x];
      }

      delete [] _raster;
      _raster = _raster2;
    }
  } catch ( ... ) {
    std::cerr << "PPMLoader( '" << filename << "' ) failed to load" << std::endl;
    _raster = 0;
  }
}


PPMLoader::~PPMLoader()
{
  if ( _raster ) delete[] _raster;
}


bool PPMLoader::failed()const
{
  return _raster == 0;
}


unsigned int PPMLoader::width()const
{
  return _nx;
}


unsigned int PPMLoader::height()const
{
  return _ny;
}


unsigned char* PPMLoader::raster()const
{
  return _raster;
}


void PPMLoader::getLine( std::ifstream& file_in, std::string& s )
{
  for (;;) {
    if ( !std::getline( file_in, s ) )
      return;
    std::string::size_type index = s.find_first_not_of( "\n\r\t " );
    if ( index != std::string::npos && s[index] != '#' )
      break;
  }
}

  
//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

optix::TextureSampler loadPPMTexture( optix::Context context,
                                    const std::string& filename,
                                    const float3& default_color )
{
  // Create tex sampler and populate with default values
  optix::TextureSampler sampler = context->createTextureSampler();
  sampler->setWrapMode( 0, RT_WRAP_REPEAT );
  sampler->setWrapMode( 1, RT_WRAP_REPEAT );
  sampler->setWrapMode( 2, RT_WRAP_REPEAT );
  sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
  sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
  sampler->setMaxAnisotropy( 1.0f );
  sampler->setMipLevelCount( 1u );
  sampler->setArraySize( 1u );

  // Read in PPM, set texture buffer to empty buffer if fails
  PPMLoader ppm( filename );
  if ( ppm.failed() ) {

    // Create buffer with single texel set to default_color
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, 1u, 1u );
    unsigned char* buffer_data = static_cast<unsigned char*>( buffer->map() );
    buffer_data[0] = (unsigned char)clamp((int)(default_color.x * 255.0f), 0, 255);
    buffer_data[1] = (unsigned char)clamp((int)(default_color.y * 255.0f), 0, 255);
    buffer_data[2] = (unsigned char)clamp((int)(default_color.z * 255.0f), 0, 255);
    buffer_data[3] = 255;
    buffer->unmap();

    sampler->setBuffer( 0u, 0u, buffer );
    sampler->setFilteringModes( RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE );

    return sampler;
  }

  const unsigned int nx = ppm.width();
  const unsigned int ny = ppm.height();

  // Create buffer and populate with PPM data
  optix::Buffer buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, nx, ny );
  unsigned char* buffer_data = static_cast<unsigned char*>( buffer->map() );

  for ( unsigned int i = 0; i < nx; ++i ) {
    for ( unsigned int j = 0; j < ny; ++j ) {

      unsigned int ppm_index = ( (ny-j-1)*nx + i )*3;
      unsigned int buf_index = ( (j     )*nx + i )*4;

      buffer_data[ buf_index + 0 ] = ppm.raster()[ ppm_index + 0 ];
      buffer_data[ buf_index + 1 ] = ppm.raster()[ ppm_index + 1 ];
      buffer_data[ buf_index + 2 ] = ppm.raster()[ ppm_index + 2 ];
      buffer_data[ buf_index + 3 ] = 255;
    }
  }

  buffer->unmap();

  sampler->setBuffer( 0u, 0u, buffer );
  sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

  return sampler;
}