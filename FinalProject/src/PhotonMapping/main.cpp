#include <iostream>
#include <cstdio>
#include <string>
#include <cstdlib>

#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <GLUTDisplay.h>

#include "photon_mapping_scene.h"

void printUsageAndExit(const std::string& argv0, bool do_exit = true) {
  std::cerr
    << "Usage : " << argv0 << " [options]" << std::endl
    << "App options:" << std::endl
    << "  -h | --help    Print this usage message" << std::endl
    << std::endl;

  GLUTDisplay::printUsage();

  if (do_exit)
    exit(1);
}

int main(int argc, char** argv) {
  GLUTDisplay::init(argc, argv);  // to-do: investigate

  // parse command line parameters
  //unsigned int width = 512u;
  //unsigned int height = 512u;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      printUsageAndExit(argv[0]);
    } else {
      std::cerr << "Unknown options: '" << arg << "'" << std::endl;
      printUsageAndExit(argv[0]);
    }
  }

  if (!GLUTDisplay::isBenchmark())
    printUsageAndExit(argv[0], false);  // only print not exit

  try {
    PhotonMappingScene scene;
    //scene.setDisplayResolution(width, height);

    //GLUTDisplay::setUseSRGB(true);  // to-do: standard RGB?
    GLUTDisplay::setProgressiveDrawingTimeout(300.0f);  // to-do: ususally it is set by input. Internally, it is set to 10s by default.
    GLUTDisplay::run("PhotonMappingScene", &scene, GLUTDisplay::CDProgressive);
  } catch (optix::Exception &e) {
    sutilReportError(e.getErrorString().c_str());
    exit(1);
  }

  return 0;
}
