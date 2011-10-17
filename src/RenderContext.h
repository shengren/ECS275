
#ifndef RenderContext_h
#define RenderContext_h

#include <cstdlib>
#include <ctime>

class Scene;

class RenderContext {
 public:
  RenderContext(const Scene* scene)
      : scene(scene),
        antialiasing(false),
        psresolution(4),
        lsresolution(0) {
    srand48((unsigned int)time(NULL));
  }
  ~RenderContext() {}

  const Scene* getScene() const {
    return scene;
  }

  void setAntiAliasing(const bool val) {
    antialiasing = val;
  }
  const bool hasAntiAliasing() const {
    return antialiasing;
  }

  void setPixelSamplingResolution(const int val) {
    psresolution = val;
    if (psresolution < 1)
      psresolution = 1;
    if (psresolution > 8)
      psresolution = 8;
  }
  const int getPixelSamplingResolution() const {
    return psresolution;
  }

  void setLensSamplingResolution(const int val) {
    lsresolution = val;
    if (lsresolution < 0)
      lsresolution = 0;
    if (lsresolution > 8)
      lsresolution = 16;
  }
  const int getLensSamplingResolution() const {
    return lsresolution;
  }

  const double generateRandomNumber() const {
    return drand48();
  }

 private:
  const Scene* scene;
  bool antialiasing;
  int psresolution;
  int lsresolution;

};

#endif
