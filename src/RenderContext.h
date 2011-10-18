
#ifndef RenderContext_h
#define RenderContext_h

#include <cstdlib>
#include <ctime>

#include "Scene.h"

class RenderContext {
 public:
  RenderContext(Scene *scene)
      : scene(scene) {
    srand48((unsigned int)time(NULL));
  }
  ~RenderContext() {}

  const Scene* getScene() const {
    return scene;
  }

  const double generateRandomNumber() const {
    return drand48();
  }

 private:
  const Scene *scene;

};

#endif
