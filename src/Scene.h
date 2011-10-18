
#ifndef Scene_h
#define Scene_h

#include "Color.h"
#include <string>
#include <vector>

class Background;
class Camera;
class Color;
class Image;
class Light;
class Object;
class RenderContext;
class Ray;
struct Point2D;

class Scene {
 public:
  Scene();
  virtual ~Scene();

  inline Object* getObject() const {
    return object;
  }
  void setObject(Object* obj) {
    object = obj;
  }

  inline Background* getBackground() const {
    return background;
  }
  void setBackground(Background* bg) {
    background = bg;
  }

  inline Camera* getCamera() const {
    return camera;
  }
  void setCamera(Camera* cam) {
    camera = cam;
  }

  inline Image* getImage() const {
    return image;
  }
  void setImage(Image* im) {
    image = im;
  }

  void addLight(Light* light) {
    lights.push_back(light);
  }
  const std::vector<Light*>& getLights() const {
    return lights;
  }

  Color getAmbient() const {
    return ambient;
  }
  void setAmbient(const Color& amb) {
    ambient = amb;
  }

  int getMaxRayDepth() const {
    return maxRayDepth;
  }
  void setMaxRayDepth(int rd) {
    maxRayDepth = rd;
  }
  double getMinAttenuation() const {
    return minAttenuation;
  }
  void setMinAttenuation(double atten) {
    minAttenuation = atten;
  }

  void setPixelSamplingFrequency(const int val) {
    psfreq = val;
    if (psfreq < 1)
      psfreq = 1;
    if (psfreq > 8)
      psfreq = 8;
  }
  const int getPixelSamplingFrequency() const {
    return psfreq;
  }

  void setLensSamplingFrequency(const int val) {
    lsfreq = val;
    if (lsfreq < 0)
      lsfreq = 0;
    if (lsfreq > 8)
      lsfreq = 8;
  }
  const int getLensSamplingFrequency() const {
    return lsfreq;
  }

  void preprocess();
  void render();
  double traceRay(Color& result, const RenderContext& context, const Ray& ray, const Color& attenuation, int depth) const;
  double traceRay(Color& result, const RenderContext& context, const Object* obj, const Ray& ray, const Color& attenuation, int depth) const;

 private:
  Scene(const Scene&);
  Scene& operator=(const Scene&);
  std::vector<Point2D> sampleInPixel(const int x,
                                     const int y,
                                     const int xres,
                                     const int yres,
                                     const RenderContext& context);

  Background* background;
  Camera* camera;
  Color ambient;
  Image* image;
  Object* object;
  std::vector<Light*> lights;
  int maxRayDepth;
  double minAttenuation;
  int psfreq;
  int lsfreq;

};

struct Point2D {
  double x;
  double y;
  Point2D() : x(0.0), y(0.0) {}
  Point2D(double x, double y) : x(x), y(y) {}
};

#endif
