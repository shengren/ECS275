
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
    if (psfreq < 0)
      psfreq = 0;
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

  void setTimeSamplingFrequency(const int val) {
    tsfreq = val;
    if (tsfreq < 0)
      tsfreq = 0;
    if (tsfreq > 32)
      tsfreq = 32;
  }
  const int getTimeSamplingFrequency() {
    return tsfreq;
  }

  void setSamplingFrequency(const int val) {
    sfreq = val;
    if (sfreq < 0)
      sfreq = 0;
    if (sfreq > 32)
      sfreq = 32;
  }
  const int getSamplingFrequency() {
    return sfreq;
  }

  void setShutter(const double val) {
    shutter = val;
    if (shutter < 0.0)
      shutter = 0.0;
  }
  const double getShutter() const {
    return shutter;
  }

  void preprocess();
  void render();
  void renderFast();
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
  std::vector<double> sampleOnTime(const RenderContext& context);

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
  int tsfreq;
  int sfreq;  // used in permutation based distributed ray tracing
  double shutter;  // to-do: a camera parameter?
  //std::vector<std::vector<Color> > buffer;

};

struct Point2D {
  double x;
  double y;
  Point2D() : x(0.0), y(0.0) {}
  Point2D(double x, double y) : x(x), y(y) {}
};

#endif
