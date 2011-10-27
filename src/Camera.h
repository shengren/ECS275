
#ifndef Camera_h
#define Camera_h

#include <vector>

class Ray;
class RenderContext;
class Point2D;

class Camera {
 public:
  Camera();
  virtual ~Camera();

  virtual void preprocess(double aspect_ratio) = 0;
  virtual void makeRay(Ray& ray, const RenderContext& context, double x, double y) const = 0;
  virtual void makeRays(std::vector<Ray>& rays, const RenderContext& context,
                        double x, double y) const = 0;
  virtual void makeRays(std::vector<Ray>& rays, const RenderContext& context,
                        std::vector<Point2D> subpixels) const = 0;

 private:
  Camera(const Camera&);
  Camera& operator=(const Camera&);
};

#endif

