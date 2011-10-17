#ifndef ThinLensCamera_h
#define ThinLensCamera_h

#include <vector>

#include "Camera.h"
#include "Point.h"
#include "Vector.h"

class ThinLensCamera : public Camera {
 public:
  ThinLensCamera(const Point& _center,
                 const Point& _shoot_at,
                 const Vector& _up,
                 double _hfov,
                 double _aperture,
                 double _focal_dist);
  virtual ~ThinLensCamera();

  virtual void preprocess(double aspect_ratio);
  virtual void makeRays(std::vector<Ray>& rays,
                        const RenderContext& context,
                        double x,
                        double y) const;
  virtual void makeRay(Ray& ray,
                       const RenderContext& context,
                       double x,
                       double y) const;

 private:
  ThinLensCamera(const ThinLensCamera&);
  ThinLensCamera& operator=(const ThinLensCamera&);

  Point center;
  Point shoot_at;
  Vector up;
  double hfov;
  double aperture;
  double focal_dist;
  Vector u;
  Vector v;
  Vector shoot_dir;
  Vector lens_u;
  Vector lens_v;

};

#endif  // ThinLensCamera_h
