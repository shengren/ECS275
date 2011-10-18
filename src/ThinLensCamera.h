#ifndef ThinLensCamera_h
#define ThinLensCamera_h

#include <vector>

#include "Camera.h"
#include "Point.h"
#include "Vector.h"

class ThinLensCamera : public Camera {
 public:
  ThinLensCamera(const Point& center,
                 const Point& shoot_at,
                 const Vector& up,
                 double hfov,
                 double aperture,
                 double focal_dist);
  virtual ~ThinLensCamera();

  virtual void preprocess(double aspect_ratio);
  virtual void makeRays(std::vector<Ray>& rays,
                        const RenderContext& context,
                        double x,
                        double y) const;
  virtual void makeRay(Ray& ray,
                       const RenderContext& context,
                       double x,
                       double y) const;  // deprecated

 private:
  ThinLensCamera(const ThinLensCamera&);
  ThinLensCamera& operator=(const ThinLensCamera&);

  Point center;
  Point shoot_at;
  Vector up;
  double hfov;
  double aperture;
  double focal_dist;
  Vector shoot_dir;
  Vector u;
  Vector v;
  Vector lens_u;
  Vector lens_v;

};

#endif  // ThinLensCamera_h
