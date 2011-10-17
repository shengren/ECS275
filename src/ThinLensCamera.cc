#include "ThinLensCamera.h"

#include <cmath>
#include <vector>
#include <cstdio>

#include "Ray.h"
#include "Math.h"
#include "RenderContext.h"

using namespace std;

ThinLensCamera::ThinLensCamera(const Point& _center,
                               const Point& _shoot_at,
                               const Vector& _up,
                               double _hfov,
                               double _aperture,
                               double _focal_dist)
    : center(_center),
      shoot_at(_shoot_at),
      up(_up),
      hfov(_hfov),
      aperture(_aperture),
      focal_dist(_focal_dist) {
}

ThinLensCamera::~ThinLensCamera() {
}

void ThinLensCamera::preprocess(double aspect_ratio) {
  shoot_dir = shoot_at - center;
  shoot_dir.normalize();
  u = Cross(shoot_dir, up);
  u.normalize();
  lens_u = u;
  v = Cross(u, shoot_dir);
  v.normalize();
  lens_v = v;
  double ulen = tan(hfov / 2.0 * M_PI / 180.0) * focal_dist;
  double vlen = ulen / aspect_ratio;
  u *= ulen;
  v *= vlen;
}

void ThinLensCamera::makeRays(vector<Ray>& rays,
                              const RenderContext& context,
                              double x,
                              double y) const {
  rays.clear();
  Vector offset = u * x + v * y;

  //Vector direction = shoot_dir + offset;
  Vector direction = (center + shoot_dir * focal_dist + u * x + v * y) - center;
  direction.normalize();
  rays.push_back(Ray(center, direction));

  // sampling
  const int res = context.getLensSamplingResolution();
  if (res > 0 && aperture > 0.0) {
    double s_a = 0.0;
    for (int i = 0; i < res; ++i) {
      double s_rr = 0.0;
      for (int j = 0; j < res; ++j) {
        double a = s_a + 2.0 * M_PI / res * context.generateRandomNumber();
        double r = sqrt(s_rr + aperture * aperture / res *
                        context.generateRandomNumber());

        Point sample = center + lens_u * r * cos(a) + lens_v * r * sin(a);
        Vector direction = (center + shoot_dir * focal_dist + u * x + v * y) - sample;
        direction.normalize();
        rays.push_back(Ray(sample, direction));

        s_rr += aperture * aperture / res;
      }
      s_a += 2.0 * M_PI / res;
    }
  }
}

void ThinLensCamera::makeRay(Ray& ray,
                             const RenderContext& context,
                             double x,
                             double y) const {
  Vector direction = shoot_dir + u * x + v * y;
  direction.normalize();
  ray = Ray(center, direction);
}
