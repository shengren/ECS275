#include "ThinLensCamera.h"

#include <cmath>
#include <vector>
#include <cstdio>
#include <algorithm>

#include "Ray.h"
#include "Math.h"
#include "RenderContext.h"

using namespace std;

ThinLensCamera::ThinLensCamera(const Point& center,
                               const Point& shoot_at,
                               const Vector& up,
                               double hfov,
                               double aperture,
                               double focal_dist)
    : center(center),
      shoot_at(shoot_at),
      up(up),
      hfov(hfov),
      aperture(aperture),
      focal_dist(focal_dist) {
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

  Point target = center + shoot_dir * focal_dist + u * x + v * y;
  Vector direction = target - center;
  direction.normalize();

  // sampling
  const int freq = (context.getScene())->getLensSamplingFrequency();
  const double d_a = 2.0 * M_PI / freq;
  const double d_rr = aperture * aperture / freq;
  if (freq > 0 && aperture > 0.0) {
    double s_a = 0.0;
    for (int i = 0; i < freq; ++i) {
      double s_rr = 0.0;
      for (int j = 0; j < freq; ++j) {
        double a = s_a + d_a * context.generateRandomNumber();
        double r = sqrt(s_rr + d_rr * context.generateRandomNumber());
        Point sample = center + lens_u * r * cos(a) + lens_v * r * sin(a);
        Vector direction = target - sample;
        direction.normalize();
        rays.push_back(Ray(sample, direction));
        s_rr += aperture * aperture / freq;
      }
      s_a += 2.0 * M_PI / freq;
    }
  }

  if (rays.empty())  // depth of field is disabled
    rays.push_back(Ray(center, direction));
}

void ThinLensCamera::makeRays(vector<Ray>& rays,
                              const RenderContext& context,
                              const vector<Point2D> subpixels) const {
  rays.clear();

  vector<Point> samples;
  // sampling
  const int freq = (context.getScene())->getLensSamplingFrequency();
  const double d_a = 2.0 * M_PI / freq;
  const double d_rr = aperture * aperture / freq;
  if (freq > 0 && aperture > 0.0) {
    double s_a = 0.0;
    for (int i = 0; i < freq; ++i) {
      double s_rr = 0.0;
      for (int j = 0; j < freq; ++j) {
        double a = s_a + d_a * context.generateRandomNumber();
        double r = sqrt(s_rr + d_rr * context.generateRandomNumber());
        Point s = center + lens_u * r * cos(a) + lens_v * r * sin(a);
        samples.push_back(s);
        s_rr += aperture * aperture / freq;
      }
      s_a += 2.0 * M_PI / freq;
    }
  }

  if (samples.empty())  // depth of field is disabled
    samples.push_back(center);

  if (samples.size() == 1 && subpixels.size() > 1) {  // expand to match subpixels by copying
    while (samples.size() < subpixels.size())
      samples.push_back(samples[0]);
  } else {
    random_shuffle(samples.begin(), samples.end());
  }

  for (int i = 0; i < samples.size(); ++i) {
    Point target = center + shoot_dir * focal_dist +
                   u * subpixels[i].x + v * subpixels[i].y;
    Vector direction = target - samples[i];
    direction.normalize();
    rays.push_back(Ray(samples[i], direction));
  }
}

// deprecated
void ThinLensCamera::makeRay(Ray& ray,
                             const RenderContext& context,
                             double x,
                             double y) const {
  Vector direction = shoot_dir + u * x + v * y;
  direction.normalize();
  ray = Ray(center, direction);
}
