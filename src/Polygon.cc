#include "Polygon.h"

#include <vector>
#include <cassert>
#include <cmath>

#include "Ray.h"
#include "HitRecord.h"
#include "RenderContext.h"

using namespace std;

Polygon::Polygon(Material* material, const vector<Point>& point_list,
                 Vector direction, double speed)
    : Primitive(material),
      point_list(point_list),
      direction(direction),
      speed(speed) {
  assert(point_list.size() >= 3);
  Vector v1 = point_list[2] - point_list[1];
  Vector v2 = point_list[0] - point_list[1];
  n = Cross(v1, v2);
  n.normalize();
}

Polygon::Polygon(Material* material, bool is_luminous,
                 const std::vector<Point>& point_list,
                 Vector direction, double speed)
    : Primitive(material, is_luminous),
      point_list(point_list),
      direction(direction),
      speed(speed) {
  assert(point_list.size() >= 3);
  Vector v1 = point_list[2] - point_list[1];
  Vector v2 = point_list[0] - point_list[1];
  n = Cross(v1, v2);
  n.normalize();
}

Polygon::~Polygon() {
}

void Polygon::getBounds(BoundingBox& bbox) const {
}

void Polygon::intersect(HitRecord& hit, const RenderContext& context,
                        const Ray& ray) const {
  if (Dot(n, ray.direction()) > 0.0)
    return;
  double t = Dot(point_list[0] - ray.origin(), n) / Dot(ray.direction(), n);
  if (t < 0.0)
    return;
  // get the hit poing on the plane containing the polygon
  Point hitpos = ray.origin() + ray.direction() * t;
  bool inside = true;
  for (int i = 0; i < point_list.size(); ++i) {
    Vector v1 = point_list[(i + 1) % point_list.size()] - point_list[i];
    Vector v2 = hitpos - point_list[i];
    if (abs(v2.length2()) < 1e-12)  // ==0, on vertex
      break;
    Vector v = Cross(v1, v2);
    if (abs(v.length2()) < 1e-12)  // ==0, on edge
      break;
    if (Dot(v, n) < 0.0) {
      inside = false;
      break;
    }
  }
  if (inside)
    hit.hit(t, this, matl);
}

void Polygon::normal(Vector& normal, const RenderContext& context,
                     const Point& hitpos, const Ray& ray,
                     const HitRecord& hit) const {
  normal = n;
}

void Polygon::move(double dt) {
  for (int i = 0; i < point_list.size(); ++i)
    point_list[i] += direction * speed * dt;
}

void Polygon::getSamples(Color& color, std::vector<Vector>& directions,
                         const RenderContext& context,
                         const Point& hitpos) const {
  color = matl->getColor();
  // to-do: stratified?
  directions.clear();
  int num_samples = 128;
  for (int i = 0; i < num_samples; ++i) {
    double rest = 1.0;
    Point sp(0.0, 0.0, 0.0);
    for (int i = 0; i < point_list.size(); ++i) {
      if (i < point_list.size() - 1) {
        double r = context.generateRandomNumber() * rest;
        sp += point_list[i] * r;
        rest -= r;
      } else {
        sp += point_list[i] * rest;  // sum of weights = 1
      }
    }
    Vector dir = sp - hitpos;
    directions.push_back(dir);
  }
}
