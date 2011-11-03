#include "Polygon.h"

#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>

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
}

Polygon::Polygon(Material* material, bool is_luminous, int sf,
                 const std::vector<Point>& point_list,
                 Vector direction, double speed)
    : Primitive(material, is_luminous, sf),
      point_list(point_list),
      direction(direction),
      speed(speed) {
  assert(point_list.size() >= 3);
}

Polygon::~Polygon() {
}

void Polygon::preprocess() {
  // normal
  Vector v1 = point_list[2] - point_list[1];
  Vector v2 = point_list[0] - point_list[1];
  n = Cross(v1, v2);
  n.normalize();
  // area
  Point o(0.0, 0.0, 0.0);
  a = 0.0;
  int num = point_list.size();
  for (int i = 0; i < num; ++i) {
    Vector t = Cross(point_list[i] - o, point_list[(i + 1) % num] - o);
    double ta = t.length() * 0.5;
    if (Dot(n, t) < 0.0) ta = -ta;
    a += ta;
  }
  a = abs(a);
  assert(a > 0.0);
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

void Polygon::getSamples(std::vector<Vector>& rays,
                         const RenderContext& context,
                         const Point& hitpos) const {
  rays.clear();

  // stratified sampling on parallelograms(including rectangles and squares)
  if (point_list.size() == 4) {
    Vector u = point_list[0] - point_list[1];
    Vector v = point_list[2] - point_list[1];
    if (((point_list[1] + (u + v)) - point_list[3]).length2() < 1e-12) {  // check if it is a parallelogram
      Vector du = u / (double)sf;
      Vector dv = v / (double)sf;
      for (int i = 0; i < sf; ++i)
        for (int j = 0; j < sf; ++j) {
          Point sp = point_list[1] +
                     du * (i + context.generateRandomNumber()) +
                     dv * (j + context.generateRandomNumber());
          rays.push_back(sp - hitpos);
        }
      return;
    }
  }

  // to-do: now no implementation for irregular polygons
  /*
  int num_samples = 128;
  for (int i = 0; i < num_samples; ++i) {
    double rest = 1.0;
    Point sp(0.0, 0.0, 0.0);
    for (int i = 0; i < point_list.size() - 1; ++i) {  // except the last point in the list
      double w = context.generateRandomNumber() * rest;
      sp += point_list[i] * w;
      rest -= w;
    }
    sp += point_list[point_list.size() - 1] * rest;  // sum of weights = 1
    rays.push_back(sp - hitpos);
  }
  */
}

double Polygon::getArea() const {
  return a;
}
