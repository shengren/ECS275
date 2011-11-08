
#include "Sphere.h"
#include "BoundingBox.h"
#include "HitRecord.h"
#include "Point.h"
#include "Ray.h"
#include "Vector.h"
#include <math.h>
#include <cassert>
#include <cstdio>

Sphere::Sphere(Material* material, const Point& center, double radius)
  : Primitive(material), initial_center(center), center(center), radius(radius)
{
  inv_radius = 1./radius;
}

Sphere::Sphere(Material* material, const Point& center, double radius,
               Vector direction, double speed)
    : Primitive(material),
      initial_center(center),
      center(center),
      radius(radius),
      direction(direction),
      speed(speed)
{
  inv_radius = 1./radius;
}

Sphere::Sphere(Material* material, bool is_luminous, int sf,
               const Point& center, double radius,
               Vector direction, double speed)
    : Primitive(material, is_luminous, sf),
      initial_center(center),
      center(center),
      radius(radius),
      direction(direction),
      speed(speed)
{
  inv_radius = 1./radius;
}

Sphere::~Sphere()
{
}

void Sphere::preprocess() {
  a = 4.0 * M_PI * radius * radius;
}

void Sphere::getBounds(BoundingBox& bbox) const
{
  Vector diag(radius, radius, radius);
  bbox.extend(center+diag);
  bbox.extend(center-diag);
}

void Sphere::intersect(HitRecord& hit, const RenderContext&, const Ray& ray) const
{
  Vector O(ray.origin()-center);
  const Vector& V(ray.direction());
  double b = Dot(O, V);
  double c = Dot(O, O)-radius*radius;
  double disc = b*b-c;
  if(disc > 0){
    double sdisc = sqrt(disc);
    double root1 = (-b - sdisc);
    if(!hit.hit(root1, this, matl)){
      double root2 = (-b + sdisc);
      hit.hit(root2, this, matl);
    }
  }
}

void Sphere::normal(Vector& normal, const RenderContext&, const Point& hitpos,
                    const Ray& ray, const HitRecord& hit) const
{
  normal = (hitpos-center)*inv_radius;
  double dist = normal.normalize();
}

void Sphere::move(double dt)
{
  center = initial_center + direction * speed * dt;
}

void Sphere::getSamples(std::vector<Vector>& rays,
                        const RenderContext& context,
                        const Point& hitpos) const {
  rays.clear();

  for (int i = 0; i < sf * sf; ++i) {
    double theta = M_PI * context.generateRandomNumber();
    double phi = 2.0 * M_PI * context.generateRandomNumber();
    Vector sdir(sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta));
    Point sp = center + sdir * radius;
    if (Dot(hitpos - center, sp - center) < 0.0)
      sp = center + (-sdir) * radius;
    rays.push_back(sp - hitpos);
  }
}

void Sphere::getSample(Vector& ray,
                       const RenderContext& context,
                       const Point& hitpos) const {
  double theta = M_PI * context.generateRandomNumber();
  double phi = 2.0 * M_PI * context.generateRandomNumber();
  Vector sdir(sin(theta) * cos(phi),
              sin(theta) * sin(phi),
              cos(theta));
  Point sp = center + sdir * radius;
  if (Dot(hitpos - center, sp - center) < 0.0)
    sp = center + (-sdir) * radius;
  ray = sp - hitpos;
}

double Sphere::getArea() const {
  return a * 0.5;  // because samples are always on the front side w.r.t. shadow rays
  //return a;
}
