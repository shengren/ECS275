
#ifndef Sphere_h
#define Sphere_h

#include "Primitive.h"
#include "Point.h"

class Ray;

class Sphere : public Primitive {
 public:
  Sphere(Material* material, const Point& center, double radius);
  Sphere(Material* material, const Point& center, double radius,
         Vector direction, double speed);
  virtual ~Sphere();

  virtual void getBounds(BoundingBox& bbox) const;
  virtual void intersect(HitRecord& hit, const RenderContext& context, const Ray& ray) const;
  virtual void normal(Vector& normal, const RenderContext& context,
                      const Point& hitpos, const Ray& ray, const HitRecord& hit) const;
  virtual void move(double dt);
  virtual void getSamples(std::vector<Vector>& rays,
                          const RenderContext& context,
                          const Point& hitpos) const {}

 protected:
  Point initial_center;
  Point center;
  double radius;
  double inv_radius;
  Vector direction;
  double speed;

};

#endif
