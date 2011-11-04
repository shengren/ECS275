#ifndef Polygon_h
#define Polygon_h

#include "Primitive.h"
#include "Vector.h"

#include <vector>

class Ray;

class Polygon : public Primitive {
 public:
  Polygon(Material* material, const std::vector<Point>& point_list,
          Vector direction, double speed);
  Polygon(Material* material, bool is_luminous, int sf,
          const std::vector<Point>& point_list,
          Vector direction, double speed);  // to-do: may assign default values for direction and speed
  virtual ~Polygon();

  virtual void preprocess();  // compute normal and area. to-do: call matl->preprocess?
  virtual void getBounds(BoundingBox& bbox) const;
  virtual void intersect(HitRecord& hit, const RenderContext& context, const Ray& ray) const;
  virtual void normal(Vector& normal, const RenderContext& context,
                      const Point& hitpos, const Ray& ray, const HitRecord& hit) const;
  virtual void move(double dt);
  virtual void getSamples(std::vector<Vector>& rays,  // from hitpos to samples w/o normalization
                          const RenderContext& context,
                          const Point& hitpos) const;
  virtual void getSample(Vector& ray,
                         const RenderContext& context,
                         const Point& hitpos) const;
  virtual double getArea() const;

 private:
  Polygon(const Polygon&);
  Polygon& operator=(const Polygon&);

  std::vector<Point> point_list;
  Vector direction;
  double speed;
  Vector n;
  double a;

};

#endif  // Polygon_h
