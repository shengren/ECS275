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
  Polygon(Material* material, bool is_luminous,
          const std::vector<Point>& point_list,
          Vector direction, double speed);
  virtual ~Polygon();

  virtual void getBounds(BoundingBox& bbox) const;
  virtual void intersect(HitRecord& hit, const RenderContext& context, const Ray& ray) const;
  virtual void normal(Vector& normal, const RenderContext& context,
                      const Point& hitpos, const Ray& ray, const HitRecord& hit) const;
  virtual void move(double dt);
  virtual void getSamples(Color& color,
                          std::vector<Vector>& paths,  // from hitpos to samples w/o normalization
                          const RenderContext& context,
                          const Point& hitpos) const;

 private:
  std::vector<Point> point_list;
  Vector direction;
  double speed;
  Vector n;

};

#endif  // Polygon_h
