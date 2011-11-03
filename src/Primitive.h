
#ifndef Primitive_h
#define Primitive_h

#include <vector>
#include "Object.h"
#include "Material.h"

//class Material;
class Point;
class Vector;

class Primitive : public Object {
 public:
  Primitive(Material* matl);
  Primitive(Material* matl, bool is_luminous, int sf);  // to-do: may not be the best solution to support area lights
  virtual ~Primitive();

  virtual void preprocess();
  virtual void intersect(HitRecord& hit, const RenderContext& context, const Ray& ray) const = 0;
  virtual void normal(Vector& normal, const RenderContext& context,
                      const Point& hitpos, const Ray& ray, const HitRecord& hit) const = 0;
  virtual void computeUVW(Vector& uvw, const RenderContext& context,
                          const Ray& ray, const HitRecord& hit) const;
  virtual void move(double dt) = 0;
  virtual void getSamples(std::vector<Vector>& rays,  // w/o normalization
                          const RenderContext& context,
                          const Point& hitpos) const = 0;
  virtual Color getColor() const {
    return matl->getColor();
  }
  virtual bool isLuminous() const {
    return is_luminous;
  }
  virtual double getArea() const {
    // to-do:
    // used for sampling on light source area
    // not a pure virtual function, because this can avoid implementations
    // on primitives other than polygon temporarily
    return 0.0;
  }

 protected:
  Material* matl;
  bool is_luminous;
  int sf;

 private:
  Primitive(const Primitive&);
  Primitive& operator=(const Primitive&);
};


#endif

