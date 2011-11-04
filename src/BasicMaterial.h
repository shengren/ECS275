
#ifndef BasicMaterial_h
#define BasicMaterial_h

#include "Material.h"
#include "Color.h"
#include "Vector.h"

class BasicMaterial : public Material {
 public:
  BasicMaterial(const Color& color,
                const bool is_luminous,
                const bool is_reflective,
                const double Kd,  // to-do: default values
                const double Ks,
                const double p);
  virtual ~BasicMaterial();

  virtual void shade(Color& result,
                     const RenderContext& context,
                     const Ray& ray,
                     const HitRecord& hit,
                     const Color& atten,
                     int depth) const;
  virtual Color getColor() const {
    return color;
  }

 private:
  BasicMaterial(const BasicMaterial&);
  BasicMaterial& operator=(const BasicMaterial&);

  Vector getPerfectSpecularDirection(Vector v, Vector n) const;
  double modifiedPhongBRDF(Vector in, Vector n, Vector out) const;
  Vector uniformSamplingOfHemisphere(const Vector n,
                                     const RenderContext& context) const;
  Color directIlluminate(const RenderContext& context,
                         const Ray& ray,
                         const HitRecord& hit) const;
  Color indirectIlluminate(const RenderContext& context,
                           const Ray& ray,
                           const HitRecord& hit,
                           const int depth) const;

  Color color;
  bool is_luminous;
  bool is_reflective;
  double Kd;  // these three are for modified Phong BRDF
  double Ks;
  double p;
};

#endif  // BasicMaterial_h
