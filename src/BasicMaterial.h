
#ifndef BasicMaterial_h
#define BasicMaterial_h

#include "Material.h"
#include "Color.h"

class BasicMaterial : public Material {
 public:
  BasicMaterial(const Color& color,
                const bool is_luminous,
                const bool is_reflective);
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

  Color color;
  bool is_luminous;
  bool is_reflective;

};

#endif  // BasicMaterial_h
