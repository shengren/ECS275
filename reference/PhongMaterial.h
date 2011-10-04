#ifndef PhongMaterial_h
#define PhongMaterial_h

#include "Material.h"
#include "Color.h"

class PhongMaterial : public Material {
  public:
    PhongMaterial(const Color& color, 
                  double Ka, double Kd, double Ks, double p, 
                  double Kr);
    virtual ~PhongMaterial();

    virtual void shade(Color& result, 
                       const RenderContext& context, 
                       const Ray& ray, 
                       const HitRecord& hit,
                       const Color& atten,
                       int depth = 1) const;

  private:
    PhongMaterial(const PhongMaterial&);
    PhongMaterial& operator=(const PhongMaterial&);

    Color color;
    double Ka;
    double Kd;
    double Ks;
    double p;
    double Kr;
};

#endif
