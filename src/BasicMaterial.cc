
#include "BasicMaterial.h"

using namespace std;

BasicMaterial::BasicMaterial(const Color& color,
                             const bool is_luminous,
                             const bool is_reflective)
  : color(color), is_luminous(is_luminous), is_reflective(is_reflective) {
}

BasicMaterial::~BasicMaterial() {
}

void BasicMaterial::shade(Color& result,
                          const RenderContext& context,
                          const Ray& ray,
                          const HitRecord& hit,
                          const Color& atten,
                          int depth) const {
}
