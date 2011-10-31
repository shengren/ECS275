
#include "BasicMaterial.h"

#include <vector>

#include "HitRecord.h"
#include "Point.h"
#include "Primitive.h"
#include "Ray.h"
#include "RenderContext.h"
#include "Scene.h"
#include "Vector.h"
#include "Math.h"

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
  const Scene* scene = context.getScene();
  const vector<Primitive*>& arealights = scene->getAreaLights();
  Point hitpos = ray.origin() + ray.direction() * hit.minT();
  Vector normal;
  hit.getPrimitive()->normal(normal, context, hitpos, ray, hit);
  double costheta = Dot(normal, ray.direction());
  if (costheta > 0)
    normal = -normal;
  const Object* world = scene->getObject();

  //Color light = scene->getAmbient() * 0.5;  // to-do: hardcoded
  Color light(0.0, 0.0, 0.0);

  // direct illumination - trace shadow rays to all area lights
  Primitive*const* begin = &arealights[0];
  Primitive*const* end = &arealights[0] + arealights.size();
  while (begin != end) {
    Color light_color;
    vector<Vector> light_directions;  // not only the direction but also the distance
    (*begin++)->getSamples(light_color, light_directions, context, hitpos);

    int num_pass = 0;
    for (int i = 0; i < light_directions.size(); ++i) {
      Vector dir = light_directions[i];
      double len = dir.length();
      dir.normalize();
      double cosphi = Dot(normal, dir);
      if (cosphi > 0) {
        HitRecord shadowhit(len);
        Ray shadowray(hitpos, dir);
        world->intersect(shadowhit, context, shadowray);
        if (!shadowhit.getPrimitive())
          ++num_pass;
      }
    }
    light += light_color * ((double)num_pass / light_directions.size());
  }
  result = light * color;
}
