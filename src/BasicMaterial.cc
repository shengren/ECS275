
#include "BasicMaterial.h"

#include <vector>
#include <cfloat>
#include <cmath>

#include "HitRecord.h"
#include "Point.h"
#include "Primitive.h"
#include "Ray.h"
#include "RenderContext.h"
#include "Scene.h"
#include "Vector.h"
#include "Math.h"
#include "ConstantBackground.h"  // for getting background color

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
  if (depth >= scene->getMaxRayDepth())
    return;

  Point hitpos = ray.origin() + ray.direction() * hit.minT();
  Vector normal;
  hit.getPrimitive()->normal(normal, context, hitpos, ray, hit);
  double costheta = Dot(normal, ray.direction());
  if (costheta > 0)
    normal = -normal;
  const Object* world = scene->getObject();

  //Color light = scene->getAmbient() * 0.5;  // to-do: hardcoded

  // direct illumination - trace shadow rays to all area lights
  Color direct_light(0.0, 0.0, 0.0);
  const vector<Primitive*>& arealights = scene->getAreaLights();
  Primitive*const* begin = &arealights[0];
  Primitive*const* end = &arealights[0] + arealights.size();
  while (begin != end) {
    Color light_color;
    vector<Vector> light_paths;  // not only the direction but also the distance
    (*begin++)->getSamples(light_color, light_paths, context, hitpos);
    int num_pass = 0;
    for (int i = 0; i < light_paths.size(); ++i) {
      double cosphi = Dot(normal, light_paths[i]);
      if (cosphi > 0) {
        HitRecord shadowhit(light_paths[i].length());
        Vector dir = light_paths[i];
        dir.normalize();
        Ray shadowray(hitpos, dir);
        world->intersect(shadowhit, context, shadowray);
        if (!shadowhit.getPrimitive())  // hit nothing
          ++num_pass;
      }
    }
    direct_light += light_color * ((double)num_pass / light_paths.size());
  }
  result = direct_light * color;

  // indirect illumination - path tracing
  // create the coordinate system around the hitpos based on its normal
  Vector u;
  if (abs(normal.x() - 1.0) < 1e-12 &&
      abs(normal.y() - 0.0) < 1e-12 &&
      abs(normal.z() - 0.0) < 1e-12) {
    u = Cross(normal, Vector(0.0, 1.0, 0.0));
  } else {
    u = Cross(normal, Vector(1.0, 0.0, 0.0));
  }
  u.normalize();
  Vector v = Cross(u, normal);
  v.normalize();
  // naive sampling on unit hemisphere
  double phi = 2.0 * M_PI * context.generateRandomNumber();
  double r = context.generateRandomNumber();  // unit hemisphere radius = 1.0
  Point sp = hitpos + u * (r * cos(phi)) + v * (r * sin(phi)) +
             normal * sqrt(1.0 - r * r);
  Vector dir = sp - hitpos;
  dir.normalize();  // to-do: necessary?
  Ray next_ray(hitpos, dir);
  HitRecord next_hit(DBL_MAX);
  world->intersect(next_hit, context, next_ray);
  Color c(0.0, 0.0, 0.0);
  if (next_hit.getPrimitive()) {
    Color atten;
    next_hit.getMaterial()->shade(c,
                                  context,
                                  next_ray,
                                  next_hit,
                                  atten,
                                  depth + 1);
  } else {
    scene->getBackground()->getBackgroundColor(c, context, next_ray);
  }
  double cosomega = Dot(dir, normal);
  double BRDF = 1.0 * cosomega;  // to-do: hardcoded
  Color indirect_light = c * (BRDF * cosomega);
  result += indirect_light * color;
}
