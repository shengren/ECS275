
#include "BasicMaterial.h"

#include <vector>
#include <cfloat>
#include <cmath>
#include <cstdio>

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

  if (is_luminous) {  // for objects can emit, workaround
    result = color;
    return;
  }

  result = directIlluminate(context, ray, hit) * color;
}

Color BasicMaterial::directIlluminate(const RenderContext& context,
                                      const Ray& ray,
                                      const HitRecord& hit) const {
  const Scene* scene = context.getScene();
  const Object* world = scene->getObject();
  Point hitpos = ray.origin() + ray.direction() * hit.minT();
  Vector normal;
  hit.getPrimitive()->normal(normal, context, hitpos, ray, hit);
  if (Dot(normal, ray.direction()) > 0.0)
    normal = -normal;
  Color ret(0.0, 0.0, 0.0);

  // direct illumination - trace shadow rays to all area lights
  const vector<Primitive*>& arealights = scene->getAreaLights();
  for (int i = 0; i < arealights.size(); ++i) {
    const Primitive& light_source = *(arealights[i]);
    vector<Vector> light_rays;  // not only the direction but also the distance
    light_source.getSamples(light_rays, context, hitpos);
    int num_pass = 0;
    for (int i = 0; i < light_rays.size(); ++i) {
      if (Dot(normal, light_rays[i]) > 0) {  // visibility part I
        HitRecord shadowhit(light_rays[i].length());
        Vector dir = light_rays[i];
        dir.normalize();
        Ray shadowray(hitpos, dir);
        world->intersect(shadowhit, context, shadowray);
        if (!shadowhit.getPrimitive())  // hit nothing, visibility part II
          ++num_pass;
      }
    }
    ret += light_source.getColor() * ((double)num_pass / light_rays.size());
  }

  return ret;
}

Color BasicMaterial::indirectIlluminate(const RenderContext& context,
                                        const Ray& ray,
                                        const HitRecord& hit) const {
  return Color(0.0, 0.0, 0.0);
  /*
  // indirect illumination - path tracing
  // create the coordinate system around the hitpos based on its normal
  Vector u;
  if (abs(abs(normal.x()) - 1.0) < 1e-12 &&
      abs(normal.y()) < 1e-12 &&
      abs(normal.z()) < 1e-12) {
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
  */
}
