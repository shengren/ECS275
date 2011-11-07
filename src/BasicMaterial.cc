
#include "BasicMaterial.h"

#include <vector>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cassert>

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
                             const bool is_reflective,
                             const double Kd,
                             const double Ks,
                             const double p)
  : color(color), is_luminous(is_luminous), is_reflective(is_reflective),
    Kd(Kd), Ks(Ks), p(p) {
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
  if (depth > scene->getMaxRayDepth() * 10)
    return;

  if (is_luminous) {  // for objects can emit, workaround
    result = color;
    return;
  }

  //Color direct = doDirectIlluminate(context, ray, hit);
  Color direct = doMultipleDirectIlluminate(context, ray, hit);
  Color indirect = doIndirectIlluminate(context, ray, hit, depth);
  //if (indirect.maxComponent() > 1.0)
  //  indirect.normalize();
  //indirect.truncate();
  result = direct + indirect;
  //result = direct;
  result *= color;
}

Color BasicMaterial::doDirectIlluminate(const RenderContext& context,
                                        const Ray& ray,
                                        const HitRecord& hit) const {
  const Scene* scene = context.getScene();
  const Object* world = scene->getObject();
  Point hitpos = ray.origin() + ray.direction() * hit.minT();
  Vector normal;
  hit.getPrimitive()->normal(normal, context, hitpos, ray, hit);
  if (Dot(normal, ray.direction()) > 0.0)  // to-do: necessary?
    normal = -normal;
  Color ret(0.0, 0.0, 0.0);

  // direct illumination - trace shadow rays to all area lights
  const vector<Primitive*>& arealights = scene->getAreaLights();
  for (int i = 0; i < arealights.size(); ++i) {  // to-do: randomly choose a light source
    const Primitive& light_source = *(arealights[i]);
    Vector light_ray;
    light_source.getSample(light_ray, context, hitpos);
    if (Dot(normal, light_ray) > 1e-10) {  // visibility part I
      HitRecord shadowhit(DBL_MAX);
      Vector dir = light_ray;
      dir.normalize();
      Ray shadowray(hitpos, dir);
      world->intersect(shadowhit, context, shadowray);
      assert(shadowhit.getPrimitive() != NULL);
      if (shadowhit.getPrimitive() == arealights[i]) {  // hit the light source, visibility part II
        double BRDF = getModifiedPhongBRDF(dir, normal, -ray.direction());
        Vector light_normal;
        light_source.normal(light_normal, context, Point(),
                            shadowray, shadowhit);  // to-do: shadow ray hit point is fake now
        double geom = getGeometry(normal, light_ray, light_normal);
        ret = light_source.getColor();  // to-do: not support multiple lights
        ret *= BRDF;
        ret *= geom * light_source.getArea();
      }
    }
  }

  return ret;
}

Color BasicMaterial::doIndirectIlluminate(const RenderContext& context,
                                          const Ray& ray,
                                          const HitRecord& hit,
                                          const int depth) const {
  const Scene* scene = context.getScene();
  const Object* world = scene->getObject();
  Point hitpos = ray.origin() + ray.direction() * hit.minT();
  Vector normal;
  hit.getPrimitive()->normal(normal, context, hitpos, ray, hit);
  if (Dot(normal, ray.direction()) > 0.0)  // to-do: necessary?
    normal = -normal;
  Color ret(0.0, 0.0, 0.0);

  // indirect illumination - path tracing
  double prr = color.maxComponent();
  if (depth > scene->getMaxRayDepth()) {
    if (context.generateRandomNumber() > prr)
      return ret;  // black
  } else {
    prr = 1.0;
  }

  Vector dir;
  if (is_reflective) {
    dir = getPerfectSpecularDirection(-ray.direction(), normal);
  } else {
    //dir = SampleOfHemisphereUniform(normal, context);
    dir = SampleOfHemisphereCosine(normal, context);
  }

  Ray recursive_ray(hitpos, dir);
  HitRecord recursive_hit(DBL_MAX);
  world->intersect(recursive_hit, context, recursive_ray);
  Color c(0.0, 0.0, 0.0);
  if (recursive_hit.getPrimitive() != NULL &&
      !(recursive_hit.getPrimitive()->isLuminous())) {  // to-do: return black if hit a light source?
    recursive_hit.getMaterial()->shade(c,
                                       context,
                                       recursive_ray,
                                       recursive_hit,
                                       Color(0.0, 0.0, 0.0),
                                       depth + 1);
    ret = c;  // pair to cosine sampling on hemisphere
  } else {
    scene->getBackground()->getBackgroundColor(c, context, recursive_ray);
    ret = c;
  }
  double BRDF = getModifiedPhongBRDF(dir, normal, -ray.direction());
  ret *= BRDF;

  if (is_reflective) {
    ret *= Dot(normal, dir);
  } else {
    //ret *= 2.0 * M_PI;  // pair to uniform hemisphere sampling
    ret *= M_PI;  // pair to cosine sampling on hemisphere
  }

  ret /= prr;  // for Russian Roulette

  return ret;
}

Vector BasicMaterial::getPerfectSpecularDirection(Vector v, Vector n) const {
  Vector s = n * (2.0 * Dot(v, n)) - v;
  s.normalize();
  return s;
}

double BasicMaterial::getModifiedPhongBRDF(Vector in, Vector n, Vector out) const {
  Vector s = getPerfectSpecularDirection(in, n);
  double cos = Dot(out, s);
  cos = (cos > 1.0) ? 1.0 : cos;
  return Kd + Ks * pow(cos, p);
}

double BasicMaterial::getGeometry(Vector ns, Vector sray, Vector nl) const {
  double dist = sray.normalize();
  return Dot(ns, sray) * Dot(nl, -sray) / (dist * dist);
}

Vector BasicMaterial::SampleOfHemisphereUniform(
    const Vector n,
    const RenderContext& context) const {
  // create the coordinate system around the hitpos based on its normal
  Vector u;
  if (abs(abs(n.x()) - 1.0) < 1e-10 &&
      abs(n.y()) < 1e-10 &&
      abs(n.z()) < 1e-10) {
    u = Cross(n, Vector(0.0, 1.0, 0.0));
  } else {
    u = Cross(n, Vector(1.0, 0.0, 0.0));
  }
  u.normalize();
  Vector v = Cross(u, n);
  v.normalize();
  // uniform sampling on unit hemisphere
  double phi = 2.0 * M_PI * context.generateRandomNumber();
  double theta = 0.5 * M_PI * context.generateRandomNumber();
  Point o(0.0, 0.0, 0.0);
  Point sp = o +
             u * (sin(theta) * cos(phi)) +
             v * (sin(theta) * sin(phi)) +
             n * cos(theta);
  Vector ret = sp - o;
  ret.normalize();
  return ret;
}

Vector BasicMaterial::SampleOfHemisphereCosine(const Vector n,
                                               const RenderContext& context) const {
  // create the coordinate system around the hitpos based on its normal
  Vector u;
  if (abs(abs(n.x()) - 1.0) < 1e-10 &&
      abs(n.y()) < 1e-10 &&
      abs(n.z()) < 1e-10) {
    u = Cross(n, Vector(0.0, 1.0, 0.0));
  } else {
    u = Cross(n, Vector(1.0, 0.0, 0.0));
  }
  u.normalize();
  Vector v = Cross(u, n);
  v.normalize();
  // cosine sampling on unit hemisphere
  double phi = 2.0 * M_PI * context.generateRandomNumber();
  double r2 = context.generateRandomNumber();
  Point o(0.0, 0.0, 0.0);
  Point sp = o +
             u * (sqrt(1.0 - r2) * cos(phi)) +
             v * (sqrt(1.0 - r2) * sin(phi)) +
             n * sqrt(r2);
  Vector ret = sp - o;
  ret.normalize();
  return ret;
}

Color BasicMaterial::doMultipleDirectIlluminate(const RenderContext& context,
                                                const Ray& ray,
                                                const HitRecord& hit) const {
  const Scene* scene = context.getScene();
  const Object* world = scene->getObject();
  Point hitpos = ray.origin() + ray.direction() * hit.minT();
  Vector normal;
  hit.getPrimitive()->normal(normal, context, hitpos, ray, hit);
  if (Dot(normal, ray.direction()) > 0.0)  // to-do: necessary?
    normal = -normal;
  Color ret(0.0, 0.0, 0.0);

  // direct illumination - trace shadow rays to all area lights
  const vector<Primitive*>& arealights = scene->getAreaLights();
  for (int i = 0; i < arealights.size(); ++i) {
    const Primitive& light_source = *(arealights[i]);
    vector<Vector> light_rays;  // not only the direction but also the distance
    light_source.getSamples(light_rays, context, hitpos);

    double ratio = 0.0;
    for (int j = 0; j < light_rays.size(); ++j) {
      if (Dot(normal, light_rays[j]) > 1e-10) {  // visibility part I
        HitRecord shadowhit(DBL_MAX);
        Vector dir = light_rays[j];
        dir.normalize();
        Ray shadowray(hitpos, dir);
        world->intersect(shadowhit, context, shadowray);
        assert(shadowhit.getPrimitive() != NULL);
        if (shadowhit.getPrimitive() == arealights[i]) {  // hit the light source, visibility part II
          double BRDF = getModifiedPhongBRDF(dir, normal, -ray.direction());
          Vector light_normal;
          light_source.normal(light_normal, context, Point(),
                              shadowray, shadowhit);  // to-do: shadow ray hit point is fake now
          double geom = getGeometry(normal, light_rays[j], light_normal);
          ratio += BRDF * geom;
          //ratio += 1.0;
        }
      }
    }
    ratio *= light_source.getArea() / (double)light_rays.size();
    //ratio /= (double)light_rays.size();

    ret = light_source.getColor() * ratio;  // to-do: not support multiple lights
  }

  return ret;
}

