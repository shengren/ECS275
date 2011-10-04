
#include "LambertianMaterial.h"
#include "HitRecord.h"
#include "Light.h"
#include "Point.h"
#include "Primitive.h"
#include "Ray.h"
#include "RenderContext.h"
#include "Scene.h"
#include "Vector.h"
#include "Math.h"
#include "ConstantBackground.h"  // for getting background color
#include <float.h>  // for using DBL_MAX
using namespace std;

LambertianMaterial::LambertianMaterial(const Color& color, float Kd, float Ka)
  :color(color), Kd(Kd), Ka(Ka)
{
}

LambertianMaterial::~LambertianMaterial()
{
}

/*
void LambertianMaterial::shade(Color& result, const RenderContext& context,
                               const Ray& ray, const HitRecord& hit, const Color&, int) const
{
  const Scene* scene = context.getScene();
  const vector<Light*>& lights = scene->getLights();
  Point hitpos = ray.origin()+ray.direction()*hit.minT();
  Vector normal;
  hit.getPrimitive()->normal(normal, context, hitpos, ray, hit);
  double costheta = Dot(normal, ray.direction());
  if(costheta > 0)
    normal = -normal;

  const Object* world = scene->getObject();

  Color light = scene->getAmbient()*Ka;

#if 0
  for(vector<Light*>::const_iterator iter = lights.begin(); iter != lights.end(); iter++){
#else
    Light*const* begin = &lights[0];
    Light*const* end = &lights[0]+lights.size();
    while(begin != end){
#endif
    Color light_color;
    Vector light_direction;
    double dist = (*begin++)->getLight(light_color, light_direction, context, hitpos);
    double cosphi = Dot(normal, light_direction);
    if(cosphi > 0){
      // Cast shadow rays...
      HitRecord shadowhit(dist);
      Ray shadowray(hitpos, light_direction);
      world->intersect(shadowhit, context, shadowray);
      if(!shadowhit.getPrimitive())
        // No shadows...
        light += light_color*(Kd*cosphi);
    }
  }
  result = light*color;
}
*/

// LSR 11-10-02
// change it a recursive function guarded by 'depth'
// max depth = 10, default depth = 1
// this is a temporary solution.
// to-do:
// 1. make a subclass to provide this recursive function
// 2. update Parser to support parsing 'specular reflection coefficient' of recursive rays
// 3. cannot set default value to 'depth'
void LambertianMaterial::shade(Color& result, 
                               const RenderContext& context,
                               const Ray& ray, 
                               const HitRecord& hit, 
                               const Color& atten, 
                               int depth) const
{
  const Scene* scene = context.getScene();
  const vector<Light*>& lights = scene->getLights();
  Point hitpos = ray.origin()+ray.direction()*hit.minT();
  Vector normal;
  hit.getPrimitive()->normal(normal, context, hitpos, ray, hit);
  double costheta = Dot(normal, ray.direction());
  if(costheta > 0)
    normal = -normal;

  const Object* world = scene->getObject();

  Color light = scene->getAmbient()*Ka;

#if 0
  for(vector<Light*>::const_iterator iter = lights.begin(); iter != lights.end(); iter++){
#else
    Light*const* begin = &lights[0];
    Light*const* end = &lights[0]+lights.size();
    while(begin != end){
#endif
    Color light_color;
    Vector light_direction;
    double dist = (*begin++)->getLight(light_color, light_direction, context, hitpos);
    double cosphi = Dot(normal, light_direction);
    if(cosphi > 0){
      // Cast shadow rays...
      HitRecord shadowhit(dist);
      Ray shadowray(hitpos, light_direction);
      world->intersect(shadowhit, context, shadowray);
      if (!shadowhit.getPrimitive()) {
        // No shadows...
        light += light_color * (Kd * cosphi);  // diffuse

        // specular
        Vector light_reflection = normal * (2.0 * Dot(light_direction, normal)) 
                                  - light_direction;
        double Ks = 1.0;  // temporarily hard coded
        double p = 20.0;  // temporarily hard coded
        double cosalpha = Dot(light_reflection, -(ray.direction()));
        if (cosalpha > 0) {
            light += light_color * (Ks * pow(cosalpha, p));
        }
      }
    }
  }
  result = light*color;

    // recursive specular reflection
    // DEBUG:
    if (depth < 100) {
        // to-do: write a function to compute the specular reflection vector
        // direction
        Vector reflection_direction = 
            ray.direction() - 2.0 * Dot(normal, ray.direction()) * normal;
        // to-do: get this value from input scene description file
        double coefficient = 0.2;  // specular reflection coefficient
        HitRecord reflection_hit(DBL_MAX);
        Ray reflection_ray(hitpos, reflection_direction);
        world->intersect(reflection_hit, context, reflection_ray);
        // continue if intersects some primitive
        if (reflection_hit.getPrimitive()) {
            Color reflected_color;
            Color atten(1, 1, 1);  // place holder
            shade(reflected_color, 
                  context, 
                  reflection_ray, 
                  reflection_hit, 
                  atten, 
                  depth + 1);
            result += (reflected_color * coefficient) * color;
        }
        else {
            Color reflected_color;  // background color
            scene->getBackground()->getBackgroundColor(reflected_color, context, reflection_ray);
            result += (reflected_color * coefficient) * color;
        }
    }
}
