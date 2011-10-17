
#include "Scene.h"

#include <float.h>
#include <iostream>
#include <stdlib.h>
#include <cassert>
#include <cstdio>

#include "Background.h"
#include "Camera.h"
#include "HitRecord.h"
#include "Image.h"
#include "Light.h"
#include "Material.h"
#include "Object.h"
#include "Ray.h"
#include "RenderContext.h"

using namespace std;

Scene::Scene()
{
  object = 0;
  background = 0;
  camera = 0;
  ambient = Color(0, 0, 0);
  image = 0;
  minAttenuation = 0;
}

Scene::~Scene()
{
  delete object;
  delete background;
  delete camera;
  delete image;
  for(int i=0;i<static_cast<int>(lights.size());i++){
    Light* light = lights[i];
    delete light;
  }
}

void Scene::preprocess()
{
  background->preprocess();
  for(int i=0;i<static_cast<int>(lights.size());i++){
    Light* light = lights[i];
    light->preprocess();
  }
  double aspect_ratio = image->aspect_ratio();
  camera->preprocess(aspect_ratio);
  object->preprocess();
}

void Scene::render()
{
  if(!object || !background || !camera || !image){
    cerr << "Incomplete scene, cannot render!\n";
    exit(1);
  }
  int xres = image->getXresolution();
  int yres = image->getYresolution();
  RenderContext context(this);
  double dx = 2./xres;
  double xmin = -1. + dx/2.;
  double dy = 2./yres;
  double ymin = -1. + dy/2.;
  Color atten(1,1,1);
  for(int i=0;i<yres;i++){
    //cerr << "y=" << i << '\n';
    double y = ymin + i*dy;
    for(int j=0;j<xres;j++){
      double x = xmin + j*dx;
      //cerr << "x=" << j << ", y=" << i << '\n';
      Ray ray;
      camera->makeRay(ray, context, x, y);
      HitRecord hit(DBL_MAX);
      object->intersect(hit, context, ray);
      Color result;
      if(hit.getPrimitive()){
        // Ray hit something...
        const Material* matl = hit.getMaterial();
        matl->shade(result, context, ray, hit, atten, 0);
      } else {
        background->getBackgroundColor(result, context, ray);
      }
      image->set(j, i, result);
    }
  }
}

void Scene::render(const RenderContext& context)
{
  if(!object || !background || !camera || !image){
    cerr << "Incomplete scene, cannot render!\n";
    exit(1);
  }
  int xres = image->getXresolution();
  int yres = image->getYresolution();

  if (context.hasAntiAliasing()) {
    Color atten(1.0, 1.0, 1.0);
    int total = yres * xres;
    int num_finished = 0;
    for (int j = 0; j < yres; ++j) {
      for (int i = 0; i < xres; ++i) {
        vector<Point2D> pl = sampleInPixel(
            context.getPixelSamplingResolution(),
            i, j, xres, yres, context);
        Color result(0.0, 0.0, 0.0);
        for (int k = 0; k < pl.size(); ++k) {
          //Ray ray;
          //camera->makeRay(ray, context, pl[k].x, pl[k].y);
          vector<Ray> rays;
          camera->makeRays(rays, context, pl[k].x, pl[k].y);
          Color subpixel_result(0.0, 0.0, 0.0);
          for (int r = 0; r < rays.size(); ++r) {
            HitRecord hit(DBL_MAX);
            object->intersect(hit, context, rays[r]);
            Color c(0.0, 0.0, 0.0);
            if (hit.getPrimitive()) {
              const Material* matl = hit.getMaterial();
              matl->shade(c, context, rays[r], hit, atten, 0);
            } else {
              background->getBackgroundColor(c, context, rays[r]);
            }
            subpixel_result += c;
          }
          subpixel_result /= (double)rays.size();
          result += subpixel_result;
        }
        result /= (double)pl.size();
        image->set(i, j, result);
        if ((++num_finished) % (total / 10) == 0)
          printf("%d / %d\n", num_finished, total);
      }
    }
  } else {
    double dx = 2./xres;
    double xmin = -1. + dx/2.;
    double dy = 2./yres;
    double ymin = -1. + dy/2.;
    Color atten(1,1,1);
    int total = yres * xres;
    int num_finished = 0;
    for(int i=0;i<yres;i++){
      //cerr << "y=" << i << '\n';
      double y = ymin + i*dy;
      for(int j=0;j<xres;j++){
        double x = xmin + j*dx;
        //cerr << "x=" << j << ", y=" << i << '\n';
        //Ray ray;
        //camera->makeRay(ray, context, x, y);
        vector<Ray> rays;
        camera->makeRays(rays, context, x, y);
        Color result(0.0, 0.0, 0.0);
        for (int r = 0; r < rays.size(); ++r) {
          HitRecord hit(DBL_MAX);
          object->intersect(hit, context, rays[r]);
          Color c(0.0, 0.0, 0.0);
          if(hit.getPrimitive()){
            // Ray hit something...
            const Material* matl = hit.getMaterial();
            matl->shade(c, context, rays[r], hit, atten, 0);
          } else {
            background->getBackgroundColor(c, context, rays[r]);
          }
          result += c;
        }
        result /= (double)rays.size();
        image->set(j, i, result);
        if ((++num_finished) % (total / 10) == 0)
          printf("%d / %d\n", num_finished, total);
      }
    }
  }
}

double Scene::traceRay(Color& result, const RenderContext& context, const Ray& ray, const Color& atten, int depth) const
{
  if(depth >= maxRayDepth || atten.maxComponent() < minAttenuation){
    result = Color(0, 0, 0);
    return 0;
  } else {
    HitRecord hit(DBL_MAX);
    object->intersect(hit, context, ray);
    if(hit.getPrimitive()){
      // Ray hit something...
      const Material* matl = hit.getMaterial();
      matl->shade(result, context, ray, hit, atten, depth);
      return hit.minT();
    } else {
      background->getBackgroundColor(result, context, ray);
      return DBL_MAX;
    }
  }
}

double Scene::traceRay(Color& result, const RenderContext& context, const Object* obj, const Ray& ray, const Color& atten, int depth) const
{
  if(depth >= maxRayDepth || atten.maxComponent() < minAttenuation){
    result = Color(0, 0, 0);
    return 0;
  } else {
    HitRecord hit(DBL_MAX);
    obj->intersect(hit, context, ray);
    if(hit.getPrimitive()){
      // Ray hit something...
      const Material* matl = hit.getMaterial();
      matl->shade(result, context, ray, hit, atten, depth);
      return hit.minT();
    } else {
      background->getBackgroundColor(result, context, ray);
      return DBL_MAX;
    }
  }
}

vector<Point2D> Scene::sampleInPixel(const int n,
                                     const int x,
                                     const int y,
                                     const int xres,
                                     const int yres,
                                     const RenderContext& context) {
  vector<Point2D> ret;
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sx = x + (i + context.generateRandomNumber()) / n;
      sx = sx * 2.0 / xres - 1.0;
      double sy = y + (j + context.generateRandomNumber()) / n;
      sy = sy * 2.0 / yres - 1.0;
      ret.push_back(Point2D(sx, sy));
    }
  }
  return ret;
}
