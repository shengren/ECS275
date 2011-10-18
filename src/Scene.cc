
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
  psfreq = 0;  // to-do: use initialization list?
  lsfreq = 0;
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

  buffer.resize(yres);
  for (int i = 0; i < xres; ++i)
    buffer[i].resize(yres);

  RenderContext context(this);
  Color atten(1.0, 1.0, 1.0);  // to-do: let user assign this?

  vector<double> tl = sampleOnTime(context);

  for (int t = 1; t < tl.size(); ++t) {
    object->move(tl[t] - tl[t - 1]);  // at least 2 samples in 'tl' (0.0 and 0.0)
    for (int j = 0; j < yres; ++j) {
      for (int i = 0; i < xres; ++i) {
        Color result(0.0, 0.0, 0.0);
        vector<Point2D> pl = sampleInPixel(i, j, xres, yres, context);
        for (int k = 0; k < pl.size(); ++k) {
          vector<Ray> rays;
          camera->makeRays(rays, context, pl[k].x, pl[k].y);
          Color spresult(0.0, 0.0, 0.0);
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
            spresult += c;
          }
          spresult /= (double)rays.size();
          result += spresult;
        }
        result /= (double)pl.size();
        //image->set(i, j, result);
        buffer[i][j] += result;
      }
    }
  }

  for (int i = 0; i < xres; ++i) {
    for (int j = 0; j < yres; ++j) {
      buffer[i][j] /= (double)(tl.size() - 1);  // to-do: count and divide only one time
      image->set(i, j, buffer[i][j]);
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

vector<Point2D> Scene::sampleInPixel(const int x,
                                     const int y,
                                     const int xres,
                                     const int yres,
                                     const RenderContext& context) {
  vector<Point2D> ret;

  double sx = (x + 0.5) * 2.0 / xres - 1.0;
  double sy = (y + 0.5) * 2.0 / yres - 1.0;
  ret.push_back(Point2D(sx, sy));

  // sampling
  const int freq = psfreq;
  for (int i = 0; i < freq; ++i) {
    for (int j = 0; j < freq; ++j) {
      double sx = x + (i + context.generateRandomNumber()) / freq;
      sx = sx * 2.0 / xres - 1.0;
      double sy = y + (j + context.generateRandomNumber()) / freq;
      sy = sy * 2.0 / yres - 1.0;
      ret.push_back(Point2D(sx, sy));
    }
  }

  return ret;
}

vector<double> Scene::sampleOnTime(const RenderContext& context) {
  vector<double> ret;

  ret.push_back(0.0);
  ret.push_back(0.0);

  // sampling
  const int freq = tsfreq;
  if (freq > 0) {
    for (int i = 0; i < freq; ++i) {
      double st = shutter * (i + context.generateRandomNumber()) / freq;
      ret.push_back(st);
    }
  }

  return ret;
}
