
#include "Scene.h"

#include <float.h>
#include <iostream>
#include <algorithm>
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
  tsfreq = 0;
  sfreq = 0;
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

  /*
  buffer.resize(yres);
  for (int i = 0; i < xres; ++i)
    buffer[i].resize(yres);
  */

  RenderContext context(this);
  Color atten(1.0, 1.0, 1.0);  // to-do: let user assign this?
  double total_per_pixel = max(1, psfreq * psfreq) *
                           max(1, lsfreq * lsfreq) *
                           max(1, tsfreq);

  for (int j = 0; j < yres; ++j) {
    for (int i = 0; i < xres; ++i) {
      Color buffer(0.0, 0.0, 0.0);
      vector<Point2D> pl = sampleInPixel(i, j, xres, yres, context);  // anti-aliasing
      for (int k = 0; k < pl.size(); ++k) {
        vector<Ray> rays;
        camera->makeRays(rays, context, pl[k].x, pl[k].y);  // depth of field
        for (int r = 0; r < rays.size(); ++r) {
          vector<double> tl = sampleOnTime(context);  // motion blur
          for (int t = 0; t < tl.size(); ++t) {
            object->move(tl[t]);
            HitRecord hit(DBL_MAX);
            object->intersect(hit, context, rays[r]);
            Color c(0.0, 0.0, 0.0);
            if (hit.getPrimitive()) {
              const Material* matl = hit.getMaterial();
              matl->shade(c, context, rays[r], hit, atten, 0);
            } else {
              background->getBackgroundColor(c, context, rays[r]);
            }
            buffer += c;
          }
        }
      }
      image->set(i, j, buffer / total_per_pixel);

      // progress bar
      int pre = -1;
      int done = (j * xres + i + 1) * 100 / (yres * xres);
      if (pre != done) {
        printf("\r%3d%% pixels have been rendered", done);
        fflush(stdout);
        pre = done;
      }
    }
  }
  printf("\n");  // start a new line after progress bar
}

void Scene::renderFast()
{
  if (sfreq <= 0) {  // used as a switch for permutation based distributed ray tracing
    render();
    return;
  }

  if(!object || !background || !camera || !image){
    cerr << "Incomplete scene, cannot render!\n";
    exit(1);
  }
  int xres = image->getXresolution();
  int yres = image->getYresolution();

  RenderContext context(this);
  Color atten(1.0, 1.0, 1.0);  // to-do: let user assign this?
  double total_per_pixel = sfreq * sfreq;
  if (psfreq <= 1 && lsfreq <= 1 && tsfreq <= 1)  // all features are disabled
    total_per_pixel = 1.0;

  // to-do: need these?
  // 0 or 1 means that the feature is disabled
  // 0: one default sample
  // 1: one random sample
  if (psfreq > 1)
    psfreq = sfreq;
  if (lsfreq > 1)
    lsfreq = sfreq;
  if (tsfreq > 1)
    tsfreq = sfreq * sfreq;

  for (int j = 0; j < yres; ++j) {
    for (int i = 0; i < xres; ++i) {
      Color buffer(0.0, 0.0, 0.0);

      vector<Point2D> pl = sampleInPixel(i, j, xres, yres, context);  // anti-aliasing
      // expand to match the maximum between the number of samples on the lens and on the timeline
      const int max_num_samples = max(lsfreq * lsfreq, tsfreq);
      if (pl.size() == 1 && max_num_samples > 1) {
        while (pl.size() < max_num_samples)
          pl.push_back(pl[0]);
      }
      vector<Ray> rays;
      camera->makeRays(rays, context, pl);  // depth of field
      vector<double> tl = sampleOnTime(context);  // motion blur
      if (tl.size() == 1 && tl.size() < rays.size()) {  // expand to match the number of rays
        while (tl.size() < rays.size())
          tl.push_back(tl[0]);
      } else {
        random_shuffle(tl.begin(), tl.end());
      }

      for (int k = 0; k < pl.size(); ++k) {
        object->move(tl[k]);
        HitRecord hit(DBL_MAX);
        object->intersect(hit, context, rays[k]);
        Color c(0.0, 0.0, 0.0);
        if (hit.getPrimitive()) {
          const Material* matl = hit.getMaterial();
          matl->shade(c, context, rays[k], hit, atten, 0);
        } else {
          background->getBackgroundColor(c, context, rays[k]);
        }
        buffer += c;
      }

      image->set(i, j, buffer / total_per_pixel);

      // progress bar
      int pre = -1;
      int done = (j * xres + i + 1) * 100 / (yres * xres);
      if (pre != done) {
        printf("\r%3d%% pixels have been rendered", done);
        fflush(stdout);
        pre = done;
      }
    }
  }
  printf("\n");  // start a new line after progress bar
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

  if (ret.empty())  // anti-aliasing is disabled
    ret.push_back(Point2D(sx, sy));

  return ret;
}

vector<double> Scene::sampleOnTime(const RenderContext& context) {
  vector<double> ret;

  // sampling
  const int freq = tsfreq;
  if (freq > 0) {
    for (int i = 0; i < freq; ++i) {
      double st = shutter * (i + context.generateRandomNumber()) / freq;
      ret.push_back(st);
    }
  } else {
    ret.push_back(0.0);  // motion blur is disabled
  }

  return ret;
}
