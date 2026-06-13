#pragma once
#include "display.h"
#include <sys/types.h>

#include "camera/camera.h"
#include "renderer/framebuffer.h"
#include "renderer/kernels/gaussian_pipeline.cuh"
#include "renderer/renderer.h"
#include "scene/scene.h"

class GpuRenderer : public Renderer {
   public:
    GpuRenderer(int w, int h);
    ~GpuRenderer();
    void set_scene(const Scene& scene) override;
    void render(const Camera& cam) override;

   private:
    int width, height;
    Display display_;
    FrameBuffer frame_buf_;

    const Scene* scene_;
    rasterizer::SceneBuffers scene_bufs_;
    rasterizer::ScratchBuffers scratch_bufs_;
};