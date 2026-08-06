#include "renderer/gpu_renderer.h"

#include "display.h"
#include "Eigen/Core"

#include "camera/camera.h"
#include "renderer/framebuffer.h"
#include "renderer/kernels/gaussian_pipeline.cuh"
#include "scene/scene.h"

GpuRenderer::GpuRenderer(int w, int h)
    : width(w), height(h), display_(width, height), frame_buf_(width, height) {}
GpuRenderer::~GpuRenderer() {
    rasterizer::free_scratch(scratch_bufs_);
    rasterizer::free_scene(scene_bufs_);
}

void GpuRenderer::set_scene(const Scene& scene) {
    scene_ = &scene;
    size_t N = scene_->gaussians().size();
    const auto& covs = scene_->covarience_3d();
    std::vector<float> covs_float(N * 6);

    // build a padded host buffer for mean
    // one time cost lets if worth it
    const auto& means = scene_->gaussians().mean;
    std::vector<float> padded_means(N * 4, 0.0f);
    for (size_t i = 0; i < N; i++) {
        padded_means[4 * i + 0] = means[i].x();
        padded_means[4 * i + 1] = means[i].y();
        padded_means[4 * i + 2] = means[i].z();
        // [3] stays 0.0f — w component, unused

        // covs
        // taking only 6 elements upper triangule, since it is symmetric
        const Eigen::Matrix3f& c = covs[i];
        covs_float[6 * i + 0] = c(0, 0);
        covs_float[6 * i + 1] = c(0, 1);
        covs_float[6 * i + 2] = c(0, 2);
        covs_float[6 * i + 3] = c(1, 1);
        covs_float[6 * i + 4] = c(1, 2);
        covs_float[6 * i + 5] = c(2, 2);
    }

    // leaks if set_scene called twice , tc later , guard ?
    rasterizer::upload(scene_bufs_,
                       padded_means.data(),
                       reinterpret_cast<const float*>(scene_->gaussians().scale.data()),
                       reinterpret_cast<const float*>(scene_->gaussians().rotation.data()),
                       reinterpret_cast<const float*>(scene_->gaussians().color.data()),
                       scene_->gaussians().opacity.data(),
                       covs_float.data(),
                       scene_->gaussians().size());

    rasterizer::alloc_scratch(scratch_bufs_, scene_bufs_.N, width, height);
}

void GpuRenderer::render(const Camera& cam) {
    const int N = scene_->gaussians().size();
    Eigen::Matrix4f vm = cam.view_matrix();
    auto intrinsic = cam.intrinsic();
    rasterizer::forward(scene_bufs_,
                        scratch_bufs_,
                        vm.data(),
                        intrinsic.fx,
                        intrinsic.fy,
                        intrinsic.cx,
                        intrinsic.cy,
                        frame_buf_.mutable_data());

    display_.show(frame_buf_);
    // cudaMemcpy D->H-> glTexSubImage2D -> draw quad
}