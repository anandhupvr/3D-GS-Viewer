#pragma once
#include <cstdint>
namespace rasterizer {

struct SceneBuffers {
    void* d_means = nullptr;
    void* d_scales = nullptr;
    void* d_rotations = nullptr;
    void* d_colors = nullptr;
    void* d_opacities = nullptr;
    void* d_cov3d = nullptr;
    int N = 0;
};

struct ScratchBuffers {
    void* d_viewmat = nullptr;
    void* d_proj_xy = nullptr;
    void* d_depths = nullptr;
    void* d_tiles_touched = nullptr;
    void* d_radii = nullptr;
    void* d_conic_opacities = nullptr;
    void* d_point_offsets = nullptr;
    void* d_cub_temp = nullptr;  // temporary buffer for cub::DeviceScan
    size_t d_cub_temp_bytes = 0;
    void* d_framebuf = nullptr;
    int width, height;

    // bining
    void* d_keys_unsorted = nullptr;
    void* d_vals_unsorted = nullptr;
    void* d_keys_sorted = nullptr;
    void* d_point_list = nullptr;  // guassian ID after sort
    int L_max = 0;
};

void upload(SceneBuffers& scene,
            const float* h_means,
            const float* h_scales,
            const float* h_rotations,
            const float* h_color,
            const float* h_opacity,
            const float* h_cov3d,
            const int size);

void forward(const SceneBuffers& scene,
             ScratchBuffers& scratch,
             const float* h_view,
             const float fx,
             const float fy,
             const float cx,
             const float cy,
             uint8_t* frame_buf);
void alloc_scratch(ScratchBuffers& scratch, int N, int W, int H);
void free_scratch(ScratchBuffers& scratch);
void free_scene(SceneBuffers& scene);
};  // namespace rasterizer