#include <cmath>

#include <cccl/cub/device/device_scan.cuh>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

// #include <glm/glm.hpp>

#include "renderer/kernels/cuda_utils.cuh"
#include "renderer/kernels/gaussian_math.cuh"
#include "renderer/kernels/gaussian_pipeline.cuh"

namespace rasterizer {

__global__ void project_gaussians(int N,
                                  const float* __restrict__ d_means,
                                  const float* __restrict__ d_view,
                                  const float* __restrict__ d_cov3d,
                                  const float* __restrict__ d_opacity,
                                  float fx,
                                  float fy,
                                  float cx,
                                  float cy,
                                  float* d_means2d,
                                  float* d_depths,
                                  int* d_tiles_touched,
                                  float* d_radii,
                                  float4* d_conic_opacities,
                                  uchar4* d_framebuf,
                                  int W,
                                  int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // float4 p = make_float4(d_means[3 * idx], d_means[3 * idx + 1], d_means[3 * idx + 2], 0);
    float4 p = reinterpret_cast<const float4*>(d_means)[idx];
    float3 cam = gmath::world_to_camera(p, d_view);

    if (cam.z < 0.2f)
        return;

    float inv_z = 1.0f / cam.z;
    int u = (int)(fx * cam.x * inv_z + cx);
    int v = (int)(fy * cam.y * inv_z + cy);

    if (u < 0 || u >= W || v < 0 || v >= H)
        return;

    d_means2d[2 * idx] = (float)u;
    d_means2d[2 * idx + 1] = (float)v;
    d_depths[idx] = cam.z;

    auto const cov3d_i = d_cov3d + 6 * idx;

    /// covarience3d already provided from cpu, later port it to gpu here
    float3 cov2d = gmath::compute_cov3d_inv(cov3d_i, cam, d_view, fx, fy);

    // // calculate inv
    // // regularization(prevents singular mat on inversion)
    float a = cov2d.x * 0.3f;
    float b = cov2d.y;
    float c = cov2d.z + 0.3f;

    // // invert 2x2 : [[a,b], [b, c]]^-1 = 1/ (ac - b^2) * [[c, -b], [-b, a]]
    float det = a * c - b * b;
    float inv_det = 1.0f / (det + 1e-7f);
    // conic as float3 {a', b', c'} = upper triangle of inverse
    float3 cov2d_inv = make_float3(c * inv_det, -b * inv_det, a * inv_det);

    // approximate gaussian (circle)
    float mid = 0.5f * (cov2d.x + cov2d.z);  // (a+c)/2
    float disc = sqrtf(fmaxf(0.0f, mid * mid - det));
    float lambda1 = mid + disc;  // larger eigenvalue
    float lambda2 = mid - disc;  // smaller eigenvalue
    float radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    // helper for sorting tile_touched
    int tW = (int)ceilf((float)W / 16);
    int tH = (int)ceilf((float)H / 16);

    int tile_min_x = max(0, (int)floorf((u - radius) / 16.0f));
    int tile_max_x = min(tW, (int)ceilf((u + radius) / 16.0f));
    int tile_min_y = max(0, (int)floorf((v - radius) / 16.0f));
    int tile_max_y = min(tH, (int)ceilf((v + radius) / 16.0f));

    d_tiles_touched[idx] = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y);
    d_radii[idx] = radius;

    // conic selection equation
    d_conic_opacities[idx] = make_float4(cov2d_inv.x, cov2d_inv.y, cov2d_inv.z, d_opacity[idx]);

    // write white dot to framebuffer ** this is per thread write !!!!!
    d_framebuf[v * W + u] = make_uchar4(255, 255, 255, 255);
}

void upload(SceneBuffers& scene,
            const float* h_means,
            const float* h_scales,
            const float* h_rotations,
            const float* h_colors,
            const float* h_opacity,
            const float* h_cov3d,
            const int N) {
    // this 32(warp) x 16 (float4) = 512 bytest = 4 transactions and multiple of 128
    const size_t means_bytes = N * 4 * sizeof(float);  // change from 3 -> 4 then pad w=0
    const size_t scales_bytes = N * 3 * sizeof(float);
    const size_t rot_bytes = N * 4 * sizeof(float);
    const size_t col_bytes = N * 3 * sizeof(float);
    const size_t opacity_bytes = N * sizeof(float);
    const size_t cov3d_bytes = N * 6 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&scene.d_means, means_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_scales, scales_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_rotations, rot_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_colors, col_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_opacities, opacity_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_cov3d, cov3d_bytes));

    CUDA_CHECK(cudaMemcpy(scene.d_means, h_means, means_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_scales, h_scales, scales_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_rotations, h_rotations, rot_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_colors, h_colors, col_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_opacities, h_opacity, opacity_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_cov3d, h_cov3d, cov3d_bytes, cudaMemcpyHostToDevice));
    scene.N = N;
    printf("uploaded scene buffer");
}

void forward(const SceneBuffers& scene,
             ScratchBuffers& scratch,
             const float* h_view,
             const float fx,
             const float fy,
             const float cx,
             const float cy,
             uint8_t* frame_buf) {
    // const size_t view_mat_size = 16 * sizeof(float);  // only need 12
    // float* d_view_mat;
    // CUDA_CHECK(cudaMalloc(&d_view_mat, view_mat_size));
    CUDA_CHECK(
        cudaMemset(scratch.d_framebuf, 0, (size_t)scratch.width * scratch.height * sizeof(uchar4)));
    CUDA_CHECK(cudaMemcpy(scratch.d_viewmat, h_view, 16 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((scene.N + block.x - 1) / block.x);

    project_gaussians<<<grid, block>>>(scene.N,
                                       static_cast<const float*>(scene.d_means),
                                       static_cast<const float*>(scratch.d_viewmat),
                                       static_cast<const float*>(scene.d_cov3d),
                                       static_cast<const float*>(scene.d_opacities),
                                       fx,
                                       fy,
                                       cx,
                                       cy,
                                       static_cast<float*>(scratch.d_proj_xy),
                                       static_cast<float*>(scratch.d_depths),
                                       static_cast<int*>(scratch.d_tiles_touched),
                                       static_cast<float*>(scratch.d_radii),
                                       static_cast<float4*>(scratch.d_conic_opacities),
                                       static_cast<uchar4*>(scratch.d_framebuf),  // debug/test
                                       scratch.width,
                                       scratch.height);
    cudaDeviceSynchronize();

    // prefix sum
    // tiles_touched [3,3,1,0,2] -> point_offset [3, 6, 7, 7, 9]
    int* d_in = static_cast<int*>(scratch.d_tiles_touched);
    int* d_out = static_cast<int*>(scratch.d_point_offsets);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        &scratch.d_cub_temp, scratch.d_cub_temp_bytes, d_in, d_out, scene.N));

    // sorting

    int L = 0;
    cudaMemcpy(&L,
               static_cast<int*>(scratch.d_point_offsets) + scene.N - 1,
               sizeof(int),
               cudaMemcpyDeviceToHost);
    printf("total points to rasterize: %d : gaussian count : %d\n ", L, scene.N);
    cudaMemcpy(frame_buf,
               scratch.d_framebuf,
               (size_t)scratch.width * scratch.height * 4,
               cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaFree(scratch.d_cub_temp));
}

void alloc_scratch(ScratchBuffers& scratch, int N, int W, int H) {
    scratch.width = W;
    scratch.height = H;

    CUDA_CHECK(cudaMalloc(&scratch.d_depths, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_proj_xy, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_viewmat, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_tiles_touched, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scratch.d_point_offsets, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scratch.d_radii, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_conic_opacities, N * sizeof(float4)));

    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_bytes, (int*)nullptr, (int*)nullptr, N));
    CUDA_CHECK(cudaMalloc(&scratch.d_cub_temp, temp_bytes));
    scratch.d_cub_temp_bytes = temp_bytes;

    CUDA_CHECK(cudaMalloc(&scratch.d_framebuf, (size_t)W * H * sizeof(uchar4)));
}

void free_scratch(ScratchBuffers& scratch) {
    cudaFree(scratch.d_proj_xy);
    cudaFree(scratch.d_depths);
    cudaFree(scratch.d_viewmat);
    cudaFree(scratch.d_tiles_touched);
    cudaFree(scratch.d_point_offsets);
    cudaFree(scratch.d_radii);
    cudaFree(scratch.d_conic_opacities);
    cudaFree(scratch.d_framebuf);

    scratch.d_proj_xy = nullptr;
    scratch.d_depths = nullptr;
    scratch.d_viewmat = nullptr;
    scratch.d_tiles_touched = nullptr;
    scratch.d_point_offsets = nullptr;
    scratch.d_radii = nullptr;
    scratch.d_conic_opacities = nullptr;
    scratch.d_framebuf = nullptr;
}

void free_scene(SceneBuffers& scene) {
    cudaFree(scene.d_means);
    cudaFree(scene.d_scales);
    cudaFree(scene.d_rotations);
    cudaFree(scene.d_colors);
    cudaFree(scene.d_opacities);
    scene.d_means = nullptr;
    scene.d_scales = nullptr;
    scene.d_rotations = nullptr;
    scene.d_colors = nullptr;
    scene.d_opacities = nullptr;
    scene.N = 0;
}

};  // namespace rasterizer