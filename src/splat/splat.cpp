#include "splat/splat.h"

#include "Eigen/Core"
#include "Eigen/Geometry"

Eigen::Matrix3f splat::compute_covariance(const Eigen::Vector3f& scale,
                                          Eigen::Quaternionf& rotation) {
    // E = R S S.T R.T , E = 3x3 symetric matrix describe any oriented
    // ellipsoid shape
    Eigen::Matrix3f s = scale.asDiagonal();
    Eigen::Matrix3f r = rotation.toRotationMatrix();

    auto covarience = r * s * s.transpose() * r.transpose();

    return covarience;
}

Eigen::Vector3f splat::world_to_camera(const Eigen::Vector3f& world_pos,
                                       const Eigen::Matrix4f& view_matrix) {
    // convert to homogenous coordinate
    Eigen::Vector4f cam_point = view_matrix * world_pos.homogeneous();
    Eigen::Vector3f cam_pos = cam_point.head<3>() / cam_point.w();
    return cam_pos;
}

Eigen::Vector2f splat::camera_to_screen(const Eigen::Vector3f& camera_pos,
                                        const Eigen::Matrix3f& intrinsic) {
    // project to screen space
    Eigen::Vector3f screen_point = intrinsic * camera_pos;
    // prespective divide
    return screen_point.head<2>() / screen_point.z();
}

Eigen::Matrix<float, 2, 3> splat::compute_jacobian(const Eigen::Vector3f& camera_pose,
                                                   float fx,
                                                   float fy) {
    Eigen::Matrix<float, 2, 3> jacobian;
    jacobian << fx / camera_pose.z(), 0, -fx * camera_pose.x() / std::pow(camera_pose.z(), 2), 0,
        fy / camera_pose.z(), -fy * camera_pose.y() / std::pow(camera_pose.z(), 2);

    return jacobian;
}

Eigen::Matrix2f splat::project_covarience(const Eigen::Matrix3f& cov3d_world,
                                          const Eigen::Vector3f& camera_pos,
                                          const Eigen::Matrix3f& view_roataion,
                                          float fx,
                                          float fy) {
    auto j = splat::compute_jacobian(camera_pos, fx, fy);
    auto cov3d_camera = view_roataion * cov3d_world * view_roataion.transpose();
    return j * cov3d_camera * j.transpose();
}

// axis-aligned bouding box in screen space (x_min, y_min, x_max, y_max)
// using simplest method, circle approximation would more accurate(using eigen values)
// if the gaussian is rotated bbox misses the corners
// halft width = sqrt(cov2d(0, 0) * 3), halft height = sqrt(cov2d(1, 1) * 3)
Eigen::Vector4i splat::compute_bounding_box(const Eigen::Matrix2f& cov2d,
                                            const Eigen::Vector2f& center_pixels,
                                            int w,
                                            int h) {
    float half_width = std::sqrt(cov2d(0, 0) * 3);
    float half_height = std::sqrt(cov2d(1, 1) * 3);

    Eigen::Vector4i bbox;
    bbox(0) = std::max(0, (int)floor(center_pixels.x() - half_width));      // x_min
    bbox(1) = std::max(0, (int)floor(center_pixels.y() - half_height));     // y_min
    bbox(2) = std::min(w - 1, (int)ceil(center_pixels.x() + half_width));   // x_max
    bbox(3) = std::min(h - 1, (int)ceil(center_pixels.y() + half_height));  // y_max
    return bbox;
}