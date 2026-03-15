#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/common.h>

using PointT       = pcl::PointXYZ;
using NormalT      = pcl::Normal;
using PointNormalT = pcl::PointNormal;

using CloudT   = pcl::PointCloud<PointT>;
using CloudNT  = pcl::PointCloud<PointNormalT>;
using NormalsT = pcl::PointCloud<NormalT>;

struct RegistrationResult
{
    std::string     name;
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    double          fitness         = 0.0;
    double          inlier_rmse     = 0.0;
    size_t          correspondence_size = 0;
    bool            success         = false;
};

void draw_registration_result(
    const CloudT::Ptr& source,          // 未变换的源点云（用于变换）
    const CloudT::Ptr& target,          // 目标点云
    const Eigen::Matrix4f& transform,
    const std::string& window_name = "Registration Result")
{
    auto viewer = pcl::visualization::PCLVisualizer::Ptr(
        new pcl::visualization::PCLVisualizer(window_name));

    viewer->setBackgroundColor(0.1, 0.1, 0.1);  // 深背景，更接近 Open3D

    // ------------------ 变换后的源点云（橙色，主显示） ------------------
    CloudT::Ptr source_trans(new CloudT);
    pcl::transformPointCloud(*source, *source_trans, transform);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> 
        source_handler(source_trans, 255, 180, 0);  // 橙黄 [1.0, 0.706, 0.0] ≈ 255,180,0

    viewer->addPointCloud<PointT>(source_trans, source_handler, "source_transformed");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "source_transformed");  // ← 关键：调小到1~1.5

    // 可选：加一点透明，让重叠更明显（PCL对点云透明支持一般，0.7~0.9较好）
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, 0.85, "source_transformed");

    // ------------------ 目标点云（青色） ------------------
    pcl::visualization::PointCloudColorHandlerCustom<PointT> 
        target_handler(target, 0, 166, 237);  // 青蓝 [0.0, 0.651, 0.929] ≈ 0,166,237

    viewer->addPointCloud<PointT>(target, target_handler, "target");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "target");  // 同上，小点径

    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, 0.85, "target");

    // 坐标轴小一点，不抢镜
    viewer->addCoordinateSystem(0.2);

    viewer->initCameraParameters();

    // 尝试更合理的初始相机视角（根据你的点云调整这些值）
    // 示例：假设点云在原点附近，尺寸几米
    viewer->setCameraPosition(  0,   0,  -3,     // 相机位置
                                0,   0,   0,     // 看向哪里
                                0,  -1,   0);    // up 方向

    viewer->spin();
}
RegistrationResult run_icp(
    const CloudNT::Ptr& source,
    const CloudNT::Ptr& target,
    const Eigen::Matrix4f& init_guess,
    const std::string& method_name,
    float max_correspondence_distance,
    int max_iterations = 50)
{
    RegistrationResult res;
    res.name = method_name;

    CloudNT::Ptr output(new CloudNT);

    if (method_name == "Point-to-Point")
    {
        pcl::IterativeClosestPoint<PointNormalT, PointNormalT> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaximumIterations(max_iterations);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);
        icp.setTransformationEpsilon(1e-8);
        icp.setEuclideanFitnessEpsilon(1e-8);

        icp.align(*output, init_guess);

        res.transformation = icp.getFinalTransformation();
        res.fitness         = icp.getFitnessScore(max_correspondence_distance);
        res.inlier_rmse     = std::sqrt(res.fitness);
        res.correspondence_size = 0;  // PCL 不公开 correspondences_
        res.success         = icp.hasConverged();
    }
    else if (method_name == "Point-to-Plane")
    {
        pcl::IterativeClosestPointWithNormals<PointNormalT, PointNormalT> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaximumIterations(max_iterations);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);
        icp.setTransformationEpsilon(1e-8);

        icp.align(*output, init_guess);

        res.transformation = icp.getFinalTransformation();
        res.fitness         = icp.getFitnessScore(max_correspondence_distance);
        res.inlier_rmse     = std::sqrt(res.fitness);
        res.correspondence_size = 0;
        res.success         = icp.hasConverged();
    }
    else if (method_name == "Generalized-ICP")
    {
        pcl::GeneralizedIterativeClosestPoint<PointNormalT, PointNormalT> gicp;
        gicp.setInputSource(source);
        gicp.setInputTarget(target);
        gicp.setMaximumIterations(max_iterations);
        gicp.setMaxCorrespondenceDistance(max_correspondence_distance);
        gicp.setTransformationEpsilon(1e-8);

        gicp.align(*output, init_guess);

        res.transformation = gicp.getFinalTransformation();
        res.fitness         = gicp.getFitnessScore(max_correspondence_distance);
        res.inlier_rmse     = std::sqrt(res.fitness);
        res.correspondence_size = 0;
        res.success         = gicp.hasConverged();
    }

    return res;
}

int main()
{
    CloudT::Ptr cloud_source(new CloudT);
    CloudT::Ptr cloud_target(new CloudT);

    if (pcl::io::loadPCDFile<PointT>("cloud_bin_0.pcd", *cloud_source) == -1 ||
        pcl::io::loadPCDFile<PointT>("cloud_bin_1.pcd", *cloud_target) == -1)
    {
        std::cerr << "无法读取点云文件！\n";
        return -1;
    }

    std::cout << "源点云点数: " << cloud_source->size() << "\n";
    std::cout << "目标点云点数: " << cloud_target->size() << "\n";

    // 降采样
    const float voxel_size = 0.005f;

    pcl::VoxelGrid<PointT> vg;
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);

    CloudT::Ptr source_ds(new CloudT);
    CloudT::Ptr target_ds(new CloudT);

    vg.setInputCloud(cloud_source); vg.filter(*source_ds);
    vg.setInputCloud(cloud_target); vg.filter(*target_ds);

    std::cout << "降采样后源点云: " << source_ds->size() << "\n";
    std::cout << "降采样后目标点云: " << target_ds->size() << "\n";

    // 法向量估计
    pcl::NormalEstimationOMP<PointT, NormalT> ne;
    ne.setRadiusSearch(voxel_size * 2.0f);

    NormalsT::Ptr source_normals(new NormalsT);
    NormalsT::Ptr target_normals(new NormalsT);

    std::cout << "计算源点云法向量 ...\n";
    ne.setInputCloud(source_ds);
    ne.compute(*source_normals);

    std::cout << "计算目标点云法向量 ...\n";
    ne.setInputCloud(target_ds);
    ne.compute(*target_normals);

    if (source_normals->empty() || target_normals->empty())
    {
        std::cerr << "法向量计算失败！\n";
        return -1;
    }

    // 合并 xyz + normal
    CloudNT::Ptr source_with_normals(new CloudNT);
    CloudNT::Ptr target_with_normals(new CloudNT);

    pcl::concatenateFields(*source_ds, *source_normals, *source_with_normals);
    pcl::concatenateFields(*target_ds, *target_normals, *target_with_normals);

    // 初始变换
    Eigen::Matrix4f trans_init = (Eigen::Matrix4f() <<
         0.862f,  0.011f, -0.507f,  0.5f,
        -0.139f,  0.967f, -0.215f,  0.7f,
         0.487f,  0.255f,  0.835f, -1.4f,
         0.0f,    0.0f,    0.0f,    1.0f).finished();

    std::cout << "\n显示初始位姿（按 q 关闭）...\n";
    draw_registration_result(source_ds, target_ds, trans_init, "Initial Alignment - Check Overlap");

    // ICP 对比
    float threshold = voxel_size * 4.0f;
    std::cout << "\n配准最大对应距离: " << threshold << "\n";

    std::vector<std::string> methods = {
        "Point-to-Point",
        "Point-to-Plane",
        "Generalized-ICP"
    };

    std::vector<RegistrationResult> results;

    for (const auto& method : methods)
    {
        std::cout << "\n运行: " << method << " ...\n";
        auto res = run_icp(source_with_normals, target_with_normals, trans_init, method, threshold, 50);
        results.push_back(res);

        std::cout << "  fitness         : " << res.fitness << "\n";
        std::cout << "  inlier_rmse     : " << res.inlier_rmse << "\n";
        std::cout << "  correspondence  : " << res.correspondence_size << " (PCL 未公开)\n";
        std::cout << "  converged       : " << (res.success ? "Yes" : "No") << "\n";
    }

    // 排序：fitness 越大越好，其次 rmse 越小越好
    std::sort(results.begin(), results.end(),
        [](const RegistrationResult& a, const RegistrationResult& b) {
            if (std::abs(a.fitness - b.fitness) > 1e-6)
                return a.fitness > b.fitness;
            return a.inlier_rmse < b.inlier_rmse;
        });

    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "               排序结果 (fitness 降序)\n";
    std::cout << std::string(70,'=') << "\n";
    std::cout << std::left
              << std::setw(20) << "方法"
              << std::setw(14) << "fitness"
              << std::setw(14) << "rmse"
              << std::setw(16) << "对应点数"
              << "推荐\n"
              << std::string(75,'-') << "\n";

    for (const auto& r : results)
    {
        std::string recommend = (r.fitness > 0.3) ? "★ 首选" :
                                (r.fitness > 0.1) ? "可尝试" : "较差";
        std::cout << std::left
                  << std::setw(20) << r.name
                  << std::setw(14) << std::fixed << std::setprecision(6) << r.fitness
                  << std::setw(14) << r.inlier_rmse
                  << std::setw(16) << (r.correspondence_size == 0 ? "未知" : std::to_string(r.correspondence_size))
                  << recommend << "\n";
    }

    // 显示最佳结果
    if (!results.empty() && results.front().fitness > 1e-5f)
    {
        std::cout << "\n显示最佳结果: " << results.front().name << "\n";
        draw_registration_result(source_ds, target_ds, results.front().transformation,
                                 "Best - " + results.front().name);
    }
    else
    {
        std::cout << "\n所有方法 fitness 都很低，可能初始位姿偏差过大或重叠不足。\n";
    }

    std::cout << "\n程序结束。\n";
    return 0;
}
