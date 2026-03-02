#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <string>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/search/kdtree.h>

#include <Eigen/Dense>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef pcl::PointXYZI PointT;

ros::Publisher marker_pub;
ros::Publisher plane_pub;

struct LidarCircleConfig
{
    double plane_distance_threshold;
    int    plane_max_iterations;
    int    plane_min_inliers;

    int    table_min_points;

    double normal_arrow_length;
    double normal_scale_x;
    double normal_scale_y;

    double cluster_tolerance;
    int    cluster_min_size;
    int    cluster_max_size;

    double circle_min_radius;
    double circle_max_radius;
    double circle_alpha;

    int    sync_queue_size;

    std::string raw_topic;
    std::string interp_topic;
};

bool fitCircleLS(const std::vector<Eigen::Vector2d>& pts,
                 Eigen::Vector2d& center,
                 double& radius)
{
    if (pts.size() < 3) return false;

    Eigen::MatrixXd A(pts.size(), 3);
    Eigen::VectorXd b(pts.size());

    for (size_t i = 0; i < pts.size(); ++i)
    {
        double x = pts[i].x();
        double y = pts[i].y();
        A(i,0) = 2*x;
        A(i,1) = 2*y;
        A(i,2) = 1;
        b(i)   = x*x + y*y;
    }

    Eigen::Vector3d sol = A.colPivHouseholderQr().solve(b);
    center = sol.head<2>();
    radius = std::sqrt(sol(2) + center.squaredNorm());

    return true;
}

void processCloud(const sensor_msgs::PointCloud2ConstPtr& msg,
                  const std::string& ns_prefix,
                  const LidarCircleConfig& cfg)
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty())
        return;

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(cfg.plane_distance_threshold);
    seg.setMaxIterations(cfg.plane_max_iterations);

    pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr plane_coeff(new pcl::ModelCoefficients);

    seg.setInputCloud(cloud);
    seg.segment(*plane_inliers, *plane_coeff);

    if (static_cast<int>(plane_inliers->indices.size()) < cfg.plane_min_inliers)
        return;

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(plane_inliers);
    extract.setNegative(false);

    pcl::PointCloud<PointT>::Ptr table_cloud(new pcl::PointCloud<PointT>);
    extract.filter(*table_cloud);

    if (static_cast<int>(table_cloud->size()) < cfg.table_min_points)
        return;

    sensor_msgs::PointCloud2 plane_msg;
    pcl::toROSMsg(*table_cloud, plane_msg);
    plane_msg.header = msg->header;
    plane_pub.publish(plane_msg);

    Eigen::Vector3d normal(
        plane_coeff->values[0],
        plane_coeff->values[1],
        plane_coeff->values[2]);
    normal.normalize();

    visualization_msgs::MarkerArray marker_array;

    visualization_msgs::Marker normal_marker;
    normal_marker.header = msg->header;
    normal_marker.ns = ns_prefix + "_plane_normal";
    normal_marker.id = 0;
    normal_marker.type = visualization_msgs::Marker::ARROW;
    normal_marker.action = visualization_msgs::Marker::ADD;

    Eigen::Vector3d centroid(0,0,0);
    for (auto& p : table_cloud->points)
        centroid += Eigen::Vector3d(p.x, p.y, p.z);
    centroid /= table_cloud->size();

    geometry_msgs::Point p1, p2;
    p1.x = centroid.x(); p1.y = centroid.y(); p1.z = centroid.z();
    p2.x = centroid.x() + normal.x() * cfg.normal_arrow_length;
    p2.y = centroid.y() + normal.y() * cfg.normal_arrow_length;
    p2.z = centroid.z() + normal.z() * cfg.normal_arrow_length;

    normal_marker.points = {p1, p2};
    normal_marker.scale.x = cfg.normal_scale_x;
    normal_marker.scale.y = cfg.normal_scale_y;
    normal_marker.color.g = 1.0;
    normal_marker.color.a = 1.0;

    marker_array.markers.push_back(normal_marker);

    Eigen::Vector3d u = normal.unitOrthogonal();
    Eigen::Vector3d v = normal.cross(u);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(table_cloud);

    std::vector<pcl::PointIndices> clusters;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(cfg.cluster_tolerance);
    ec.setMinClusterSize(cfg.cluster_min_size);
    ec.setMaxClusterSize(cfg.cluster_max_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(table_cloud);
    ec.extract(clusters);

    pcl::ProjectInliers<PointT> projector;
    projector.setModelType(pcl::SACMODEL_PLANE);
    projector.setModelCoefficients(plane_coeff);

    int marker_id = 0;

    for (auto& c : clusters)
    {
        pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
        for (int idx : c.indices)
            cluster->push_back(table_cloud->points[idx]);

        pcl::PointCloud<PointT>::Ptr projected(new pcl::PointCloud<PointT>);
        projector.setInputCloud(cluster);
        projector.filter(*projected);

        Eigen::Vector3d mean(0,0,0);
        for (auto& p : projected->points)
            mean += Eigen::Vector3d(p.x, p.y, p.z);
        mean /= projected->size();

        std::vector<Eigen::Vector2d> pts2d;
        for (auto& p : projected->points)
        {
            Eigen::Vector3d P(p.x, p.y, p.z);
            P -= mean;
            pts2d.emplace_back(P.dot(u), P.dot(v));
        }

        Eigen::Vector2d c2;
        double r;
        if (!fitCircleLS(pts2d, c2, r))
            continue;

        if (r < cfg.circle_min_radius || r > cfg.circle_max_radius)
            continue;

        Eigen::Vector3d center3 = mean + c2.x()*u + c2.y()*v;

        visualization_msgs::Marker m;
        m.header = msg->header;
        m.ns = ns_prefix + "_lidar_circles";
        m.id = marker_id++;
        m.type = visualization_msgs::Marker::SPHERE;
        m.action = visualization_msgs::Marker::ADD;

        m.pose.position.x = center3.x();
        m.pose.position.y = center3.y();
        m.pose.position.z = center3.z();
        m.pose.orientation.w = 1.0;

        m.scale.x = r * 2.0;
        m.scale.y = r * 2.0;
        m.scale.z = r * 2.0;

        m.color.r = 1.0;
        m.color.a = cfg.circle_alpha;

        marker_array.markers.push_back(m);
    }

    marker_pub.publish(marker_array);
}

void syncedCallback(const sensor_msgs::PointCloud2ConstPtr& raw,
                    const sensor_msgs::PointCloud2ConstPtr& interp,
                    const LidarCircleConfig& cfg)
{
    ROS_INFO("Ejecutando callback sincronizado.");
    processCloud(raw, "raw", cfg);
    processCloud(interp, "interp", cfg);
}

int main(int argc, char** argv)
{
    ROS_INFO("Inicializando nodo.");

    ros::init(argc, argv, "lidar_circle_detector");
    ros::NodeHandle nh("~");

    std::string config_file;
    if (!nh.getParam("config_file", config_file))
        return 1;

    YAML::Node config = YAML::LoadFile(config_file);

    LidarCircleConfig cfg;

    cfg.plane_distance_threshold = config["plane"]["distance_threshold"].as<double>();
    cfg.plane_max_iterations     = config["plane"]["max_iterations"].as<int>();
    cfg.plane_min_inliers        = config["plane"]["min_inliers"].as<int>();

    cfg.table_min_points         = config["table"]["min_points"].as<int>();

    cfg.normal_arrow_length      = config["normal"]["arrow_length"].as<double>();
    cfg.normal_scale_x           = config["normal"]["scale_x"].as<double>();
    cfg.normal_scale_y           = config["normal"]["scale_y"].as<double>();

    cfg.cluster_tolerance        = config["clustering"]["tolerance"].as<double>();
    cfg.cluster_min_size         = config["clustering"]["min_size"].as<int>();
    cfg.cluster_max_size         = config["clustering"]["max_size"].as<int>();

    cfg.circle_min_radius        = config["circle"]["min_radius"].as<double>();
    cfg.circle_max_radius        = config["circle"]["max_radius"].as<double>();
    cfg.circle_alpha             = config["circle"]["alpha"].as<double>();

    cfg.sync_queue_size          = config["sync"]["queue_size"].as<int>();

    cfg.raw_topic                = config["topics"]["raw"].as<std::string>();
    cfg.interp_topic             = config["topics"]["interpolated"].as<std::string>();

    ROS_INFO("Configuracion cargada.");

    plane_pub = nh.advertise<sensor_msgs::PointCloud2>("detected_plane", 1);
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("lidar_circles", 1);

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_raw(
        nh, cfg.raw_topic, 1);

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_interp(
        nh, cfg.interp_topic, 1);

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2,
        sensor_msgs::PointCloud2> SyncPolicy;

    message_filters::Synchronizer<SyncPolicy> sync(
        SyncPolicy(cfg.sync_queue_size), sub_raw, sub_interp);

    sync.registerCallback(
        boost::bind(&syncedCallback, _1, _2, cfg));

    ROS_INFO("Nodo en ejecucion");
    ros::spin();
    return 0;
}