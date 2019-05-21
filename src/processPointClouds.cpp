// PCL lib Functions for processing point clouds

#include "processPointClouds.h"
#include<unordered_set>

//constructor:
template <typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}

//de-constructor:
template <typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}

template <typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // Fill in the function to do voxel grid point reduction and region based filtering
    typename pcl::VoxelGrid<PointT> ds;
    ds.setInputCloud(cloud);
    ds.setLeafSize(filterRes, filterRes, filterRes);
    ds.filter(*cloud);

    typename pcl::CropBox<PointT> point_reduction;
    point_reduction.setInputCloud(cloud);
    point_reduction.setMin(Eigen::Vector4f(-2.1, -1.6, -2, 1));
    point_reduction.setMax(Eigen::Vector4f( 2.9,  1.6,  0, 1));
    point_reduction.setNegative(true);
    point_reduction.filter(*cloud);

    typename pcl::CropBox<PointT> roi;
    roi.setInputCloud(cloud);
    roi.setMin(minPoint);
    roi.setMax(maxPoint);
    roi.filter(*cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloud;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud)
{
    // Create two new point clouds, one cloud with obstacles and other with segmented plane
    typename pcl::PointCloud<PointT>::Ptr segmented_plane(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr obstacle_cloud(new pcl::PointCloud<PointT>());

    for (int index : inliers->indices)
    {
        segmented_plane->points.push_back(cloud->points[index]);
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstacle_cloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(cloud, cloud);
    return segResult;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0)
    {
        std::cout << "No planar model estimated." << std::endl;
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, cloud);
    return segResult;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::RANSAC_3d(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    std::unordered_set<int> inliersResult;
    srand(time(NULL));
    while (maxIterations--)
    {
        std::unordered_set<int> inliers;
        while (inliers.size() < 3)
            inliers.insert(rand() % (cloud->points.size()));
        auto iter = inliers.begin();
        float x1 = cloud->points[*iter].x;
        float y1 =  cloud->points[*iter].y;
        float z1 = cloud->points[*iter].z;
        iter++;
        float x2 = cloud->points[*iter].x;
        float y2 = cloud->points[*iter].y;
        float z2 = cloud->points[*iter].z;
        iter++;
        float x3 = cloud->points[*iter].x;
        float y3 = cloud->points[*iter].y;
        float z3 = cloud->points[*iter].z;

        float a = (y1 - y2) * (z3 - z1) - (z1 - z2) * (y3 - y1);
        float b = (z1 - z2) * (x3 - x1) - (x1 - x2) * (z3 - z1);
        float c = (x1 - x2) * (y3 - y1) - (y1 - y2) * (x3 - x1);
        float d = a * x1 + b * y1 + c * z1;

        for (int index = 0; index < cloud->points.size(); index++)
        {
            if (inliers.count(index) > 0)
                continue;
            PointT point = cloud->points[index];
            float x4 = point.x;
            float y4 = point.y;
            float z4 = point.z;

            float dist = fabs(a * x4 + b * y4 + c * z4 + d) / sqrt(a * a + b * b + c * c);
            if (dist <= distanceThreshold)
                inliers.insert(index);
        }
        if (inliers.size() > inliersResult.size())
            inliersResult = inliers;
    }
    typename pcl::PointCloud<PointT>::Ptr cloudInliers(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr cloudOutliers(new pcl::PointCloud<PointT>());

    for (int index = 0; index < cloud->points.size(); index++)
    {
        PointT point = cloud->points[index];
        if (inliersResult.count(index))
            cloudInliers->points.push_back(point);
        else
            cloudOutliers->points.push_back(point);
    }
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(cloudOutliers, cloudInliers);
    return segResult;
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // Fill in the function to perform euclidean clustering to group detected obstacles
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> eucledian_clustering;
    eucledian_clustering.setClusterTolerance(clusterTolerance);
    eucledian_clustering.setMinClusterSize(minSize);
    eucledian_clustering.setMaxClusterSize(maxSize);
    eucledian_clustering.setSearchMethod(tree);
    eucledian_clustering.setInputCloud(cloud);
    eucledian_clustering.extract(cluster_indices);

    for (pcl::PointIndices getIndices : cluster_indices)
    {
        typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        for (int index : getIndices.indices)
            cloud_cluster->points.push_back(cloud->points[index]);

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1.1;
        cloud_cluster->is_dense = true;

        clusters.push_back(cloud_cluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template<typename PointT>
void ProcessPointClouds<PointT>::helper_clustering(int idx, typename pcl::PointCloud<PointT>::Ptr cloud, std::vector<int>& cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol)
{
	processed[idx] = true;
	cluster.push_back(idx);
	std::vector<int> nearest = tree->search(cloud->points[idx], distanceTol);
	for(auto id : nearest)
	{
		if(!processed[id])
			helper_clustering(id, cloud, cluster, processed, tree, distanceTol);
	}
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::euclidean_cluster(typename pcl::PointCloud<PointT>::Ptr cloud, KdTree* tree, float distanceTol, int minSize, int maxSize)
{
	std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
	std::vector<bool> processed(cloud->points.size(), false);
	for(size_t idx = 0; idx < cloud->points.size(); ++idx)
	{
		if(processed[idx] == false)
		{
			std::vector<int> cluster_idx;
            typename pcl::PointCloud<PointT>::Ptr cloudCluster (new pcl::PointCloud<PointT>);
			helper_clustering(idx, cloud, cluster_idx, processed, tree, distanceTol);
            if(cluster_idx.size() >= minSize && cluster_idx.size() <= maxSize)
            {
                for(int i = 0; i < cluster_idx.size(); i++)
                {
                    PointT point;
                    point = cloud->points[cluster_idx[i]];
                    cloudCluster->points.push_back(point);
                }
                cloudCluster->width = cloudCluster->points.size();
                cloudCluster->height = 1;
                clusters.push_back(cloudCluster);
            }
            else{
                for(int i = 1; i < cluster_idx.size(); i++)
                {
                    processed[cluster_idx[i]] = false;
                }
            }
		}
	}
	return clusters;
}

template <typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

template <typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII(file, *cloud);
    std::cerr << "Saved " << cloud->points.size() << " data points to " + file << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size() << " data points from " + file << std::endl;

    return cloud;
}

template <typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;
}