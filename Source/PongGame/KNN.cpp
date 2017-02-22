// Fill out your copyright notice in the Description page of Project Settings.

#include "PongGame.h"
#include "KNN.h"

KNN::KNN()
{
}

KNN::KNN(int cap, int dim)
{
	UE_LOG(LogTemp, Warning, TEXT("KNN constructor"));
	current_capacity = 0;
	capacity = cap;
	time = 0;
	states = cv::Mat(capacity, dim, CV_32FC1, double(0));
	q_values.resize(capacity, 0);
	LRUs.resize(capacity,0);
	
	//cv::flann::KDTreeIndexParams indexParams(4);
	//kdtree = new cv::flann::Index(states, indexParams);

	//states_index = 0;

	//kdtree = 0;

	//t = 0;
}

double KNN::knn_value(cv::Mat key, int knn)//single query
{
	if (current_capacity == 0)
		return 0.0;
	
	float value = 0;
	int ind = 0;

	//---flann
	//std::vector<int> index(knn);
	//std::vector<float> dist(knn);
	//kdtree->knnSearch(key, index, dist, knn, cv::flann::SearchParams(64));
	//for (int i = 0; i < knn; i++)
	//{
	//	ind = index[i];
	//	value += q_values[ind];
	//	LRUs[ind] = time;
	//	time += 0.01;
	//}
	//return value / knn;

	////---flann
	//std::vector<int> index(knn);
	//std::vector<float> dist(knn);
	//t->knnSearch(key, index, dist, knn, cv::flann::SearchParams(64));
	//for (int i = 0; i < knn; i++)
	//{
	//	ind = index[i];
	//	value += q_values[ind];
	//	LRUs[ind] = time;
	//	time += 0.01;
	//}
	//return value / knn;

	//---flann
	std::vector<int> index(knn);
	std::vector<float> dist(knn);
	//cv::flann::Index kdtree(states.rowRange(0, current_capacity), cv::flann::KDTreeIndexParams(4));

	kdtree.knnSearch(key, index, dist, knn, cv::flann::SearchParams(64));
	for (int i = 0; i < knn; i++)
	{
		ind = index[i];
		value += q_values[ind];
		LRUs[ind] = time;
		time += 0.01;
	}
	return value / knn;

	
	//--nanoflann
	//std::vector<__int64> index(knn);
	//std::vector<float> dist(knn);
	////conver cv mat key to vector
	//const float* p = key.ptr<float>(0);
	//std::vector<float> key_vec(p, p + key.cols);

	////build kdtree
	//Eigen::Map<Eigen::MatrixXf> states_mapped((float*)states.rowRange(0, current_capacity).data, current_capacity, states.cols);
	//kd_tree states_index(states.cols, states_mapped, 10 /* max leaf */);
	//states_index.index->buildIndex();

	////knn search
	////nanoflann::KNNResultSet<float> resultSet(knn);
	////resultSet.init(&index[0], &dist[0]);
	////states_index.index->findNeighbors(resultSet, &key_vec[0], nanoflann::SearchParams(10));
	//states_index.index->knnSearch(&key_vec[0], knn, &index[0], &dist[0]);

	//for (int i = 0; i < knn; i++)
	//{
	//	ind = index[i];
	//	value += q_values[ind];
	//	LRUs[ind] = time;
	//	time += 0.01;
	//}
	//UE_LOG(LogTemp, Warning, TEXT("KNN =  %f"), value / knn);
	//return value / knn;

	//--nanoflann
	//std::vector<__int64> index(knn);
	//std::vector<float> dist(knn);
	//////conver cv mat key to vector
	//const float* p = key.ptr<float>(0);
	//std::vector<float> key_vec(p, p + key.cols);


	//////knn search

	//t->index->knnSearch(&key_vec[0], knn, &index[0], &dist[0]);

	//for (int i = 0; i < knn; i++)
	//{
	//	ind = index[i];
	//	value += q_values[ind];
	//	LRUs[ind] = time;
	//	time += 0.01;
	//}
	//UE_LOG(LogTemp, Warning, TEXT("KNN =  %f"), value / knn);
	//return value / knn;
}

void KNN::add(cv::Mat key, double value)
{
	if (current_capacity>=capacity)
	{
		int old_ind = std::min_element(LRUs.begin(), LRUs.end()) - LRUs.begin();
		key.row(0).copyTo(states.row(old_ind));
		q_values[old_ind] = value;
		LRUs[old_ind] = time;
	}
	else
	{
		key.row(0).copyTo(states.row(current_capacity));
		q_values[current_capacity] = value;
		LRUs[current_capacity] = time;
		current_capacity += 1;
	}
	time += 0.01;
	UE_LOG(LogTemp, Warning, TEXT("current_capacity = %i"), current_capacity);
	UE_LOG(LogTemp, Warning, TEXT("ADD to kdtree"));

	kdtree.build(states.rowRange(0, current_capacity), cv::flann::KDTreeIndexParams(4));

	//---flann
	//cv::flann::KDTreeIndexParams indexParams(4);
	//delete kdtree; ERROR
	//kdtree = new cv::flann::Index(states.rowRange(0, current_capacity), indexParams);

	//---flann
	//cv::flann::KDTreeIndexParams indexParams(4);
	//UE_LOG(LogTemp, Warning, TEXT("rows = %i"), states.rows);
	//t.reset(new cv::flann::Index(states.rowRange(0, current_capacity), indexParams));
	//t.reset(new cv::flann::Index(states, indexParams));



	//---nanoflann

	//Eigen::Map<Eigen::MatrixXf> states_mapped((float*)states.data, states.rows, states.cols);
	//Eigen::Map<Eigen::MatrixXf> states_mapped((float*)states.rowRange(0, current_capacity).data, current_capacity, states.cols);
	//states_index = kd_tree(states.cols, states_mapped, 10 /* max leaf */);
	//UE_LOG(LogTemp, Warning, TEXT("%d"), states_index)
	//states_index.index->buildIndex();

	//---nanoflann

	////Eigen::Map<Eigen::MatrixXf> states_mapped((float*)states.data, states.rows, states.cols);
	//Eigen::Map<Eigen::MatrixXf> states_mapped((float*)states.rowRange(0, current_capacity).data, current_capacity, states.cols);
	////std::unique_ptr<kd_tree> p_temp(new kd_tree(states.cols, states_mapped, 10 ));
	////t = std::move(p_temp);
	//t.reset(new kd_tree(states.cols, states_mapped, 10));
	////UE_LOG(LogTemp, Warning, TEXT("%d"), states_index)
	//t->index->buildIndex();

}

bool KNN::peek(cv::Mat key, double value, bool bModify, double* result_qval)
{
	if (current_capacity == 0)
		return false;

	const float eps_dist = 0.01;
	UE_LOG(LogTemp, Warning, TEXT("PEEK"));
	int knn = 1;

	////---flann
	//std::vector<int> index(knn);
	//std::vector<float> dist(knn);
	//kdtree->knnSearch(key, index, dist, knn, cv::flann::SearchParams(32));
	////printmat(key);
	//if (dist[0] < kEpsDist)
	//{
	//	LRUs[index[0]] = time;
	//	time += 0.01;
	//	if (bModify)
	//		q_values[index[0]] = std::max(q_values[index[0]], value);
	//	//return q_values[index[0]];
	//	*result_qval = q_values[index[0]];
	//	return true;
	//}
	//return false;

	//---flann
	//std::vector<int> index(knn);
	//std::vector<float> dist(knn);
	//t->knnSearch(key, index, dist, knn, cv::flann::SearchParams(32));
	//if (dist[0] < kEpsDist)
	//{
	//	LRUs[index[0]] = time;
	//	time += 0.01;
	//	if (bModify)
	//		q_values[index[0]] = std::max(q_values[index[0]], value);
	//	*result_qval = q_values[index[0]];
	//	return true;
	//}
	//return false;

	//----flann

	std::vector<int> index(knn);
	std::vector<float> dist(knn);
	
	//cv::flann::Index kdtree(states.rowRange(0, current_capacity), cv::flann::KDTreeIndexParams(4));
	kdtree.knnSearch(key, index, dist, knn, cv::flann::SearchParams(32));
	if (dist[0] < eps_dist)
	{
		LRUs[index[0]] = time;
		time += 0.01;
		if (bModify)
			q_values[index[0]] = std::max(q_values[index[0]], value);
		*result_qval = q_values[index[0]];
		return true;
	}
	return false;
	
	//---nanoflann
	//std::vector<__int64> index(knn);
	////std::vector<size_t> index(knn);
	//std::vector<float> dist(knn);
	////conver cv mat key to vector
	//const float* p = key.ptr<float>(0);
	//std::vector<float> key_vec(p, p + key.cols);

	////build kdtree
	//Eigen::Map<Eigen::MatrixXf> states_mapped((float*)states.rowRange(0, current_capacity).data, current_capacity, states.cols);
	//kd_tree states_index(states.cols, states_mapped, 10 /* max leaf */);
	//states_index.index->buildIndex();

	////knn search
	//nanoflann::KNNResultSet<float> resultSet(knn);
	////resultSet.init(&index[0], &dist[0]);
	////states_index.index->findNeighbors(resultSet, &key_vec[0], nanoflann::SearchParams(10));
	//states_index.index->knnSearch(&key_vec[0],knn, &index[0], &dist[0]);
	//if (dist[0] < kEpsDist)
	//{
	//	LRUs[index[0]] = time;
	//	time += 0.01;
	//	if (bModify)
	//	{
	//		q_values[index[0]] = std::max(q_values[index[0]], value);
	//		UE_LOG(LogTemp, Warning, TEXT("Modify %f vs %f"), q_values[index[0]], value);
	//	}
	//	*result_qval = q_values[index[0]];
	//	return true;
	//}
	//return false;

	//---nanoflann
	//std::vector<__int64> index(knn);
	//////std::vector<size_t> index(knn);
	//std::vector<float> dist(knn);
	//////conver cv mat key to vector
	//const float* p = key.ptr<float>(0);
	//std::vector<float> key_vec(p, p + key.cols);

	////knn search
	//t->index->knnSearch(&key_vec[0],knn, &index[0], &dist[0]);
	//if (dist[0] < kEpsDist)
	//{
	//	LRUs[index[0]] = time;
	//	time += 0.01;
	//	if (bModify)
	//	{
	//		q_values[index[0]] = std::max(q_values[index[0]], value);
	//		UE_LOG(LogTemp, Warning, TEXT("Modify %f vs %f"), q_values[index[0]], value);
	//	}
	//	*result_qval = q_values[index[0]];
	//	return true;
	//}
	//return false;

}

KNN::~KNN()
{
}
 
KNN::KNN(const KNN& that)
{
	//cv::flann::KDTreeIndexParams indexParams(4);
	/*if (that.current_capacity)
		kdtree = new cv::flann::Index(that.states.rowRange(0, that.current_capacity), indexParams);
	else
		kdtree = new cv::flann::Index(that.states, indexParams);*/

	//if (that.current_capacity)
	//	kdtree = new cv::flann::Index(that.states.rowRange(0, that.current_capacity), indexParams);
	//	else
	//	kdtree = new cv::flann::Index(that.states, indexParams);*/

	//if (that.current_capacity)
	//{
	//	Eigen::Map<Eigen::MatrixXf> states_mapped((float*)that.states.rowRange(0, that.current_capacity).data, that.current_capacity, that.states.cols);
	//	states_index = new kd_tree(that.states.cols, states_mapped, 10 /* max leaf */);
	//	states_index->index->buildIndex();
	//}
	//else
	//	states_index = 0;
	//t = 0;
	
	current_capacity = that.current_capacity;
	capacity = that.capacity;
	time = that.time;
	states = that.states;
	q_values = that.q_values;
	LRUs = that.LRUs;
}

KNN& KNN::operator=(const KNN& that)
{
	if (this != &that)
	{
		//cv::flann::KDTreeIndexParams indexParams(4);
		//delete kdtree;
		//if (that.current_capacity)
		//	kdtree = new cv::flann::Index(that.states.rowRange(0, that.current_capacity), indexParams);
		//else
		//	kdtree = new cv::flann::Index(that.states, indexParams);

		//delete states_index;
		//if (that.current_capacity)
		//{
		//	Eigen::Map<Eigen::MatrixXf> states_mapped((float*)that.states.rowRange(0, that.current_capacity).data, that.current_capacity, that.states.cols);
		//	states_index = new (std::nothrow) kd_tree(that.states.cols, states_mapped, 10 /* max leaf */);
		//	UE_LOG(LogTemp, Warning, TEXT("copy constrctor %i"), states_index)
		//	states_index->index->buildIndex();
		//}
		//else
		//	states_index = 0;

		//t = 0;
		current_capacity = that.current_capacity;
		capacity = that.capacity;
		time = that.time;
		states = that.states;
		q_values = that.q_values;
		LRUs = that.LRUs;
	}
	return *this;
}


void KNN::saveIndex(std::string filename) const
{
	kdtree.save(filename);
}

void KNN::loadIndex(std::string filename)
{
	UE_LOG(LogTemp, Warning, TEXT("Index Loaded"));
	//kdtree.load(states.rowRange(0, current_capacity), filename);
	//kdtree.build(states.rowRange(0, current_capacity), cv::flann::SavedIndexParams(filename));
	kdtree.build(states.rowRange(0, current_capacity), cv::flann::KDTreeIndexParams(4));
}