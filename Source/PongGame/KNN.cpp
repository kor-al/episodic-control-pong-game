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
	states = cv::Mat(capacity, dim, CV_32FC1, float(0));
	q_values.resize(capacity, 0);
	LRUs.resize(capacity, 0);

}

float KNN::knn_value(cv::Mat key, int knn)//single query
{
	if (current_capacity == 0)
		return 0.0;
	UE_LOG(LogTemp, Warning, TEXT("> KNN method"));
	float value = 0;
	int ind = 0;

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

}

void KNN::add(cv::Mat key, float value)
{
	if (current_capacity >= capacity)
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
	UE_LOG(LogTemp, Warning, TEXT("> ADDED to kdtree, now current_capacity = %i"), current_capacity);

	kdtree.build(states.rowRange(0, current_capacity), cv::flann::KDTreeIndexParams(4));

}

bool KNN::peek(cv::Mat key, float value, bool bModify, float* result_qval)
{
	if (current_capacity == 0)
		return false;

	const float eps_dist = 1e-8;
	UE_LOG(LogTemp, Warning, TEXT("> PEEK, current cap = %u"), current_capacity);
	int knn = 1;

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
			UE_LOG(LogTemp, Warning, TEXT(">---modify: max (%f, %f)"), q_values[index[0]], value)
			q_values[index[0]] = std::max(q_values[index[0]], value);
		*result_qval = q_values[index[0]];
		return true;
	}
	return false;


}

KNN::~KNN()
{
}



void KNN::saveIndex(std::string filename) const
{
	kdtree.save(filename);
}

void KNN::loadIndex(std::string filename)
{
	UE_LOG(LogTemp, Warning, TEXT("____Index Loaded____"));
	//kdtree.load(states.rowRange(0, current_capacity), filename);
	//kdtree.build(states.rowRange(0, current_capacity), cv::flann::SavedIndexParams(filename));
	kdtree.build(states.rowRange(0, current_capacity), cv::flann::KDTreeIndexParams(4));
}