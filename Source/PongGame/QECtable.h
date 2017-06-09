// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "KNN.h"

/**
*
*/
class PONGGAME_API QECtable
{
private:
	int knn;
	std::vector<KNN> ec_buffer;
	int buffer_maxsize;
	cv::Mat matrix_proj;
	bool bEmbedding;

	//xml achive
	//friend class boost::serialization::access;
	//template<class Archive>
	//void serialize(Archive& archive, const unsigned int version)
	//{
	//		archive & BOOST_SERIALIZATION_NVP(knn);
	//		archive & BOOST_SERIALIZATION_NVP(ec_buffer);
	//		archive & BOOST_SERIALIZATION_NVP(buffer_maxsize);
	//		archive & BOOST_SERIALIZATION_NVP(matrix_proj);
	//}

	//binary achive
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& archive, const unsigned int version)
	{
		archive & knn;
		archive & eps;
		archive & ec_buffer;
		archive & buffer_maxsize;
		archive & matrix_proj;
	}

public:
	float eps; // to serialize with QEC table
	cv::Mat fprojection(cv::Mat observation);
	float estimate(cv::Mat observation, int action, cv::Mat state = cv::Mat(0, 0, CV_32F));
	void update(cv::Mat observation, int action, float value, cv::Mat state = cv::Mat(0, 0, CV_32F));
	QECtable(int knn, int dim_state, int dim_observation, int buffer_maxsize, int num_action, bool embedding = 0);
	QECtable();
	~QECtable();
	void load_kdtrees(std::string filename);
	void save_kdtrees(std::string filename);
	void save_mat();
	void print_mat(); // just to check random matrix (.txt file)
	void load_mat();
};