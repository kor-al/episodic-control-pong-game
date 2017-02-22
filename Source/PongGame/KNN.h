// Fill out your copyright notice in the Description page of Project Settings.

#pragma once


#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include <opencv2/flann/flann.hpp>
#include <ctime>
#include <vector>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/Eigen/Dense"
//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/nanoflann-1.2.3/include/nanoflann.hpp"


#include <boost/archive/xml_oarchive.hpp> 
#include <boost/archive/xml_iarchive.hpp> 
#include <boost/serialization/split_member.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "cvMat_serialization.h"

#include <iostream> 
#include <fstream> 


/**
 * 
 */
class PONGGAME_API KNN
{
private:

	//typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>  kd_tree;
	//kd_tree* states_index;

	//std::unique_ptr<kd_tree> t;
	//std::shared_ptr<kd_tree> t;

	int capacity;
	cv::Mat states;
	std::vector<double> q_values;
	std::vector<double> LRUs;
	int current_capacity;
	double time;

	///std::shared_ptr<cv::flann::Index> t;
	//cv::flann::Index* kdtree;
	cv::flann::Index kdtree;

	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const
	{
		ar & BOOST_SERIALIZATION_NVP(capacity);
		ar & BOOST_SERIALIZATION_NVP(current_capacity);
		ar & BOOST_SERIALIZATION_NVP(q_values);
		ar & BOOST_SERIALIZATION_NVP(LRUs);
		ar & BOOST_SERIALIZATION_NVP(time);
		ar & BOOST_SERIALIZATION_NVP(states);
	}
	template<class Archive>
	void load(Archive & ar, const unsigned int version)
	{
		ar & BOOST_SERIALIZATION_NVP(capacity);
		ar & BOOST_SERIALIZATION_NVP(current_capacity);
		ar & BOOST_SERIALIZATION_NVP(q_values);
		ar & BOOST_SERIALIZATION_NVP(LRUs);
		ar & BOOST_SERIALIZATION_NVP(time);
		ar & BOOST_SERIALIZATION_NVP(states);
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER()

public:

	double knn_value(cv::Mat key, int knn);
	void add(cv::Mat key, double value);
	bool peek(cv::Mat key, double value, bool bModify, double* result_qval);
	void printmat(cv::Mat observation);

	void saveIndex(std::string filename) const;
	void loadIndex(std::string filename);

	KNN();
	KNN(int capacity, int dim);
	KNN(const KNN& that);
	KNN& operator=(const KNN& that);
	~KNN();
};

