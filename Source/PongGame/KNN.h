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



	int capacity;
	cv::Mat states;
	std::vector<float> q_values;
	std::vector<float> LRUs;
	int current_capacity;
	float time;


	cv::flann::Index kdtree;

	//xml achive
	/*friend class boost::serialization::access;
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
	BOOST_SERIALIZATION_SPLIT_MEMBER()*/

	//binary achive
	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const
	{
		ar & capacity;
		ar & current_capacity;
		ar & q_values;
		ar & LRUs;
		ar & time;
		ar & states;
	}
	template<class Archive>
	void load(Archive & ar, const unsigned int version)
	{
		ar & capacity;
		ar & current_capacity;
		ar & q_values;
		ar & LRUs;
		ar & time;
		ar & states;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER()


public:

	float knn_value(cv::Mat key, int knn);
	void add(cv::Mat key, float value);
	bool peek(cv::Mat key, float value, bool bModify, float* result_qval);

	void saveIndex(std::string filename) const;
	void loadIndex(std::string filename);

	KNN();
	KNN(int capacity, int dim);
	~KNN();
};

