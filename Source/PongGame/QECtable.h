// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/OpenCV/Includes/opencv2/core.hpp"
//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/OpenCV/Includes/opencv2/core/mat.hpp"
//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/OpenCV/Includes/opencv2/highgui.hpp"	
//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/OpenCV/Includes/opencv2/imgproc.hpp"
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

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& archive, const unsigned int version)
	{
			archive & BOOST_SERIALIZATION_NVP(knn);
			archive & BOOST_SERIALIZATION_NVP(ec_buffer);
			archive & BOOST_SERIALIZATION_NVP(buffer_maxsize);
			archive & BOOST_SERIALIZATION_NVP(matrix_proj);
	}
public:
	cv::Mat fprojection(cv::Mat observation);
	double estimate(cv::Mat observation, int action);
	void update(cv::Mat observation, int action, int reward);
	QECtable(int knn, int dim_state, int dim_observation, int buffer_maxsize, int num_action);
	QECtable();
	~QECtable();
	void load_kdtrees(std::string filename);
	void save_kdtrees(std::string filename);
	
};
