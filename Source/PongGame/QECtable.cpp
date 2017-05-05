// Fill out your copyright notice in the Description page of Project Settings.

#include "PongGame.h"
#include "QECtable.h"


QECtable::QECtable()
{
}
QECtable::QECtable(int k, int dim_state, int dim_observation, int buffer_msize, int num_action)
{
	//dim_observation = H_observation * W_observation;
	knn = k;
	buffer_maxsize = buffer_msize;
	for (int i = 0; i < num_action + 1; i++)
	{
		ec_buffer.push_back(KNN(buffer_msize, dim_state));
	}
	matrix_proj = cv::Mat(dim_state, dim_observation, CV_32FC1);
	cv::randn(matrix_proj, cv::Scalar(0.0), cv::Scalar(1.1) );
	//save_mat();
	//print_mat();
	//load_mat();

}

cv::Mat QECtable::fprojection(cv::Mat observation)
{
	observation = observation.reshape(0, observation.cols*observation.rows);
	//now observation is dim_observation x 1

	cv::Mat state = matrix_proj * observation;
	cv::Mat state_t  = state.t(); //transpose to 1 x state_dim
	return state_t;
}

float QECtable::estimate(cv::Mat observation, int action)
{
	bool bStatePresent = 0;
	float estimated_qvalue= 0;
	cv::Mat state = fprojection(observation);
	bStatePresent = ec_buffer[action].peek(state, 0, false, &estimated_qvalue);
	if (bStatePresent)
	{
		UE_LOG(LogTemp, Warning, TEXT("QEC: take existed value"))
			return estimated_qvalue;
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("QEC: knn_value"))
		return ec_buffer[action].knn_value(state, knn);
	}
}
void QECtable::update(cv::Mat observation, int action, float value)
{
	cv::Mat state = fprojection(observation);
	float estimated_qvalue;
	bool bStatePresent = ec_buffer[action].peek(state, value, true, &estimated_qvalue);
	if (!bStatePresent)
		ec_buffer[action].add(state, value);
}

void QECtable::save_kdtrees(std::string filename)
{
	for (int i = 0; i < ec_buffer.size(); i++)
		ec_buffer[i].saveIndex(filename + std::to_string(i));
}


void QECtable::load_kdtrees(std::string filename)
{
	for (int i = 0; i < ec_buffer.size(); i++)
		ec_buffer[i].loadIndex(filename + std::to_string(i));
}

void QECtable::save_mat()
{
	//std::ofstream file("rand_mat.dat");
	std::ofstream file("matrices.bin", std::ios::out | std::ios::binary);
	boost::archive::binary_oarchive oa(file);
	oa << matrix_proj;
	//boost::archive::xml_oarchive oa(file);
	//oa & BOOST_SERIALIZATION_NVP(matrix_proj);
	file.close();
	print_mat();
}

void QECtable::print_mat()
{
	int a = 0; //debug
	UE_LOG(LogTemp, Warning, TEXT("____save matrices (RANDOM MAT and mats for action = %i)____") , a);
	std::ofstream file_matproj;
	file_matproj.open("matproj.txt");
	for (int i = 0; i < matrix_proj.rows; i++)
	{
		for (int j = 0; j < matrix_proj.cols; j++)
		{
			file_matproj << matrix_proj.at<float>(i, j) << " ";
		}
		file_matproj << "\n";
	}
	file_matproj.close();

}

void QECtable::load_mat()
{

	//std::ifstream file("rand_mat.dat");
	std::ifstream file("matrices.bin", std::ios::in| std::ios::binary);
	UE_LOG(LogTemp, Warning, TEXT("load mat"));
	cv::Mat temp(matrix_proj.rows, matrix_proj.cols, CV_32FC1);
	if (file.good())
	{
		boost::archive::binary_iarchive ia(file);
		ia >> temp;
		//boost::archive::xml_iarchive ia(file);
		//ia >> BOOST_SERIALIZATION_NVP(temp);
		for (int i = 0; i < 5; i++)
		{
			UE_LOG(LogTemp, Warning, TEXT("M_temp(0,%i) = %f"), i,temp.at<float>(0, i));
		}
	}
}



QECtable::~QECtable()
{

}