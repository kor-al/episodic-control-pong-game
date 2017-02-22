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
	//save_mat_proj();
	//load_mat_proj();
	//save_buf();
	//load_buf();
}

cv::Mat QECtable::fprojection(cv::Mat observation)
{
	observation = observation.reshape(0, observation.cols*observation.rows);
	//now observation is dim_observation x 1
	observation.convertTo(observation, CV_32F);
	cv::Mat state = matrix_proj * observation;
	cv::Mat state_t  = state.t(); //transpose to 1 x state_dim
	return state_t;
}

double QECtable::estimate(cv::Mat observation, int action)
{
	bool bStatePresent = 0;
	double estimated_qvalue= 0;
	cv::Mat state = fprojection(observation);
	bStatePresent = ec_buffer[action].peek(state, 0, false, &estimated_qvalue);
	if (bStatePresent)
		return estimated_qvalue;
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("knn_value"))
		return ec_buffer[action].knn_value(state, knn);
	}
}
void QECtable::update(cv::Mat observation, int action, int reward)
{
	cv::Mat state = fprojection(observation);
	double estimated_qvalue;
	bool bStatePresent = ec_buffer[action].peek(state, reward, true, &estimated_qvalue);
	if (!bStatePresent)
		ec_buffer[action].add(state, reward);
}



void QECtable::load_kdtrees(std::string filename)
{
	for (int i = 0; i < ec_buffer.size(); i++)
		ec_buffer[i].loadIndex(filename + std::to_string(i));
}

void QECtable::save_kdtrees(std::string filename)
{
	for (int i = 0; i < ec_buffer.size(); i++)
		ec_buffer[i].saveIndex(filename + std::to_string(i));
}

QECtable::~QECtable()
{
}