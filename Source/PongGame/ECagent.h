// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "HistoryRecorder.h"
#include "Score.h"
#include "QECtable.h"
#include <ctime>
#include "float.h"

/**
 * 
 */
class PONGGAME_API ECagent
{
private:
	static const int  kKnn = 10;
	static const int kDimState = 10;
	static const int kBufferMaxsize = 10000;
	static const float kEpsMax;
	static const float kEpsMin;
	static const int kEpsDenominator = 100000;
	static const float kECdiscount;
	static const int kNumActions = 2; //0, 1 or 2
	static const int kSaveEpisodes = 10;
	static const std::string kQECtableFilename;
	static const std::string kSummaryFilename;
	static const std::string kKDtreeFilename;

	//-----

	HistoryRecorder history;
	QECtable* qectable;

	float eps_rate;
	float eps;

	int step_counter;
	float episode_reward;

	float total_reward;
	int total_episodes;

	int start_time;
	int total_time;

	cv::Mat last_observation;
	int last_action;

public:
	int start_episode(cv::Mat observation);
	int step(cv::Mat observation, int reward);
	int choose_action(cv::Mat observation, int reward);
	void end_episode(int reward, bool bTerminal = true);

	ECagent();
	~ECagent();

	void saveQECtable();
	void loadQECtable();
};
