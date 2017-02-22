// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "Score.h"
#include <vector>


/**
 * 
 */
class PONGGAME_API HistoryNode
{
public:
	cv::Mat observation;
	int action;
	int reward;
	bool bTerminal;
	HistoryNode(cv::Mat obs, int a, int r, bool bT) :
		observation(obs), action(a), reward(r), bTerminal(bT)
	{};
	~HistoryNode()
	{};
};

class PONGGAME_API HistoryRecorder
{
public:
	std::vector<HistoryNode> history;
	HistoryRecorder();
	~HistoryRecorder();
	void add_node(cv::Mat obs, int a, int r, bool bT)
	{
		HistoryNode node = HistoryNode(obs, a, r, bT);
		history.push_back(node);
	};
};

