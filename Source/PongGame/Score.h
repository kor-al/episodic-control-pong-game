// Fill out your copyright notice in the Description page of Project Settings.

#pragma once


/**
 * 
 */
class PONGGAME_API Score
{

public:
	int CpuScore;
	int PlayerScore;
	int update(Score newScore);
	Score();
	Score(int CpuS, int PlayerS);
	~Score();
};
