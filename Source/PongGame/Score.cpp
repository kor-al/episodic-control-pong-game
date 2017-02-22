// Fill out your copyright notice in the Description page of Project Settings.

#include "PongGame.h"
#include "Score.h"

Score::Score()
{
	CpuScore = 0;
	PlayerScore = 0;
}

Score::Score(int CpuS, int PlayerS)
{
		CpuScore = CpuS;
		PlayerScore = PlayerS;
}

bool operator==(const Score& l, const Score& r)
{
	return (l.CpuScore == r.CpuScore) && (l.PlayerScore == r.PlayerScore);
}


int Score::update(Score newScore)
{
	int reward = 0;
	if (*this == newScore)
		return reward;
	else
	{
		reward = (newScore.PlayerScore - newScore.CpuScore) - (PlayerScore - CpuScore);
		PlayerScore = newScore.PlayerScore; 
		CpuScore = newScore.CpuScore;
		return reward;
	}
}

Score::~Score()
{
}
