// Fill out your copyright notice in the Description page of Project Settings.

#include "PongGame.h"
#include "AIPaddleController.h"
#include "Pong_GameMode.h"
#include "Paddle.h"
#include "Ball.h"
#include "ScreenCapturer.h"
#include <array>
#include <vector>
#include <cstdlib>
#include <iostream>
#include "Runtime/Slate/Public/Framework/Application/SlateApplication.h"
#include "SceneViewport.h"
#include "Runtime/RenderCore/Public/RenderingThread.h"

// Sets default values
AAIPaddleController::AAIPaddleController()
{
	PrimaryActorTick.bCanEverTick = true;
	bFirstTick = true;
	UE_LOG(LogTemp, Warning, TEXT("AI controller Constructor"))
}

void  AAIPaddleController::BeginPlay()
{
	cv::setBreakOnError(true);
	Super::BeginPlay();
	step_count = 0;
	UE_LOG(LogTemp, Warning, TEXT("AI controller BeginPlay"));
	agent.loadQECtable();
}

void AAIPaddleController::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	step_count++;
	if (step_count%kFrameSkip)
		return;

	int action = 0;
	//UE_LOG(LogTemp, Warning, TEXT("AI controller Tick"));
	APaddle* paddle = (APaddle*)GetPawn();
	APong_GameMode* game_mode = paddle->pGameMode;
	Score sc = get_score(game_mode);

	//Attribute this to previous action
	int reward = current_score.update(sc);
	
	//float ball_position = get_ball_position(game_mode);
	//float pawn_position = ((APawn*)paddle)->GetActorLocation().Z;
	//paddle->MovementDirection = sign(ball_position - pawn_position);
	
	cv::Mat screen = get_screen(game_mode);
	//skip empty frames
	if (screen.cols == 0)
		return;
	cv::Mat screen_t = transform_image(screen);
	if (bFirstTick )
	{
		action = agent.start_episode(screen_t);
		bFirstTick = false;
		return;
	}
	UE_LOG(LogTemp, Warning, TEXT("reward = %d"), reward);
	action = agent.step(screen_t, reward);

	if (reward)
	{
		agent.end_episode(reward);
		bFirstTick = true;
	}
	paddle->MovementDirection = get_action_direction(action);
}


int AAIPaddleController::get_ball_position(class APong_GameMode*game_mode)
{
	if (!game_mode)
		return 0;
	else
	{
		ABall* ball = game_mode->Ball_Ref;
		FVector loc = ((AActor*)ball)->GetActorLocation();
		return loc.Z;
	}
}

float AAIPaddleController::sign(int x)
{
	if (x > 0) return 1.0;
	if (x < 0) return -1.0;
	return 0.0;
}

cv::Mat AAIPaddleController::get_screen(class APong_GameMode*game_mode)
{
	int H = game_mode->ScreenCapturer->Height;
	int W = game_mode->ScreenCapturer->Width;
	TArray<uint8> data = game_mode->ScreenCapturer->Screenshot;
	uint8* values = data.GetData();
	cv::Mat chan[3] = {
		cv::Mat(W,H,CV_8U, values),
		cv::Mat(W,H,CV_8U, values + H*W),
		cv::Mat(W,H,CV_8U, values + 2*H*W)
	};
	cv::Mat merged;
	cv::merge(chan, 3, merged);
	return merged;
}  

Score AAIPaddleController::get_score(class APong_GameMode*game_mode)
{
	if (!game_mode)	return Score();
	else return Score(game_mode->Cpu_Score, game_mode->Player_Score);
}

int AAIPaddleController::get_action_direction(int action)
{
	if (action == 0)
		return 0;
	else if (action == 1)
		return 1;
	else
		return -1;
}

cv::Mat AAIPaddleController::transform_image(cv::Mat screen)
{

	cv::cvtColor(screen, screen, CV_BGR2GRAY);
	cv::threshold(screen, screen, 1, 255, cv::THRESH_BINARY);
	cv::Mat kernel;   //the default structuring element (kernel) for max filter (dilate)
	cv::dilate(screen, screen, kernel);
	cv::threshold(screen, screen, 1, 255, cv::THRESH_BINARY);
	//cv::Mat dst(newH, newW, CV_8U);
	cv::resize(screen, screen, cv::Size(kTransformedImageDim, kTransformedImageDim), cv::INTER_AREA);
	cv::threshold(screen, screen, 1, 255, cv::THRESH_BINARY);
	return screen;
}
