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
	UE_LOG(LogTemp, Warning, TEXT("AI controller Constructor"));
}

void  AAIPaddleController::BeginPlay()
{
	cv::setBreakOnError(true);
	Super::BeginPlay();
	step_count = 0;
	total_frames_count = 0;
	UE_LOG(LogTemp, Warning, TEXT("AI controller BeginPlay"));
	//agent.loadQECtable();
}

void AAIPaddleController::Tick(float DeltaSeconds)
{


	Super::Tick(DeltaSeconds);
	int action = 0;
	//UE_LOG(LogTemp, Warning, TEXT("AI controller Tick"));
	APaddle* paddle = (APaddle*)GetPawn();
	APong_GameMode* game_mode = paddle->pGameMode;
	Score sc = get_score(game_mode);

	int reward = current_score.update(sc);
	cv::Mat screen = get_screen(game_mode);
	bool *bLearningMode = &game_mode->ScreenCapturer->bLearningMode;
	int NumLearningFrames =  game_mode->ScreenCapturer->NumLearningFrames;

	//float ball_position = get_ball_position(game_mode);
	//float pawn_position = ((APawn*)paddle)->GetActorLocation().Z;
	//paddle->MovementDirection = sign(ball_position - pawn_position);

	//skip empty frames
	if (screen.cols == 0)
		return;
	//--------------------------------------------
	step_count++;
	cv::Mat screen_t = transform_image(screen);
	UE_LOG(LogTemp, Warning, TEXT("____reward = %i"), reward);


	if (!*bLearningMode && reward)
	{
		agent.end_episode(reward);
		bFirstTick = true;
		step_count = 0;
		return;
	}

	if (step_count%kFrameSkip) //add skipped frames into one frame to process (while the last action is repeated)
	{
		if (step_count == 1)
		{
			game_mode->ScreenCapturer->MergedScreenshot.Empty();
			screen_t.copyTo(accum_obs);
		}
		else
		{
			cv::addWeighted(screen_t, 1, accum_obs, 1, 0, accum_obs);
		}
		action = prev_action;
	}
	else
	{
		step_count = 0;

		//cv::imwrite("D:/Alice/Documents/HSE/masters/observations/" + std::to_string(total_frames_count) + ".png", accum_obs);
		assign_accum_screen(game_mode, accum_obs);

		total_frames_count++;

		if (*bLearningMode)
		{
			action = agent.random_action();
			if (total_frames_count > NumLearningFrames)
			{
				//stop learning
				*bLearningMode = false;
			}
		}
		else
		{
			cv::Mat* obs;

			if (bEmbedding)
			{
				cv::Mat state = tarray2cvmat(game_mode->ScreenCapturer->State);
				obs = &state;
			}
			else
				obs = &accum_obs;

			if (bFirstTick)
			{
				action = agent.start_episode(*obs);
				bFirstTick = false;
			}
			else
			{
				action = agent.step(*obs, reward);
			}
		}

	}

	UE_LOG(LogTemp, Warning, TEXT("____action = %i"), action);
	paddle->MovementDirection = get_action_direction(action);
	prev_action = action;
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
		cv::Mat(W,H,CV_8U, values + 2 * H*W)
	};
	cv::Mat merged;
	cv::merge(chan, 3, merged);
	return merged;
}

cv::Mat AAIPaddleController::tarray2cvmat(TArray<float> a)
{
	float* values = a.GetData();
	return cv::Mat(1, a.Num(), CV_32FC1, values);
}

void AAIPaddleController::assign_accum_screen(class APong_GameMode*game_mode, cv::Mat mat)
{
	cvMat2tarray(mat, game_mode->ScreenCapturer->MergedScreenshot);
}

Score AAIPaddleController::get_score(class APong_GameMode*game_mode)
{
	if (!game_mode)	return Score();
	else return Score(game_mode->Cpu_Score, game_mode->Player_Score);
}

int AAIPaddleController::get_action_direction(int action)
{
	//3 actions
	//if (action == 0)
	//	return 0;
	//else if (action == 1)
	//	return 1;
	//else
	//	return -1;

	//2 actions
	if (action == 0)
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
	//cv::normalize(screen, screen, 1, 0, cv::NORM_MINMAX); [0,1] in matrix , if commented [0,255]
	screen.convertTo(screen, CV_32FC1);
	return screen;
}

void AAIPaddleController::cvMat2tarray(cv::Mat mat, TArray<uint8>& a)
{
	a.Empty();
	for (int i = 0; i < mat.rows; ++i) {
		a.Append(mat.ptr<uint8>(i), mat.cols);
	}
}