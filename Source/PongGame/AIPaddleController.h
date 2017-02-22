// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/OpenCV/Includes/opencv2/core.hpp"
//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/OpenCV/Includes/opencv2/core/mat.hpp"
//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/OpenCV/Includes/opencv2/highgui.hpp"	
//#include "D:/Alice/Documents/Unreal Projects/PongGame/ThirdParty/OpenCV/Includes/opencv2/imgproc.hpp"
#include "Score.h"
#include "ECagent.h"

#include "AIController.h"
#include "AIPaddleController.generated.h"


/**
 * 
 */
UCLASS()
class PONGGAME_API AAIPaddleController : public AAIController
{
	GENERATED_BODY()
	
public:
	static const int kFrameSkip = 4;
	static const int kTransformedImageDim = 80;

	// Sets default values for this actor's properties
	AAIPaddleController();

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Called every frame
	virtual void Tick(float DeltaSeconds) override;

	int get_ball_position(class APong_GameMode*game_mode);

	float sign(int x);

	int step_count;

	Score current_score;

	ECagent agent;

	cv::Mat get_screen(class APong_GameMode*game_mode);

	Score get_score(class APong_GameMode*game_mode);

	int get_action_direction(int action);

	cv::Mat transform_image(cv::Mat screen);

	bool bFirstTick;

};
