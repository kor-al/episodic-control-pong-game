#pragma once

#include "Score.h"
#include "ECagent.h"

#include "AIController.h"
#include "Python.h"

#include "AIPaddleController.generated.h"



/**
*
*/
UCLASS()
class PONGGAME_API AAIPaddleController : public AAIController
{
	GENERATED_BODY()

public:
	static const int kFrameSkip = 5; //better than 4
	static const int kTransformedImageDim = 84;
	static const bool bEmbedding = 1; //0 == random projection, 1 == VAE

	// Sets default values for this actor's properties
	AAIPaddleController();

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Called every frame
	virtual void Tick(float DeltaSeconds) override;

	int get_ball_position(class APong_GameMode*game_mode);

	float sign(int x);

	int step_count; // to skip frames
	//to save observation with number and stop learning phase
	int total_frames_count;

	int last_action;

	Score current_score;

	ECagent agent;

	cv::Mat get_screen(class APong_GameMode*game_mode);

	Score get_score(class APong_GameMode*game_mode);

	int get_action_direction(int action);

	cv::Mat transform_image(cv::Mat screen);

	bool bFirstTick;

	//to merge every kFrameSkip frames
	int prev_action;
	cv::Mat accum_obs;
	void cvMat2tarray(cv::Mat mat, TArray<uint8>& a);
	void assign_accum_screen(class APong_GameMode*game_mode, cv::Mat mat);

	cv::Mat tarray2cvmat(TArray<float> a);

};