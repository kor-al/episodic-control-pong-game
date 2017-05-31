// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "GameFramework/GameMode.h"
#include "Pong_GameMode.generated.h"

/**
 * 
 */
UCLASS()
class PONGGAME_API APong_GameMode : public AGameMode
{
	GENERATED_BODY()

public:
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "scoring")
	int Cpu_Score;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "scoring")
	int Player_Score;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ball")
	class ABall* Ball_Ref ;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "screenshots")
	class AScreenCapturer* ScreenCapturer;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "screenshots")
	int Frame_Counter;

};
