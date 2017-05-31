// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include "GameFramework/Actor.h"
#include "ScreenCapturer.generated.h"

UCLASS()
class PONGGAME_API AScreenCapturer : public AActor
{
	GENERATED_BODY()

public:

	static const int kStateDim = 32;

	// Sets default values for this actor's properties
	AScreenCapturer();

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Called every frame
	virtual void Tick(float DeltaSeconds) override;

	UPROPERTY(EditAnywhere, Category = Screenshot)
		TArray<uint8> Screenshot;

	UPROPERTY(EditAnywhere, Category = Screenshot)
		int Height;

	UPROPERTY(EditAnywhere, Category = Screenshot)
		int Width;

	UPROPERTY(EditAnywhere, Category = Screenshot)
		TArray<uint8> MergedScreenshot;

	UPROPERTY(EditAnywhere, Category = Screenshot)
		float ScreenshotPeriod;

	UPROPERTY(EditAnywhere, Category = State)
		TArray<float> State;

	UPROPERTY(EditAnywhere, Category = Mode)
		bool bLearningMode;

	UPROPERTY(EditAnywhere, Category = Mode)
		int NumLearningFrames;


	//UFUNCTION(BlueprintCallable, Category = "Python")
	//	void SetState(const TArray<float> s)
	//	{
	//		for (int i = 0; i < State.Num(); i++)
	//			State[i] = s[i];
	//	}

	

private:
	bool CaptureScreenshot(TArray<uint8>* data);

	float ScreenshotTimer;

};