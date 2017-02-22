// Copyright 1998-2016 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Boilerplate C++ definitions for a single module.
	This is automatically generated by UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "PongGame.h"
#include "GeneratedCppIncludes.h"
#include "PongGame.generated.dep.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCode1PongGame() {}
	void AAIPaddleController::StaticRegisterNativesAAIPaddleController()
	{
	}
	IMPLEMENT_CLASS(AAIPaddleController, 519314201);
	void ABall::StaticRegisterNativesABall()
	{
	}
	IMPLEMENT_CLASS(ABall, 2102974918);
	void APaddle::StaticRegisterNativesAPaddle()
	{
	}
	IMPLEMENT_CLASS(APaddle, 1885151586);
	void APong_GameMode::StaticRegisterNativesAPong_GameMode()
	{
	}
	IMPLEMENT_CLASS(APong_GameMode, 786744405);
	void AScreenCapturer::StaticRegisterNativesAScreenCapturer()
	{
	}
	IMPLEMENT_CLASS(AScreenCapturer, 1101672577);
#if USE_COMPILED_IN_NATIVES
// Cross Module References
	AIMODULE_API class UClass* Z_Construct_UClass_AAIController();
	ENGINE_API class UClass* Z_Construct_UClass_AActor();
	ENGINE_API class UClass* Z_Construct_UClass_APawn();
	ENGINE_API class UClass* Z_Construct_UClass_AGameMode();

	PONGGAME_API class UClass* Z_Construct_UClass_AAIPaddleController_NoRegister();
	PONGGAME_API class UClass* Z_Construct_UClass_AAIPaddleController();
	PONGGAME_API class UClass* Z_Construct_UClass_ABall_NoRegister();
	PONGGAME_API class UClass* Z_Construct_UClass_ABall();
	PONGGAME_API class UClass* Z_Construct_UClass_APaddle_NoRegister();
	PONGGAME_API class UClass* Z_Construct_UClass_APaddle();
	PONGGAME_API class UClass* Z_Construct_UClass_APong_GameMode_NoRegister();
	PONGGAME_API class UClass* Z_Construct_UClass_APong_GameMode();
	PONGGAME_API class UClass* Z_Construct_UClass_AScreenCapturer_NoRegister();
	PONGGAME_API class UClass* Z_Construct_UClass_AScreenCapturer();
	PONGGAME_API class UPackage* Z_Construct_UPackage__Script_PongGame();
	UClass* Z_Construct_UClass_AAIPaddleController_NoRegister()
	{
		return AAIPaddleController::StaticClass();
	}
	UClass* Z_Construct_UClass_AAIPaddleController()
	{
		static UClass* OuterClass = NULL;
		if (!OuterClass)
		{
			Z_Construct_UClass_AAIController();
			Z_Construct_UPackage__Script_PongGame();
			OuterClass = AAIPaddleController::StaticClass();
			if (!(OuterClass->ClassFlags & CLASS_Constructed))
			{
				UObjectForceRegistration(OuterClass);
				OuterClass->ClassFlags |= 0x20900280;


				OuterClass->StaticLink();
#if WITH_METADATA
				UMetaData* MetaData = OuterClass->GetOutermost()->GetMetaData();
				MetaData->SetValue(OuterClass, TEXT("HideCategories"), TEXT("Collision Rendering Utilities|Transformation"));
				MetaData->SetValue(OuterClass, TEXT("IncludePath"), TEXT("AIPaddleController.h"));
				MetaData->SetValue(OuterClass, TEXT("ModuleRelativePath"), TEXT("AIPaddleController.h"));
#endif
			}
		}
		check(OuterClass->GetClass());
		return OuterClass;
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_AAIPaddleController(Z_Construct_UClass_AAIPaddleController, &AAIPaddleController::StaticClass, TEXT("AAIPaddleController"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(AAIPaddleController);
	UClass* Z_Construct_UClass_ABall_NoRegister()
	{
		return ABall::StaticClass();
	}
	UClass* Z_Construct_UClass_ABall()
	{
		static UClass* OuterClass = NULL;
		if (!OuterClass)
		{
			Z_Construct_UClass_AActor();
			Z_Construct_UPackage__Script_PongGame();
			OuterClass = ABall::StaticClass();
			if (!(OuterClass->ClassFlags & CLASS_Constructed))
			{
				UObjectForceRegistration(OuterClass);
				OuterClass->ClassFlags |= 0x20900080;


				OuterClass->StaticLink();
#if WITH_METADATA
				UMetaData* MetaData = OuterClass->GetOutermost()->GetMetaData();
				MetaData->SetValue(OuterClass, TEXT("IncludePath"), TEXT("Ball.h"));
				MetaData->SetValue(OuterClass, TEXT("ModuleRelativePath"), TEXT("Ball.h"));
#endif
			}
		}
		check(OuterClass->GetClass());
		return OuterClass;
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_ABall(Z_Construct_UClass_ABall, &ABall::StaticClass, TEXT("ABall"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(ABall);
	UClass* Z_Construct_UClass_APaddle_NoRegister()
	{
		return APaddle::StaticClass();
	}
	UClass* Z_Construct_UClass_APaddle()
	{
		static UClass* OuterClass = NULL;
		if (!OuterClass)
		{
			Z_Construct_UClass_APawn();
			Z_Construct_UPackage__Script_PongGame();
			OuterClass = APaddle::StaticClass();
			if (!(OuterClass->ClassFlags & CLASS_Constructed))
			{
				UObjectForceRegistration(OuterClass);
				OuterClass->ClassFlags |= 0x20900080;


PRAGMA_DISABLE_DEPRECATION_WARNINGS
				UProperty* NewProp_MovementDirection = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("MovementDirection"), RF_Public|RF_Transient|RF_MarkAsNative) UFloatProperty(CPP_PROPERTY_BASE(MovementDirection, APaddle), 0x0010000000000005);
				UProperty* NewProp_pGameMode = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("pGameMode"), RF_Public|RF_Transient|RF_MarkAsNative) UObjectProperty(CPP_PROPERTY_BASE(pGameMode, APaddle), 0x0010000000000005, Z_Construct_UClass_APong_GameMode_NoRegister());
PRAGMA_ENABLE_DEPRECATION_WARNINGS
				OuterClass->StaticLink();
#if WITH_METADATA
				UMetaData* MetaData = OuterClass->GetOutermost()->GetMetaData();
				MetaData->SetValue(OuterClass, TEXT("HideCategories"), TEXT("Navigation"));
				MetaData->SetValue(OuterClass, TEXT("IncludePath"), TEXT("Paddle.h"));
				MetaData->SetValue(OuterClass, TEXT("ModuleRelativePath"), TEXT("Paddle.h"));
				MetaData->SetValue(NewProp_MovementDirection, TEXT("Category"), TEXT("Paddle"));
				MetaData->SetValue(NewProp_MovementDirection, TEXT("ModuleRelativePath"), TEXT("Paddle.h"));
				MetaData->SetValue(NewProp_pGameMode, TEXT("Category"), TEXT("Paddle"));
				MetaData->SetValue(NewProp_pGameMode, TEXT("ModuleRelativePath"), TEXT("Paddle.h"));
#endif
			}
		}
		check(OuterClass->GetClass());
		return OuterClass;
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_APaddle(Z_Construct_UClass_APaddle, &APaddle::StaticClass, TEXT("APaddle"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(APaddle);
	UClass* Z_Construct_UClass_APong_GameMode_NoRegister()
	{
		return APong_GameMode::StaticClass();
	}
	UClass* Z_Construct_UClass_APong_GameMode()
	{
		static UClass* OuterClass = NULL;
		if (!OuterClass)
		{
			Z_Construct_UClass_AGameMode();
			Z_Construct_UPackage__Script_PongGame();
			OuterClass = APong_GameMode::StaticClass();
			if (!(OuterClass->ClassFlags & CLASS_Constructed))
			{
				UObjectForceRegistration(OuterClass);
				OuterClass->ClassFlags |= 0x2090028C;


PRAGMA_DISABLE_DEPRECATION_WARNINGS
				UProperty* NewProp_ScreenCapturer = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("ScreenCapturer"), RF_Public|RF_Transient|RF_MarkAsNative) UObjectProperty(CPP_PROPERTY_BASE(ScreenCapturer, APong_GameMode), 0x0010000000000005, Z_Construct_UClass_AScreenCapturer_NoRegister());
				UProperty* NewProp_Ball_Ref = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("Ball_Ref"), RF_Public|RF_Transient|RF_MarkAsNative) UObjectProperty(CPP_PROPERTY_BASE(Ball_Ref, APong_GameMode), 0x0010000000000005, Z_Construct_UClass_ABall_NoRegister());
				UProperty* NewProp_Player_Score = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("Player_Score"), RF_Public|RF_Transient|RF_MarkAsNative) UUnsizedIntProperty(CPP_PROPERTY_BASE(Player_Score, APong_GameMode), 0x0010000000000005);
				UProperty* NewProp_Cpu_Score = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("Cpu_Score"), RF_Public|RF_Transient|RF_MarkAsNative) UUnsizedIntProperty(CPP_PROPERTY_BASE(Cpu_Score, APong_GameMode), 0x0010000000000005);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
				OuterClass->ClassConfigName = FName(TEXT("Game"));
				OuterClass->StaticLink();
#if WITH_METADATA
				UMetaData* MetaData = OuterClass->GetOutermost()->GetMetaData();
				MetaData->SetValue(OuterClass, TEXT("HideCategories"), TEXT("Info Rendering MovementReplication Replication Actor Input Movement Collision Rendering Utilities|Transformation"));
				MetaData->SetValue(OuterClass, TEXT("IncludePath"), TEXT("Pong_GameMode.h"));
				MetaData->SetValue(OuterClass, TEXT("ModuleRelativePath"), TEXT("Pong_GameMode.h"));
				MetaData->SetValue(OuterClass, TEXT("ShowCategories"), TEXT("Input|MouseInput Input|TouchInput"));
				MetaData->SetValue(NewProp_ScreenCapturer, TEXT("Category"), TEXT("screenshots"));
				MetaData->SetValue(NewProp_ScreenCapturer, TEXT("ModuleRelativePath"), TEXT("Pong_GameMode.h"));
				MetaData->SetValue(NewProp_Ball_Ref, TEXT("Category"), TEXT("ball"));
				MetaData->SetValue(NewProp_Ball_Ref, TEXT("ModuleRelativePath"), TEXT("Pong_GameMode.h"));
				MetaData->SetValue(NewProp_Player_Score, TEXT("Category"), TEXT("scoring"));
				MetaData->SetValue(NewProp_Player_Score, TEXT("ModuleRelativePath"), TEXT("Pong_GameMode.h"));
				MetaData->SetValue(NewProp_Cpu_Score, TEXT("Category"), TEXT("scoring"));
				MetaData->SetValue(NewProp_Cpu_Score, TEXT("ModuleRelativePath"), TEXT("Pong_GameMode.h"));
#endif
			}
		}
		check(OuterClass->GetClass());
		return OuterClass;
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_APong_GameMode(Z_Construct_UClass_APong_GameMode, &APong_GameMode::StaticClass, TEXT("APong_GameMode"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(APong_GameMode);
	UClass* Z_Construct_UClass_AScreenCapturer_NoRegister()
	{
		return AScreenCapturer::StaticClass();
	}
	UClass* Z_Construct_UClass_AScreenCapturer()
	{
		static UClass* OuterClass = NULL;
		if (!OuterClass)
		{
			Z_Construct_UClass_AActor();
			Z_Construct_UPackage__Script_PongGame();
			OuterClass = AScreenCapturer::StaticClass();
			if (!(OuterClass->ClassFlags & CLASS_Constructed))
			{
				UObjectForceRegistration(OuterClass);
				OuterClass->ClassFlags |= 0x20900080;


PRAGMA_DISABLE_DEPRECATION_WARNINGS
				UProperty* NewProp_ScreenshotPeriod = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("ScreenshotPeriod"), RF_Public|RF_Transient|RF_MarkAsNative) UFloatProperty(CPP_PROPERTY_BASE(ScreenshotPeriod, AScreenCapturer), 0x0010000000000001);
				UProperty* NewProp_Width = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("Width"), RF_Public|RF_Transient|RF_MarkAsNative) UUnsizedIntProperty(CPP_PROPERTY_BASE(Width, AScreenCapturer), 0x0010000000000001);
				UProperty* NewProp_Height = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("Height"), RF_Public|RF_Transient|RF_MarkAsNative) UUnsizedIntProperty(CPP_PROPERTY_BASE(Height, AScreenCapturer), 0x0010000000000001);
				UProperty* NewProp_Screenshot = new(EC_InternalUseOnlyConstructor, OuterClass, TEXT("Screenshot"), RF_Public|RF_Transient|RF_MarkAsNative) UArrayProperty(CPP_PROPERTY_BASE(Screenshot, AScreenCapturer), 0x0010000000000001);
				UProperty* NewProp_Screenshot_Inner = new(EC_InternalUseOnlyConstructor, NewProp_Screenshot, TEXT("Screenshot"), RF_Public|RF_Transient|RF_MarkAsNative) UByteProperty(FObjectInitializer(), EC_CppProperty, 0, 0x0000000000000000);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
				OuterClass->StaticLink();
#if WITH_METADATA
				UMetaData* MetaData = OuterClass->GetOutermost()->GetMetaData();
				MetaData->SetValue(OuterClass, TEXT("IncludePath"), TEXT("ScreenCapturer.h"));
				MetaData->SetValue(OuterClass, TEXT("ModuleRelativePath"), TEXT("ScreenCapturer.h"));
				MetaData->SetValue(NewProp_ScreenshotPeriod, TEXT("Category"), TEXT("Screenshot"));
				MetaData->SetValue(NewProp_ScreenshotPeriod, TEXT("ModuleRelativePath"), TEXT("ScreenCapturer.h"));
				MetaData->SetValue(NewProp_Width, TEXT("Category"), TEXT("Screenshot"));
				MetaData->SetValue(NewProp_Width, TEXT("ModuleRelativePath"), TEXT("ScreenCapturer.h"));
				MetaData->SetValue(NewProp_Height, TEXT("Category"), TEXT("Screenshot"));
				MetaData->SetValue(NewProp_Height, TEXT("ModuleRelativePath"), TEXT("ScreenCapturer.h"));
				MetaData->SetValue(NewProp_Screenshot, TEXT("Category"), TEXT("Screenshot"));
				MetaData->SetValue(NewProp_Screenshot, TEXT("ModuleRelativePath"), TEXT("ScreenCapturer.h"));
#endif
			}
		}
		check(OuterClass->GetClass());
		return OuterClass;
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_AScreenCapturer(Z_Construct_UClass_AScreenCapturer, &AScreenCapturer::StaticClass, TEXT("AScreenCapturer"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(AScreenCapturer);
	UPackage* Z_Construct_UPackage__Script_PongGame()
	{
		static UPackage* ReturnPackage = NULL;
		if (!ReturnPackage)
		{
			ReturnPackage = CastChecked<UPackage>(StaticFindObjectFast(UPackage::StaticClass(), NULL, FName(TEXT("/Script/PongGame")), false, false));
			ReturnPackage->SetPackageFlags(PKG_CompiledIn | 0x00000000);
			FGuid Guid;
			Guid.A = 0x967C1713;
			Guid.B = 0xA2D7CC27;
			Guid.C = 0x00000000;
			Guid.D = 0x00000000;
			ReturnPackage->SetGuid(Guid);

		}
		return ReturnPackage;
	}
#endif

PRAGMA_ENABLE_DEPRECATION_WARNINGS
