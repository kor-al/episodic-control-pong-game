// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;

public class PongGame : ModuleRules
{
	public PongGame(TargetInfo Target)
	{
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "RenderCore", "OpenCV" });

		PrivateDependencyModuleNames.AddRange(new string[] {  });

        // Uncomment if you are using Slate UI
        PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

        PublicAdditionalLibraries.Add(@"D:/Alice/Documents/Unreal Projects/PongGame - Copy/Binaries/Win64/opencvtest.lib");

        PublicAdditionalLibraries.Add(@"D:/Python35/libs/python35.lib");

        //PublicIncludePaths.Add(@"C:/PathToOpenCV/build/include");
        //PublicAdditionalLibraries.Add(@"C:/PathToOpenCV/build/x64/vc12/lib/opencv_world300d.lib");

        //PublicIncludePaths.Add(@"D:/Alice/Documents/Unreal Projects/PongGame - Copy/ThirdParty/OpenCV/Includes");
        //PublicAdditionalLibraries.Add(@"D:/Alice/Documents/Unreal Projects/PongGame - Copy/ThirdParty/OpenCV/Libraries/Win64/opencv_world300d.lib");


        // Uncomment if you are using online features
        // PrivateDependencyModuleNames.Add("OnlineSubsystem");

        // To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
    }
}
