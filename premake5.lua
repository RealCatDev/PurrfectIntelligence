workspace "PurrfectIntelligence"
    configurations { "Debug", "Release" }
    architecture "x86_64"
	startproject "Test"

project "PurrfectIntelligence"
    kind "StaticLib"
    language "C++"
    targetdir ("%{wks.location}/bin/%{cfg.system}-%{cfg.architecture}/%{cfg.buildcfg}")
    cppdialect "C++17"

    files { "$(SolutionDir)/PurrfectIntelligence/src/**.cpp", "$(SolutionDir)/PurrfectIntelligence/src/**.h" }
    includedirs { "$(SolutionDir)/PurrfectIntelligence/src/" }

    filter "configurations:Debug"
        defines { "PURR_INTELLIGENCE_DEBUG" }
        symbols "On"
        runtime "Debug"

    filter "configurations:Release"
        optimize "On"
        runtime "Release"

project "Test"
    kind "ConsoleApp"
    language "C++"
    targetdir ("%{wks.location}/bin/%{cfg.system}-%{cfg.architecture}/%{cfg.buildcfg}")
    cppdialect "C++17"

    files { "Test/src/**.cpp", "Test/src/**.h" }

    includedirs { "$(SolutionDir)/PurrfectIntelligence/src/", "Test/src/" }

    libdirs { "%{wks.location}/bin/%{cfg.system}-%{cfg.architecture}/%{cfg.buildcfg}/" }

    links { "PurrfectIntelligence" }

    filter "configurations:Debug"
        symbols "On"
        runtime "Debug"

    filter "configurations:Release"
        runtime "Release"