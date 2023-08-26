project "Test"
    kind "ConsoleApp"
    language "C++"
    targetdir ("%{wks.location}/bin/%{cfg.system}-%{cfg.architecture}/%{cfg.buildcfg}")
    cppdialect "C++17"

    files { "src/**.cpp", "src/**.h" }

    includedirs { "$(SolutionDir)/PurrfectIntelligence/src/", "src/" }

    links { "PurrfectIntelligence" }

    filter "configurations:Debug"
        symbols "On"
        runtime "Debug"

    filter "configurations:Release"
        runtime "Release"