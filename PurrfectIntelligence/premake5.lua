project "PurrfectIntelligence"
    kind "StaticLib"
    language "C++"
    targetdir ("%{wks.location}/bin/%{cfg.system}-%{cfg.architecture}/%{cfg.buildcfg}")
    cppdialect "C++17"

    files { "src/**.cpp", "src/**.h" }
    includedirs { "src/" }

    filter "configurations:Debug"
        defines { "PURR_INTELLIGENCE_DEBUG" }
        symbols "On"
        runtime "Debug"

    filter "configurations:Release"
        optimize "On"
        runtime "Release"