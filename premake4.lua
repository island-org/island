
local action = _ACTION or ""

solution "island"
	location ( "build" )
	configurations { "Debug", "Release" }
	platforms {"native", "x64", "x32"}

    project "glfw"
        language "C"
        kind "StaticLib"
        includedirs { "3rdparty/glfw/include" }
        files
        { 
            "3rdparty/glfw/src/win32*.c",
            "3rdparty/glfw/src/clipboard.c",
            "3rdparty/glfw/src/context.c",
            "3rdparty/glfw/src/gamma.c",
            "3rdparty/glfw/src/init.c",
            "3rdparty/glfw/src/input.c",
            "3rdparty/glfw/src/joystick.c",
            "3rdparty/glfw/src/monitor.c",
            "3rdparty/glfw/src/time.c",
            "3rdparty/glfw/src/wgl_context.c",
            "3rdparty/glfw/src/window.c",
            "3rdparty/glfw/src/winmm_joystick.c",
            "3rdparty/glfw/src/input.c",
            "3rdparty/glfw/src/input.c",
        }
        targetdir("build")

        configuration "Debug"
            defines { "DEBUG", "_GLFW_WIN32", "_GLFW_WGL", "_GLFW_USE_OPENGL" }
            flags { "Symbols"}

        configuration "Release"
            defines { "NDEBUG", "_GLFW_WIN32", "_GLFW_WGL", "_GLFW_USE_OPENGL" }
            flags { "Optimize"}

    project "glew"
        language "C"
        kind "StaticLib"
        includedirs { "3rdparty/glew" }
        files { "3rdparty/glew/*.c" }
        targetdir("build")

        configuration "Debug"
            defines { "DEBUG", "GLEW_STATIC" }
            flags { "Symbols", "ExtraWarnings"}

        configuration "Release"
            defines { "NDEBUG", "GLEW_STATIC" }
            flags { "Optimize", "ExtraWarnings"}

    project "nanovg"
        language "C"
        kind "StaticLib"
        includedirs { "3rdparty/nanovg/src" }
        files { "3rdparty/nanovg/src/*.c" }
        targetdir("build")
        --defines { "FONS_USE_FREETYPE" }   -- Uncomment to compile with FreeType support

        configuration "Debug"
            defines { "DEBUG" }
            flags { "Symbols" }

        configuration "Release"
            defines { "NDEBUG" }
            flags { "Optimize" }

    project "libuv"
        language "C"
        kind "StaticLib"
        includedirs { "3rdparty/libuv/include" }
        files { "3rdparty/libuv/src/*.c", "3rdparty/libuv/src/win/*.c" }
        targetdir("build")

        configuration "Debug"
            defines { "DEBUG" }
            flags { "Symbols" }

        configuration "Release"
            defines { "NDEBUG" }
            flags { "Optimize" }

    -- project "lua"
    --     language "C"
    --     kind "StaticLib"
    --     includedirs { "3rdparty/lua/src" }
    --     files { "3rdparty/lua/src/*.c"}
    --     targetdir("build")

    --     configuration "Debug"
    --         defines { "DEBUG" }
    --         flags { "Symbols" }

    --     configuration "Release"
    --         defines { "NDEBUG" }
    --         flags { "Optimize" }
