-- http://industriousone.com/scripting-reference

local action = _ACTION or ""

local CUDA_PATH = os.getenv("CUDA_PATH");

solution "island"
    location ("_project")
    configurations { "Debug", "Release" }
    platforms {"native", "x64", "x32"}
    language "C"
    targetdir ("bin")
    kind "StaticLib"

    configuration "vs*"
        defines { "_CRT_SECURE_NO_WARNINGS" }

    configuration "Debug"
        targetdir ("bin")
        defines { "DEBUG" }
        flags { "Symbols"}
        targetsuffix "-d"

    configuration "Release"
        defines { "NDEBUG" }
        flags { "Optimize"}

    project "glfw"
        includedirs { "3rdparty/glfw/include" }
        files { 
            "3rdparty/glfw/include/GLFW/*.h",
            "3rdparty/glfw/src/context.c",
            "3rdparty/glfw/src/init.c",
            "3rdparty/glfw/src/input.c",
            "3rdparty/glfw/src/monitor.c",
            "3rdparty/glfw/src/window.c",
        }

        defines { "_GLFW_USE_OPENGL" }

        configuration "windows"
            defines { "_GLFW_WIN32", "_GLFW_WGL" }
            files {
                "3rdparty/glfw/src/win32*.c",
                "3rdparty/glfw/src/wgl_context.c",
                "3rdparty/glfw/src/winmm_joystick.c",
            }

    project "glew"
        files { 
            "3rdparty/glew/GL/*.h",
            "3rdparty/glew/*.c" 
        }
        defines "GLEW_STATIC"

    project "nanovg"
        files { "3rdparty/nanovg/src/*" }

    project "libuv"
        includedirs { "3rdparty/libuv/include" }
        files { 
            "3rdparty/libuv/include/*.h", 
            "3rdparty/libuv/src/*.c"
        }

        configuration "linux"
            files {
                "3rdparty/libuv/src/unix/*.c" 
            }

        configuration "windows"
            files {
                "3rdparty/libuv/src/win/*.c" 
            }

    project "lua"
        os.copyfile("3rdparty/lua/src/luaconf.h.orig", "3rdparty/lua/src/luaconf.h")
        includedirs { "3rdparty/lua/src" }
        files { 
            "3rdparty/lua/src/*.h",
            "3rdparty/lua/src/*.c"
        }
        excludes {
            "3rdparty/lua/src/loadlib_rel.c",
            "3rdparty/lua/src/lua.c",
            "3rdparty/lua/src/luac.c",
            "3rdparty/lua/src/print.c",
        }

    project "stb"
        includedirs { "3rdparty/stb" }
        files { 
            "3rdparty/stb/stb/*.h",
            "3rdparty/stb/*.h",
            "3rdparty/stb/*.c" 
        }

    project "AntTweakBar"
        language "C++"
        includedirs { 
            "3rdparty/AntTweakBar/include",
            "3rdparty/glew",
            "3rdparty/glfw/include",
        } 
        files {
            "3rdparty/AntTweakBar/include/*.h",
            "3rdparty/AntTweakBar/src/*",
        }

    project "blendish"
        language "C++"
        files { 
            "3rdparty/blendish/*.h",
            "3rdparty/blendish/blendish_lib.cpp" 
        }

    project "Remotery"
        files { 
            "3rdparty/Remotery/lib/*",
        }
        defines {
            "RMT_USE_OPENGL",
        }
        if CUDA_PATH ~= nil then
            includedirs { 
                path.join("$(CUDA_PATH)", "include"),
            }
            defines {
                "RMT_USE_CUDA",
            }
        end

    project "soloud"
        language "C++"
        includedirs {
            "3rdparty/soloud/include",
            "3rdparty/soloud/ext/libmodplug/src/libmodplug",
        }
        files { 
            "3rdparty/soloud/inlcude/*.h",
            "3rdparty/soloud/src/core/*.cpp",
            "3rdparty/soloud/src/audiosource/modplug/*.*",
            "3rdparty/soloud/src/audiosource/sfxr/*.*",
            "3rdparty/soloud/src/audiosource/speech/*.*",
            "3rdparty/soloud/src/audiosource/wav/*.*",
            "3rdparty/soloud/src/filter/*.cpp",
            "3rdparty/soloud/src/c_api/*.cpp",
            "3rdparty/soloud/ext/libmodplug/src/*.cpp",
        }
        defines {
            "WITH_MODPLUG",
            "MODPLUG_STATIC"
        }
        configuration "not windows"
            defines {"WITH_OSS"}
            files {
                "3rdparty/soloud/src/backend/oss/*.cpp" 
            }

        configuration "windows"
            defines {"WITH_WINMM"}
            files {
                "3rdparty/soloud/src/backend/winmm/*.cpp" 
            }

    function create_example_project( example_path )
        leaf_name = string.sub(example_path, string.len("examples/") + 1);

        project (leaf_name)
            kind "ConsoleApp"
            files { 
                "examples/" .. leaf_name .. "/*.h",
                "examples/" .. leaf_name .. "/*.lua",
                "examples/" .. leaf_name .. "/*.c",
                "examples/" .. leaf_name .. "/*.cu",
            }
            defines { 
                "GLEW_STATIC",
                "NANOVG_GL3_IMPLEMENTATION",
            }

            includedirs { 
                "include",
                "3rdparty",
                "3rdparty/glfw/include",
                "3rdparty/glew",
                "3rdparty/nanovg/src",
                "3rdparty/libuv/src",
                "3rdparty/lua/src",
                "3rdparty/stb",
                "3rdparty/AntTweakBar/include",
                "3rdparty/blendish",
                "3rdparty/soloud/include",
                "3rdparty/Remotery/lib",
            }

            libdirs {
                "bin",
            }

            -- TODO: automatically collect lib names
            configuration "Debug"
                links {
                    "glfw-d",
                    "glew-d",
                    "nanovg-d",
                    "libuv-d",
                    "lua-d",
                    "stb-d",
                    "AntTweakBar-d",
                    "blendish-d",
                    "soloud-d",
                    "Remotery-d",
                }

            configuration "Release"
                links {
                    "glfw",
                    "glew",
                    "nanovg",
                    "libuv",
                    "lua",
                    "stb",
                    "AntTweakBar",
                    "blendish",
                    "soloud",
                    "Remotery",
                }

            configuration "windows"
                links {
                    "OpenGL32",
                }

            if CUDA_PATH ~= nil then
                includedirs { 
                    path.join("$(CUDA_PATH)", "include"),
                }
                links {
                    "cuda.lib",
                    "nvrtc.lib"
                }
                configuration {"x32", "windows"}
                    libdirs {
                        path.join("$(CUDA_PATH)", "lib/win32"),
                    }
                configuration {"x64", "windows"}
                    libdirs {
                        path.join("$(CUDA_PATH)", "lib/x64"),
                    }
            end
    end

    local examples = os.matchdirs("examples/*")
    for _, example in ipairs(examples) do
        create_example_project(example)
    end

