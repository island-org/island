-- http://industriousone.com/scripting-reference

local action = _ACTION or ""

solution "island"
    location (".project")
    configurations { "Debug", "Release" }
    platforms {"native", "x64", "x32"}
    language "C"
    targetdir ("bin")

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
        kind "StaticLib"
        includedirs { "3rdparty/glfw/include" }
        files { 
            "3rdparty/glfw/include/GLFW/*.h",
            "3rdparty/glfw/src/clipboard.c",
            "3rdparty/glfw/src/context.c",
            "3rdparty/glfw/src/gamma.c",
            "3rdparty/glfw/src/init.c",
            "3rdparty/glfw/src/input.c",
            "3rdparty/glfw/src/joystick.c",
            "3rdparty/glfw/src/monitor.c",
            "3rdparty/glfw/src/time.c",
            "3rdparty/glfw/src/window.c",
            "3rdparty/glfw/src/input.c",
            "3rdparty/glfw/src/input.c",
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
        kind "StaticLib"
        files { 
            "3rdparty/glew/GL/*.h",
            "3rdparty/glew/*.c" 
        }
        defines "GLEW_STATIC"

    project "nanovg"
        kind "StaticLib"
        files { "3rdparty/nanovg/src/*" }

    project "imgui"
        kind "StaticLib"
        files { 
            "3rdparty/imgui/imgui.h",
            "3rdparty/imgui/imgui.cpp" 
        }

    project "libuv"
        kind "StaticLib"
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
        kind "StaticLib"
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
        kind "StaticLib"
        includedirs { "3rdparty/stb" }
        files { 
            "3rdparty/stb/stb/*.h",
            "3rdparty/stb/*.c" 
        }

    function create_example_project( example_path )
        example_path = string.sub(example_path, string.len("examples/") + 1);
        project (example_path)
            kind "ConsoleApp"
            files { "examples/" .. example_path .. "/*" }
            defines { 
                "GLEW_STATIC",
                "NANOVG_GL3_IMPLEMENTATION"
            }
            includedirs { 
                "3rdparty",
                "3rdparty/glfw/include",
                "3rdparty/glew",
                "3rdparty/nanovg/src",
                "3rdparty/libuv/src",
                "3rdparty/lua/src",
                "3rdparty/stb"
            }

            libdirs {
                "bin"
            }

            configuration "Debug"
                links {
                    "glfw-d",
                    "glew-d",
                    "nanovg-d",
                    "imgui-d",
                    "libuv-d",
                    "lua-d",
                    "stb-d",
                }

            configuration "Release"
                links {
                    "glfw",
                    "glew",
                    "nanovg",
                    "imgui",
                    "libuv",
                    "lua",
                    "stb",
                }

            configuration "windows"
                links {
                    "OpenGL32"
                }
    end

    local examples = os.matchdirs("examples/*")
    for _, example in ipairs(examples) do
        create_example_project(example)
    end

