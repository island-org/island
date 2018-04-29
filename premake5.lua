-- http://industriousone.com/scripting-reference

local action = _ACTION or ""

local CUDA_PATH = os.getenv("CUDA_PATH");

solution "island"
    location (action)
    configurations { "Debug", "Release" }
    platforms {"x64", "x86"}
    language "C"
    kind "StaticLib"

    filter "action:vs*"
        defines { "_CRT_SECURE_NO_WARNINGS" }

    configuration "x86"
        libdirs {
            "x86",
        }
        targetdir ("x86")

    configuration "x64"
        libdirs {
            "x64",
        }
        targetdir ("x64")

    filter "system:macosx"
        defines {
            "_MACOSX",
        }

    configuration "Debug"
        defines { "DEBUG" }
        flags { "Symbols"}
        targetsuffix "-d"

    configuration "Release"
        defines { "NDEBUG" }
    	flags { "Optimize", "OptimizeSpeed", "NoEditAndContinue", "No64BitChecks" }

    project "glfw"
        includedirs {
            "3rdparty/glfw/include"
        }
        files { 
            "3rdparty/glfw/include/GLFW/*.h",
            "3rdparty/glfw/src/context.c",
            "3rdparty/glfw/src/init.c",
            "3rdparty/glfw/src/input.c",
            "3rdparty/glfw/src/monitor.c",
            "3rdparty/glfw/src/window.c",
            "3rdparty/glfw/src/vulkan.c",
            "3rdparty/glfw/src/osmesa_context.*",
        }

        defines { "_GLFW_USE_OPENGL" }

        filter "system:windows"
            defines { "_GLFW_WIN32", }
            files {
                "3rdparty/glfw/src/win32_*",
                "3rdparty/glfw/src/egl_context.*",
                "3rdparty/glfw/src/wgl_context.*",
            }
        filter "system:macosx"
            defines {
                "_GLFW_COCOA",
            }
            files {
                "3rdparty/glfw/src/cocoa_*",
                "3rdparty/glfw/src/nsgl_*",
                "3rdparty/glfw/src/posix_thread.*",
            }

    project "v7"
        files { 
            "3rdparty/v7/*.h",
            "3rdparty/v7/*.c" 
        }

    project "glew"
        includedirs {
            "3rdparty/glew",
        }
        files { 
            "3rdparty/glew/GL/*.h",
            "3rdparty/glew/*.c" 
        }
        defines "GLEW_STATIC"

    project "nanovg"
        files { "3rdparty/nanovg/src/*" }

    project "libuv"
        includedirs {
            "3rdparty/libuv/include",
            "3rdparty/libuv/src",
        }
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

    project "island"
        includedirs {
            "3rdparty/stb"
        }
        files { 
            "3rdparty/stb/*",
            "3rdparty/stb/stb/*",
            "src/**",
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

    -- project "soloud"
    --     language "C++"
    --     includedirs {
    --         "3rdparty/soloud/include",
    --     }
    --     files { 
    --         "3rdparty/soloud/inlcude/*.h",
    --         "3rdparty/soloud/src/core/*.cpp",
    --         "3rdparty/soloud/src/audiosource/**",
    --         "3rdparty/soloud/src/filter/*.cpp",
    --         "3rdparty/soloud/src/c_api/*.cpp",
    --     }
    --     filter "system:windows"
    --         defines {"WITH_WINMM"}
    --         files {
    --             "3rdparty/soloud/src/backend/winmm/*.cpp" 
    --         }
    --     filter "system:linux"
    --         defines {"WITH_OSS"}
    --         files {
    --             "3rdparty/soloud/src/backend/oss/*.cpp" 
    --         }            
    --     filter "system:macosx"
    --         defines {
    --             "WITH_COREAUDIO",
    --         }
    --         files {
    --             "3rdparty/soloud/src/backend/coreaudio/*.cpp" 
    --         }

    project "nativefiledialog"
        includedirs {
            "3rdparty/nativefiledialog/src/include",
        }
        files { 
            "3rdparty/nativefiledialog/src/include/**",
            "3rdparty/nativefiledialog/src/*.h",
            "3rdparty/nativefiledialog/src/nfd_common.c",
        }
        filter "system:windows"
            language "C++"
            files { 
                "3rdparty/nativefiledialog/src/nfd_win.cpp",
            }
        filter "system:linux"
            files { 
                "3rdparty/nativefiledialog/src/nfd_gtk.c",
            }
            buildoptions {"`pkg-config --cflags gtk+-3.0`"}
        filter "system:macosx"
            files { 
                "3rdparty/nativefiledialog/src/nfd_cocoa.m",
            }

    function create_example_project( example_path )
        leaf_name = string.sub(example_path, string.len("examples/") + 1);

        project (leaf_name)
            kind "ConsoleApp"
            files { 
                "examples/" .. leaf_name .. "/*.h",
                "examples/" .. leaf_name .. "/*.js",
                "examples/" .. leaf_name .. "/*.c*",
            }
            defines { 
                "GLEW_STATIC",
                "NANOVG_GL3_IMPLEMENTATION",
                "RMT_USE_OPENGL",
            }

            includedirs { 
                "include",
                "3rdparty",
                "3rdparty/glfw/include",
                "3rdparty/glew",
                "3rdparty/nanovg/src",
                "3rdparty/libuv/include",
                "3rdparty/v7",
                "3rdparty/stb",
                "3rdparty/soloud/include",
                "3rdparty/Remotery/lib",
            }

            -- TODO: automatically collect lib names
            configuration "Debug"
                links {
                    "glfw-d",
                    "glew-d",
                    "nanovg-d",
                    "libuv-d",
                    "island-d",
                    -- "soloud-d",
                    "Remotery-d",
                    "v7-d",
                }

            configuration "Release"
                links {
                    "glfw",
                    "glew",
                    "nanovg",
                    "libuv",
                    "island",
                    -- "soloud",
                    "Remotery",
                    "v7",
                }

            configuration "windows"
                links {
                    "OpenGL32",
                    "Psapi",
                    "Iphlpapi",
                    "Userenv",
                }
            configuration "macosx"
                linkoptions {
                    "-framework Cocoa",
                    "-framework QuartzCore",
                    "-framework IOKit",
                    "-framework OpenGL",
                    "-framework AudioToolbox",
                }

            if CUDA_PATH ~= nil then
                includedirs { 
                    path.join("$(CUDA_PATH)", "include"),
                }

                configuration {"x32", "windows"}
                    links {
                        "cuda",
                        "cudart",
                    }
                    libdirs {
                        path.join("$(CUDA_PATH)", "lib/win32"),
                    }
                configuration {"x64", "windows"}
                    links {
                        "cuda",
                        "cudart",
                    }
                    libdirs {
                        path.join("$(CUDA_PATH)", "lib/x64"),
                    }
                defines {
                    "RMT_USE_CUDA",
                }
            end
    end

    local examples = os.matchdirs("examples/*")
    for _, example in ipairs(examples) do
        create_example_project(example)
    end

