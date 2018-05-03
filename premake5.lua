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
            path.join(action, "x86")
        }
        targetdir (path.join(action, "x86"))

    configuration "x64"
        libdirs {
            path.join(action, "x64"),
        }
        targetdir (path.join(action, "x64"))

    filter "system:macosx"
        defines {
            "_MACOSX",
        }

    configuration "Debug"
        defines { "DEBUG" }
        symbols "On"
        targetsuffix "-d"

    configuration "Release"
        defines { "NDEBUG" }
        flags { "No64BitChecks" }
        editandcontinue "Off"
        optimize "Speed"
        optimize "On"
        editandcontinue "Off"

    project "imgui"
        includedirs {
            "3rdparty/cimgui/cimgui",
            "3rdparty/cimgui/imgui",
        }
        files { 
            "3rdparty/cimgui/cimgui/*.h",
            "3rdparty/cimgui/cimgui/*.cpp",
            "3rdparty/cimgui/imgui/*",
        }

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
                "3rdparty/glfw/src/win32*",
                "3rdparty/glfw/src/egl_context*",
                "3rdparty/glfw/src/wgl_context*",
            }
        filter "system:macosx"
            defines {
                "_GLFW_COCOA",
            }
            files {
                "3rdparty/glfw/src/cocoa*",
                "3rdparty/glfw/src/nsgl*",
                "3rdparty/glfw/src/posix*",
            }
        filter "system:linux"
            defines {
                "_GLFW_X11",
            }
            files {
                "3rdparty/glfw/src/linux*",
                "3rdparty/glfw/src/glx*",
                "3rdparty/glfw/src/x11*",
                "3rdparty/glfw/src/egl_context*",
                "3rdparty/glfw/src/posix*",
                "3rdparty/glfw/src/xkb*",
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
        defines {
            "GLEW_STATIC",
            "GLEW_NO_GLU",
        }

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

        filter "system:linux"
            files {
                "3rdparty/libuv/src/unix/async.c",
                "3rdparty/libuv/src/unix/atomic-ops.h",
                "3rdparty/libuv/src/unix/core.c",
                "3rdparty/libuv/src/unix/dl.c",
                "3rdparty/libuv/src/unix/fs.c",
                "3rdparty/libuv/src/unix/getaddrinfo.c",
                "3rdparty/libuv/src/unix/getnameinfo.c",
                "3rdparty/libuv/src/unix/internal.h",
                "3rdparty/libuv/src/unix/loop-watcher.c",
                "3rdparty/libuv/src/unix/loop.c",
                "3rdparty/libuv/src/unix/pipe.c",
                "3rdparty/libuv/src/unix/poll.c",
                "3rdparty/libuv/src/unix/process.c",
                "3rdparty/libuv/src/unix/signal.c",
                "3rdparty/libuv/src/unix/spinlock.h",
                "3rdparty/libuv/src/unix/stream.c",
                "3rdparty/libuv/src/unix/tcp.c",
                "3rdparty/libuv/src/unix/thread.c",
                "3rdparty/libuv/src/unix/timer.c",
                "3rdparty/libuv/src/unix/tty.c",
                "3rdparty/libuv/src/unix/udp.c",

                "3rdparty/libuv/src/unix/linux-core.c",
                "3rdparty/libuv/src/unix/linux-inotify.c",
                "3rdparty/libuv/src/unix/linux-syscalls.c",
                "3rdparty/libuv/src/unix/linux-syscalls.h",
                "3rdparty/libuv/src/unix/procfs-exepath.c",
                "3rdparty/libuv/src/unix/proctitle.c",
                "3rdparty/libuv/src/unix/sysinfo-loadavg.c",
                "3rdparty/libuv/src/unix/sysinfo-memory.c",
            }

        filter "system:windows"
            files {
                "3rdparty/libuv/src/win/*.c" 
            }

        filter "system:macosx"
            files {
                "3rdparty/libuv/src/unix/async.c",
                "3rdparty/libuv/src/unix/atomic-ops.h",
                "3rdparty/libuv/src/unix/core.c",
                "3rdparty/libuv/src/unix/dl.c",
                "3rdparty/libuv/src/unix/fs.c",
                "3rdparty/libuv/src/unix/getaddrinfo.c",
                "3rdparty/libuv/src/unix/getnameinfo.c",
                "3rdparty/libuv/src/unix/internal.h",
                "3rdparty/libuv/src/unix/loop-watcher.c",
                "3rdparty/libuv/src/unix/loop.c",
                "3rdparty/libuv/src/unix/pipe.c",
                "3rdparty/libuv/src/unix/poll.c",
                "3rdparty/libuv/src/unix/process.c",
                "3rdparty/libuv/src/unix/signal.c",
                "3rdparty/libuv/src/unix/spinlock.h",
                "3rdparty/libuv/src/unix/stream.c",
                "3rdparty/libuv/src/unix/tcp.c",
                "3rdparty/libuv/src/unix/thread.c",
                "3rdparty/libuv/src/unix/timer.c",
                "3rdparty/libuv/src/unix/tty.c",
                "3rdparty/libuv/src/unix/udp.c",

                "3rdparty/libuv/src/unix/bsd-ifaddrs.c",
                "3rdparty/libuv/src/unix/darwin.c",
                "3rdparty/libuv/src/unix/darwin-proctitle.c",
                "3rdparty/libuv/src/unix/fsevents.c",
                "3rdparty/libuv/src/unix/kqueue.c",
                "3rdparty/libuv/src/unix/proctitle",
            }
            
    project "island"
        includedirs {
            "include",
            "3rdparty/stb",
            "3rdparty/glfw/include",
            "3rdparty/glew",
            "3rdparty/cimgui/cimgui",
            "3rdparty/cimgui/imgui",
        }
        files { 
            "3rdparty/stb/*",
            "3rdparty/stb/stb/*",
            "include/**",
            "src/**",
        }
        defines {
            "GLEW_NO_GLU",
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
                "GLEW_NO_GLU",
                "NANOVG_GL3_IMPLEMENTATION",
                "RMT_USE_OPENGL",
                "CIMGUI_DEFINE_ENUMS_AND_STRUCTS",
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
                "3rdparty/cimgui/cimgui",
            }

            libdirs {
                "lib/vulkan"
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
                    "imgui-d",
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
                    "imgui",
                }

            filter "system:windows"
                links {
                    "OpenGL32",
                    "Psapi",
                    "Iphlpapi",
                    "Userenv",
                    "vulkan-1",
                }
            filter "system:macosx"
                linkoptions {
                    "-framework Cocoa",
                    "-framework QuartzCore",
                    "-framework IOKit",
                    "-framework OpenGL",
                    "-framework AudioToolbox",
                }

            filter "system:linux"
                links {
                    "GL",
                    "X11",
                    "m",
                    "pthread",
                    "dl",
                    "vulkan",
                    "stdc++",
                }
            if CUDA_PATH ~= nil then
                includedirs { 
                    path.join("$(CUDA_PATH)", "include"),
                }

                configuration {"x86", "windows"}
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

