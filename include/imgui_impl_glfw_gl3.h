// ImGui GLFW binding with OpenGL3 + shaders
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan graphics context creation, etc.)
// (GL3W is a helper library to access OpenGL functions since there is no standard header to access modern OpenGL functions easily. Alternatives are GLEW, Glad, etc.)

// Implemented features:
//  [X] User texture binding. Cast 'GLuint' OpenGL texture identifier as void*/ImTextureID. Read the FAQ about ImTextureID in imgui.cpp.
//  [X] Gamepad navigation mapping. Enable with 'io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad'.

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(), ImGui::Render() and ImGui_ImplXXXX_Shutdown().
// If you are new to ImGui, see examples/README.txt and documentation at the top of imgui.cpp.
// https://github.com/ocornut/imgui

struct GLFWwindow;

CIMGUI_API int         ImGui_ImplGlfwGL3_Init(struct GLFWwindow* window, int install_callbacks, const char* glsl_version);
CIMGUI_API void        ImGui_ImplGlfwGL3_Shutdown();
CIMGUI_API void        ImGui_ImplGlfwGL3_NewFrame();
CIMGUI_API void        ImGui_ImplGlfwGL3_RenderDrawData(struct ImDrawData* draw_data);

// Use if you want to reset your rendering device without losing ImGui state.
CIMGUI_API void        ImGui_ImplGlfwGL3_InvalidateDeviceObjects();
CIMGUI_API int         ImGui_ImplGlfwGL3_CreateDeviceObjects();

// GLFW callbacks (installed by default if you enable 'install_callbacks' during initialization)
// Provided here if you want to chain callbacks.
// You can also handle inputs yourself and use those as a reference.
CIMGUI_API void        ImGui_ImplGlfw_MouseButtonCallback(struct GLFWwindow* window, int button, int action, int mods);
CIMGUI_API void        ImGui_ImplGlfw_ScrollCallback(struct GLFWwindow* window, double xoffset, double yoffset);
CIMGUI_API void        ImGui_ImplGlfw_KeyCallback(struct GLFWwindow* window, int key, int scancode, int action, int mods);
CIMGUI_API void        ImGui_ImplGlfw_CharCallback(struct GLFWwindow* window, unsigned int c);
