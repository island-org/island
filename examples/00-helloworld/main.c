#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <nanovg_gl.h>

#include <stdio.h>

GLenum err; 
GLFWwindow* window;
struct NVGcontext* vg;
int font;

static void errorcb(int error, const char* desc)
{
    printf("GLFW error %d: %s\n", error, desc);
}

int main()
{
    if (!glfwInit())
    {
        printf("Failed to init GLFW.");
        return -1;
    }

    glfwSetErrorCallback(errorcb);
#ifndef _WIN32 // don't require this on win32, and works with more cards
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
    window = glfwCreateWindow(800, 600, "helloworld", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return NULL;
    }
    glfwMakeContextCurrent(window);

    // glew
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        printf("Error: %s\n", glewGetErrorString(err));
        return -1;
    }
    // GLEW generates GL error because it calls glGetString(GL_EXTENSIONS), we'll consume it here.
    glGetError();

    vg = nvgCreateGL3(NVG_STENCIL_STROKES);
    if (vg == NULL)
    {
        printf("Could not init nanovg.\n");
        return -1;
    }

    while (!glfwWindowShouldClose(window))
    {
        double mx, my;
        int winWidth, winHeight;
        int fbWidth, fbHeight;
        float pxRatio;
        int i;

        glfwGetCursorPos(window, &mx, &my);
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        // Calculate pixel ration for hi-dpi devices.
        pxRatio = (float)fbWidth / (float)winWidth;

        // Update and render
        glViewport(0, 0, fbWidth, fbHeight);
        glClearColor(0.3f, 0.3f, 0.32f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

        nvgBeginFrame(vg, winWidth, winHeight, pxRatio);
        nvgEndFrame(vg);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    nvgDeleteGL3(vg);
    glfwTerminate();
}