#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <nanovg.h>
#include <nanovg_gl.h>

#include <stdio.h>

GLenum err; 
GLFWwindow* window;

int main()
{
    glfwInit();
    window = glfwCreateWindow(800, 600, "helloworld", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return NULL;
    }
    glfwMakeContextCurrent(window);

    // glew
    err = glewInit();
    if (GLEW_OK != err)
    {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }


    glfwTerminate();
}