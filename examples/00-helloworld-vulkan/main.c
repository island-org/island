#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdio.h>

int WIDTH = 800;
int HEIGHT = 600;

GLFWwindow *window;

void setup()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", NULL, NULL);
}

void draw()
{

}

void teardown()
{
    glfwDestroyWindow(window);

    glfwTerminate();
}

int main()
{
    setup();

    while (!glfwWindowShouldClose(window))
    {
        draw();
        glfwPollEvents();
    }

    teardown();
}

