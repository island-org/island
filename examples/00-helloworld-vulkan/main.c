#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdio.h>

int WIDTH = 800;
int HEIGHT = 600;

GLFWwindow *window;
VkInstance instance;

#if defined(DEBUG) || defined(_DEBUG)
#define V(x)           { vr = (x); if( vr != VK_SUCCESS ) { printf("%s %d %s %s", __FILE__, __LINE__, vr, L#x); } }
#else
#define V(x)           { vr = (x); }
#endif

VkResult vr = VK_SUCCESS;

void setup()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "helloworld-vulkan", NULL, NULL);

    VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    {
        appInfo.pApplicationName = "helloworld-vulkan";
        appInfo.pEngineName = "https://github.com/island-org/island";
        appInfo.apiVersion = VK_API_VERSION_1_0;
    }

    VkInstanceCreateInfo instInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        instInfo.pApplicationInfo = &appInfo;
        instInfo.ppEnabledExtensionNames = glfwExtensions;
        instInfo.enabledExtensionCount = glfwExtensionCount;
    }

    V(vkCreateInstance(&instInfo, NULL, &instance));
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

