#define SKETCH_2D_IMPLEMENTATION
#include "../01-processing/sketch2d.h"

#include <AntTweakBar.h>

TwBar* bar;
double time;
double speed = 0.3; // Model rotation speed
int wire = 0;       // Draw model in wireframe?
struct NVGcolor bgColor;         // Background color 
struct NVGcolor rectColor;         // Background color 

void WindowSizeCB(GLFWwindow* window, int width, int height)
{
    TwWindowSize(width, height);
}

void setup()
{
    // Initialize AntTweakBar
    TwInit(TW_OPENGL_CORE, NULL);

    bgColor = color(100, 100, 100);
    rectColor = colorA(255, 0, 0, 200);

    // Create a tweak bar
    bar = TwNewBar("TweakBar");
    WindowSizeCB(window, width, height);

    glfwSetWindowSizeCallback(window, WindowSizeCB);
    glfwSetMouseButtonCallback(window, TwEventMouseButtonGLFW);
    glfwSetCursorPosCallback(window, TwEvenCursorPosGLFW);
    glfwSetScrollCallback(window, TwEventScrollGLFW);
    glfwSetKeyCallback(window, TwEventKeyGLFW);
    glfwSetCharCallback(window, TwEventCharGLFW);

    TwDefine(" GLOBAL help='This example shows how to integrate AntTweakBar with GLFW and OpenGL.' "); // Message added to the help bar.
    // Add 'speed' to 'bar': it is a modifable (RW) variable of type TW_TYPE_DOUBLE. Its key shortcuts are [s] and [S].
    TwAddVarRW(bar, "speed", TW_TYPE_DOUBLE, &speed, 
        " label='Rot speed' min=0 max=2 step=0.01 keyIncr=s keyDecr=S help='Rotation speed (turns/second)' ");

    // Add 'wire' to 'bar': it is a modifable variable of type TW_TYPE_BOOL32 (32 bits boolean). Its key shortcut is [w].
    TwAddVarRW(bar, "wire", TW_TYPE_BOOL32, &wire, 
        " label='Wireframe mode' key=w help='Toggle wireframe display mode.' ");

    // Add 'time' to 'bar': it is a read-only (RO) variable of type TW_TYPE_DOUBLE, with 1 precision digit
    TwAddVarRO(bar, "time", TW_TYPE_DOUBLE, &time, " label='Time' precision=1 help='Time (in seconds).' ");         

    // Add 'bgColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR3F (3 floats color)
    TwAddVarRW(bar, "bgColor", TW_TYPE_COLOR3F, bgColor.rgba, " label='Background color' ");

    // Add 'rectColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR32 (32 bits color) with alpha
    TwAddVarRW(bar, "rectColor", TW_TYPE_COLOR4F, rectColor.rgba, 
        " label='Rect color' alpha help='Color and transparency of the rect.' ");

}

void draw()
{
    if (keyPressed && key == GLFW_KEY_ESCAPE) quit();

    background(bgColor);
    fill(rectColor);
    rect(width / 2, height / 2, 200, 200);

    time = millis() * 0.001;
    TwDraw();

    if (keyPressed && key == GLFW_KEY_SPACE) saveFrame("screenshot.png");
}

void shutdown()
{
    TwTerminate();
}
