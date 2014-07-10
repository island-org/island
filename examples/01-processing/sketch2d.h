// A single window framework inspired by Processing (http://processing.org/)
// See official Processing reference, http://processing.org/reference/

#ifndef SKETCH_2D_H
#define SKETCH_2D_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <nanovg_gl.h>
#include <stdio.h>

// Processing framework - virtual functions that you must implement.
void setup();
void draw();
void shutdown();

// Processing API - Structure
void pushStyle();
void popStyle();
void pushMatrix();
void popMatrix();

// Processing API - Color
struct NVGcolor color(unsigned char r, unsigned char g, unsigned char b);
struct NVGcolor lerpColor(struct NVGcolor c0, struct NVGcolor c1, float u);
float red(struct NVGcolor color);
float green(struct NVGcolor color);
float blue(struct NVGcolor color);
float alpha(struct NVGcolor color);

// Processing API - Environment
void size(int winWidth, int winHeight);
void cursor();
void noCursor();
extern int width, height;

// Processing API - Shape - 2D Primitives
void arc(float cx, float cy, float r, float a0, float a1, int dir);
void ellipse(float cx, float cy, float rx, float ry);
void line();
void point();
void quad();
void rect(float x, float y, float w, float h);
void roundedRect(float x, float y, float w, float h, float r);
void triangle();

// Processing API - Shape - Attributes
void strokeCap(int cap); // NVG_BUTT (default), NVG_ROUND, NVG_SQUARE
void strokeJoin(int join); // NVG_MITER (default), NVG_ROUND, NVG_BEVEL
void strokeWeight(float weight);

// Processing API - Mouse
extern float mouseX, mouseY;
extern float pmouseX, pmouseY;
extern int mousePressed;
enum
{
    LEFT = GLFW_MOUSE_BUTTON_LEFT,
    RIGHT = GLFW_MOUSE_BUTTON_RIGHT,
    MIDDLE = GLFW_MOUSE_BUTTON_MIDDLE,
};
extern int mouseButton;

// Processing API - Keyboard
extern int keyPressed;
extern int key;

// Processing API - Transform 

// Processing API - Color
void background(struct NVGcolor color);
void fill(struct NVGcolor color);
void noFill();
void stroke(struct NVGcolor color);
void noStroke();

#endif // SKETCH_2D_H

#ifdef SKETCH_2D_IMPLEMENTATION

float mouseX, mouseY;
float pmouseX, pmouseY;
int mousePressed;
int mouseButton;

int width, height;

static GLFWwindow* window;
static int fbWidth, fbHeight;

static struct NVGcontext* vg;

static int isFill = 1;
static int isStroke = 1;

int keyPressed;
int key;

void cursor()
{
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void noCursor()
{
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
}

void size(int winWidth, int winHeight)
{
    if (window)
    {
        glfwDestroyWindow(window);
    }
    window = glfwCreateWindow(winWidth, winHeight, "sketch", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);
}

static void beginShape()
{
    nvgBeginPath(vg);
}

static void endShape()
{
    if (isFill) nvgFill(vg);
    if (isStroke) nvgStroke(vg);
}

void ellipse(float cx, float cy, float rx, float ry)
{
    beginShape();
    nvgEllipse(vg, cx, cy, rx, ry);
    endShape();
}

void rect(float x, float y, float w, float h)
{
    beginShape();
    nvgRect(vg, x, y, w, h);
    endShape();
}

void roundedRect(float x, float y, float w, float h, float r)
{
    beginShape();
    nvgRoundedRect(vg, x, y, w, h, r);
    endShape();
}

static struct NVGcolor backgroundColor = {0.5f, 0.5f, 0.5f, 1.0f};
void background(struct NVGcolor color)
{
    backgroundColor = color;
}

void fill(struct NVGcolor color)
{
    nvgFillColor(vg, color);
    isFill = 1;
}

void noFill()
{
    isFill = 0;
}

void stroke(struct NVGcolor color)
{
    nvgStrokeColor(vg, color);
    isStroke = 1;
}

void noStroke()
{
    isStroke = 0;
}

void strokeCap(int cap)
{
    nvgLineCap(vg, cap);
}

void strokeJoin(int join)
{
    nvgLineJoin(vg, join);
}

void strokeWeight(float weight)
{
    nvgStrokeWidth(vg, weight);
}

void pushStyle()
{
    nvgSave(vg);
}

void popStyle()
{
    nvgRestore(vg);
}

void pushMatrix()
{
    nvgSave(vg);
}

void popMatrix()
{
    nvgRestore(vg);
}

struct NVGcolor color(unsigned char r, unsigned char g, unsigned char b)
{
    return nvgRGB(r, g, b);
}

struct NVGcolor lerpColor(struct NVGcolor c0, struct NVGcolor c1, float u)
{
    return nvgLerpRGBA(c0, c1, u);
}

float red(struct NVGcolor color)
{
    return color.r;
}

float green(struct NVGcolor color)
{
    return color.g;
}

float blue(struct NVGcolor color)
{
    return color.b;
}

float alpha(struct NVGcolor color)
{
    return color.a;
}

static void onGlfwError(int error, const char* desc)
{
    printf("GLFW error %d: %s\n", error, desc);
}

int main()
{
    GLenum err;

    if (!glfwInit())
    {
        printf("Failed to init GLFW.");
        return -1;
    }

    glfwSetErrorCallback(onGlfwError);
#ifndef _WIN32 // don't require this on win32, and works with more cards
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
    size(800, 600);
    glfwMakeContextCurrent(window);

    // glew
    glewExperimental = GL_TRUE;
    err = glewInit();
    if (err != GLEW_OK)
    {
        printf("Error: %s\n", glewGetErrorString(err));
        return -1;
    }
    // GLEW generates GL error because it calls glGetString(GL_EXTENSIONS), we'll consume it here.
    glGetError();

    setup();

    vg = nvgCreateGL3(NVG_STENCIL_STROKES);
    if (vg == NULL)
    {
        printf("Could not init nanovg.\n");
        return -1;
    }

    while (!glfwWindowShouldClose(window))
    {
        float pxRatio;
        int i;
        double mx, my;

        const int kMouseKeys[] = 
        {
            GLFW_MOUSE_BUTTON_LEFT,
            GLFW_MOUSE_BUTTON_RIGHT,
            GLFW_MOUSE_BUTTON_MIDDLE
        };

        // pre render
        glfwGetCursorPos(window, &mx, &my);
        mouseX = (float)mx;
        mouseY = (float)my;

        glfwGetWindowSize(window, &width, &height);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        // Calculate pixel ration for hi-dpi devices.
        pxRatio = (float)fbWidth / (float)width;

        // mouse event
        mousePressed = 0;
        for (i=0; i<GLFW_MOUSE_BUTTON_LAST; i++)
        {
            if (glfwGetMouseButton(window, i) == GLFW_PRESS)
            {
                mousePressed = 1;
                mouseButton = i;
                break;
            }
        }

        // keyboard event
        keyPressed = 0;
        for (i=0; i<GLFW_KEY_LAST; i++)
        {
            if (glfwGetKey(window, i) == GLFW_PRESS)
            {
                keyPressed = 1;
                key = i;
            }
        }

        // render
        glViewport(0, 0, fbWidth, fbHeight);
        glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, backgroundColor.a);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

        nvgBeginFrame(vg, width, height, pxRatio);

        draw();

        nvgEndFrame(vg);

        // post render
        glfwSwapBuffers(window);
        glfwPollEvents();

        pmouseX = mouseX;
        pmouseY = mouseY;
    }

    shutdown();

    nvgDeleteGL3(vg);
    glfwTerminate();
}

#endif // SKETCH_2D_IMPLEMENTATION
