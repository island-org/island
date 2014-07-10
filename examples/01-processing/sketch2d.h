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
void triangle();

// Processing API - Shape - Attributes
void strokeCap(int cap); // NVG_BUTT (default), NVG_ROUND, NVG_SQUARE
void strokeJoin(int join); // NVG_MITER (default), NVG_ROUND, NVG_BEVEL
void strokeWeight(float weight);

// Processing API - Mouse
extern double mouseX, mouseY;
extern double pmouseX, pmouseY;
extern int mousePressed;
extern int mouseButton;

// Processing API - Keyboard


// Processing API - Transform 

// Processing API - Color
void background(struct NVGcolor color);
void fill(struct NVGcolor color);
void noFill();
void stroke(struct NVGcolor color);
void noStroke();

#endif // SKETCH_2D_H

#ifdef SKETCH_2D_IMPLEMENTATION

double mouseX, mouseY;
double pmouseX, pmouseY;
int width, height;

static GLFWwindow* window;
static int fbWidth, fbHeight;

static struct NVGcontext* vg;

static int isFill = 1;
static int isStroke = 1;

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

static void errorcb(int error, const char* desc)
{
    printf("GLFW error %d: %s\n", error, desc);
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

int main()
{
    GLenum err;

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

        glfwGetCursorPos(window, &mouseX, &mouseY);
        glfwGetWindowSize(window, &width, &height);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        // Calculate pixel ration for hi-dpi devices.
        pxRatio = (float)fbWidth / (float)width;

        // Update and render
        glViewport(0, 0, fbWidth, fbHeight);
        glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, backgroundColor.a);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

        nvgBeginFrame(vg, width, height, pxRatio);

        draw();

        nvgEndFrame(vg);

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
