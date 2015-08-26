// A single window framework inspired by Processing (http://processing.org/)
// See official Processing reference, http://processing.org/reference/

#ifndef SKETCH_2D_H
#define SKETCH_2D_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <nanovg.h>
#include <stdio.h>
#include <stb/stb_vec.h>

#include "PAudio.h"

#ifdef _MSC_VER
#pragma warning(disable: 4244)  // conversion from 'float' to 'int', possible loss of data
#pragma warning(disable: 4305)  // 'initializing' : truncation from 'double' to 'float'
#endif

extern GLFWwindow* window;
//
// Processing framework - pure virtual functions that you must implement.
//
void setup();
void draw();
void shutdown();

//
// Structure
//
void pushStyle();
void popStyle();

//
// Math - Trigonometry
//
float degrees(float rad);
float radians(float deg);

// Color
struct NVGcolor color(unsigned char r, unsigned char g, unsigned char b);
struct NVGcolor colorA(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
struct NVGcolor gray(unsigned char v);
struct NVGcolor grayA(unsigned char v, unsigned char a);
struct NVGcolor lerpColor(struct NVGcolor c0, struct NVGcolor c1, float u);
float red(struct NVGcolor color);
float green(struct NVGcolor color);
float blue(struct NVGcolor color);
float alpha(struct NVGcolor color);

//
// Image
//
typedef struct
{
    int id;
    GLuint tex; // OpenGL texture id
    int width;
    int height;
} PImage;

// Supports jpg, png, psd, tga, pic and gif
PImage loadImage(const char* filename);
PImage createImage(int w, int h);
void updateImage(PImage img, const unsigned char* data);

void image(PImage img, int x, int y, int w, int h);

typedef struct
{
    int x, y;
    int w, h;
} Rect;
void imageEx(PImage img, Rect src, Rect dst);

// Supports png, tga, bmp
void saveFrame(const char* filename);

//
// Typography 
//
typedef struct
{
    int id;
} PFont;

PFont loadFont(const char* filename);
void textFont(PFont font);
void text(const char* string, float x, float y);

void textAlign(enum NVGalign align);
void textLeading(float spacing);
void textSize(float size);
void textBlur(float blur);

float textWidth(const char* string);

//
// Environment
//
void size(int winWidth, int winHeight);
void cursor();
void noCursor();
void quit();
extern int width, height;
extern int displayWidth, displayHeight;

//
// Shape - Vertex
//
void beginShape();
void endShape();
void endShapeClose(); // Connect the beginning and the end to close the shape
//void beginContour();
//void endContour();

void bezierVertex(float c1x, float c1y, float c2x, float c2y, float x, float y);
//void curveVertex(float x, float y); // Catmull-Rom spline
void quadraticVertex(float cx, float cy, float x, float y);
void vertex(float x, float y);

//
// Shape - 2D Primitives
//
void arc(float cx, float cy, float r, float a0, float a1, int dir);
void ellipse(float cx, float cy, float rx, float ry);
void line(float x1, float y1, float x2, float y2);
void point(float x, float y);
void quad(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4);
void rect(float x, float y, float w, float h);
void roundedRect(float x, float y, float w, float h, float r);
void triangle(float x1, float y1, float x2, float y2, float x3, float y3);

//
// Shape - Attributes
//
void strokeCap(int cap); // NVG_BUTT (default), NVG_ROUND, NVG_SQUARE
void strokeJoin(int join); // NVG_MITER (default), NVG_ROUND, NVG_BEVEL
void strokeWeight(float weight);

//
// Input - Mouse
//
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

//
// Input - Keyboard
//
extern int keyPressed;
extern int keyReleased;
extern int key;

//
// Input - Time & Date
//
double millis();
extern int frameCount;

//
// Math - Random
//
//float noise(float x);
//float noise(float x, float y);
float noise(float x, float y, float z);

//
// Transform 
//
void pushMatrix();
void popMatrix();
void printMatrix();
void resetMatrix();
void applyMatrix(float a, float b, float c, float d, float e, float f);

void translate(float x, float y);
void rotate(float angle);
void scale(float x, float y);
void shearX(float angle);
void shearY(float angle);

//
// Color
//
void background(struct NVGcolor color);
void fill(struct NVGcolor color);
void noFill();
void stroke(struct NVGcolor color);
void noStroke();

#endif // SKETCH_2D_H


#ifdef SKETCH_2D_IMPLEMENTATION

#include <nanovg_gl.h>
#include <stb/stb_image_write.h>
#include <stb/stb_perlin.h>

#define PAUDIO_IMPLEMENTATION
#include "PAudio.h"

float mouseX, mouseY;
float pmouseX, pmouseY;
int mousePressed;
int mouseButton;

int width, height;
int displayWidth, displayHeight;

GLFWwindow* window;
static int fbWidth, fbHeight;

static struct NVGcontext* vg;

static int isFill = 1;
static int isStroke = 1;

int keyPressed;
int keyReleased;
int key;

int frameCount;

double millis()
{
    return glfwGetTime() * 1000;
}

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
    static GLboolean sGlewInitialzed = GL_FALSE;
    GLenum err;

    if (window)
    {
        // Resize existed window
        glfwSetWindowSize(window, winWidth, winHeight);
        return;
    }

    //glfwWindowHint(GLFW_DECORATED, 0);
    window = glfwCreateWindow(winWidth, winHeight, "sketch", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(1);
    }

    glfwGetWindowSize(window, &width, &height);
    glfwMakeContextCurrent(window);

    puts(glGetString(GL_VERSION));
    puts(glGetString(GL_VENDOR));

    if (!sGlewInitialzed)
    {
        sGlewInitialzed = GL_TRUE;
        glewExperimental = GL_TRUE;
        err = glewInit();
        if (err != GLEW_OK)
        {
            printf("Error: %s\n", glewGetErrorString(err));
        }
        // GLEW generates GL error because it calls glGetString(GL_EXTENSIONS), we'll consume it here.
        glGetError();
    }

    vg = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
    if (vg == NULL)
    {
        printf("Could not init nanovg.\n");
        exit(1);
    }
}

void quit()
{
    glfwSetWindowShouldClose(window, 1);
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

void triangle(float x1, float y1, float x2, float y2, float x3, float y3)
{
    beginShape();
    vertex(x1, y1);
    vertex(x2, y2);
    vertex(x3, y3);
    endShapeClose();
}

void quad(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4)
{
    beginShape();
    vertex(x1, y1);
    vertex(x2, y2);
    vertex(x3, y3);
    vertex(x4, y4);
    endShapeClose();
}

void line(float x1, float y1, float x2, float y2)
{
    beginShape();
    vertex(x1, y1);
    vertex(x2, y2);
    endShape();
}

void point(float x, float y)
{
    ellipse(x, y, 1, 1);
}

float degrees(float rad)
{
    return nvgRadToDeg(rad);
}

float radians(float deg)
{
    return nvgDegToRad(deg);
}

PImage loadImage(const char* filename)
{
    PImage img =
    {
        nvgCreateImage(vg, filename, NVG_IMAGE_GENERATE_MIPMAPS)
    };
    if (img.id == 0)
    {
        printf("Could not load %s.\n", filename);
        exit(1);
    }

    nvgImageSize(vg, img.id, &img.width, &img.height);
    img.tex = nvglImageHandle(vg, img.id);

    return img;
}

PImage createImage(int w, int h)
{
    PImage img =
    {
        nvgCreateImageRGBA(vg, w, h, NVG_IMAGE_GENERATE_MIPMAPS, NULL),
        w,
        h
    };
    img.tex = nvglImageHandle(vg, img.id);

    return img;
}

void updateImage(PImage img, const unsigned char* data)
{
    nvgUpdateImage(vg, img.id, data);
}

void imageEx(PImage img, Rect src, Rect dst)
{
    struct NVGpaint paint = nvgImagePattern(vg, dst.x - src.x, dst.y - src.y, src.w, src.h, 0, img.id, 1);
    pushStyle();
    {
        noStroke();
        nvgFillPaint(vg, paint);
        rect(dst.x, dst.y, dst.w, dst.h);
    }
    popStyle();
}

void image(PImage img, int x, int y, int w, int h)
{
    Rect src = { 0, 0, img.width, img.height };
    Rect dst = { x, y, w, h };
    imageEx(img, src, dst);
}

void background(struct NVGcolor color)
{
    glClearColor(color.r, color.g, color.b, color.a);
    glClear(GL_COLOR_BUFFER_BIT);
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

void resetMatrix()
{
    nvgResetTransform(vg);
}

void applyMatrix(float a, float b, float c, float d, float e, float f)
{
    nvgTransform(vg, a, b, c, d, e, f);
}

void translate(float x, float y)
{
    nvgTranslate(vg, x, y);
}

void rotate(float angle)
{
    nvgRotate(vg, angle);
}

void scale(float x, float y)
{
    nvgScale(vg, x, y);
}

void shearX(float angle)
{
    nvgSkewX(vg, angle);
}

void shearY(float angle)
{
    nvgSkewY(vg, angle);
}

struct NVGcolor color(unsigned char r, unsigned char g, unsigned char b)
{
    return nvgRGB(r, g, b);
}

struct NVGcolor colorA(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
    return nvgRGBA(r, g, b, a);
}

struct NVGcolor gray(unsigned char v)
{
    return nvgRGB(v, v, v);
}

struct NVGcolor grayA(unsigned char v, unsigned char a)
{
    return nvgRGBA(v, v, v, a);
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
    const GLFWvidmode* vidMode;

    if (!glfwInit())
    {
        printf("Failed to init GLFW.");
        return -1;
    }

    glfwSetErrorCallback(onGlfwError);

#if 0
#ifndef ISLAND_GL_VERSION_MAJOR
#define ISLAND_GL_VERSION_MAJOR 4
#endif
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, ISLAND_GL_VERSION_MAJOR);

#ifndef ISLAND_GL_VERSION_MINOR
#define ISLAND_GL_VERSION_MINOR 5
#endif
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, ISLAND_GL_VERSION_MINOR);

#ifndef ISLAND_GL_FORWARD_COMPAT
#define ISLAND_GL_FORWARD_COMPAT GL_TRUE
#endif
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

#ifndef ISLAND_GL_PROFILE
#define ISLAND_GL_PROFILE GLFW_OPENGL_CORE_PROFILE
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, ISLAND_GL_PROFILE);

#ifndef ISLAND_GL_DEBUG_CONTEXT
#define ISLAND_GL_DEBUG_CONTEXT GL_TRUE
#endif
    // TODO: add debug callback if possible
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, ISLAND_GL_DEBUG_CONTEXT);
#endif

    size(1024, 768);

    glfwSetTime(0);

    vidMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    displayWidth = vidMode->width;
    displayHeight = vidMode->height;

    setupSoloud();
    setup();

    while (!glfwWindowShouldClose(window))
    {
        float pxRatio;
        int i;
        double mx, my;

        glfwGetWindowSize(window, &width, &height);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        // Calculate pixel ration for hi-dpi devices.
        pxRatio = (float)fbWidth / (float)width;

        // mouse event
        glfwGetCursorPos(window, &mx, &my);
        mouseX = (float)mx;
        mouseY = (float)my;

        mousePressed = 0;
        mouseButton = GLFW_KEY_UNKNOWN;
        for (i = 0; i < GLFW_MOUSE_BUTTON_LAST; i++)
        {
            if (glfwGetMouseButton(window, i) == GLFW_PRESS)
            {
                mousePressed = 1;
                mouseButton = i;
                break;
            }
        }

        // keyboard event
        if (keyPressed && glfwGetKey(window, key) == GLFW_RELEASE)
        {
            keyReleased = 1;
            keyPressed = 0;
        }
        else
        {
            keyReleased = 0;
            keyPressed = 0;
            key = GLFW_KEY_UNKNOWN;
            for (i = 0; i < GLFW_KEY_LAST; i++)
            {
                if (glfwGetKey(window, i) == GLFW_PRESS)
                {
                    keyPressed = 1;
                    key = i;
                    break;
                }
            }
        }

        // render
        glViewport(0, 0, fbWidth, fbHeight);
        glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        nvgBeginFrame(vg, width, height, pxRatio);

        if (keyPressed)
        {
            if (key == GLFW_KEY_ESCAPE)
            {
                quit();
            }
            else if (key == GLFW_KEY_SPACE)
            {
                saveFrame("screenshot.png");
            }
        }
        draw();

        nvgEndFrame(vg);

        // post render
        glfwSwapBuffers(window);
        glfwPollEvents();

        pmouseX = mouseX;
        pmouseY = mouseY;

        frameCount++;
    }

    shutdown();
    shutdownSoloud();

    nvgDeleteGL3(vg);
    glfwTerminate();
}

static void _flipHorizontal(unsigned char* image, int w, int h, int stride, int comp)
{
    int i = 0, j = h - 1, k;
    while (i < j) {
        unsigned char* ri = &image[i * stride];
        unsigned char* rj = &image[j * stride];
        for (k = 0; k < w*comp; k++) {
            unsigned char t = ri[k];
            ri[k] = rj[k];
            rj[k] = t;
        }
        i++;
        j--;
    }
}

void saveFrame(const char* name)
{
    const int comp = 3;
    int w = fbWidth;
    int h = fbHeight;
    unsigned char* image = (unsigned char*)malloc(w*h*comp);
    if (image == NULL)
        return;

    // Translate nvg commands to opengl functions and do the real rendering
    nvgEndFrame(vg);

    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, image);
    _flipHorizontal(image, w, h, w*comp, comp);

    if (strstr(name, ".png"))
    {
        stbi_write_png(name, w, h, comp, image, w*comp);
    }
    else if (strstr(name, ".tga"))
    {
        stbi_write_tga(name, w, h, comp, image);
    }
    else if (strstr(name, ".bmp"))
    {
        stbi_write_bmp(name, w, h, comp, image);
    }
    else
    {
        printf("Unsupported image format %s.\n", name);
    }
    free(image);
}

PFont loadFont(const char* filename)
{
    PFont font =
    {
        nvgCreateFont(vg, filename, filename),
    };
    if (font.id == -1)
    {
        printf("Can't load font %s.\n", filename);
        exit(1);
    }

    return font;
}

void text(const char* string, float x, float y)
{
    nvgText(vg, x, y, string, NULL);
}

void textFont(PFont font)
{
    nvgFontFaceId(vg, font.id);
}

void textAlign(enum NVGalign align)
{
    nvgTextAlign(vg, align);
}

void textLeading(float spacing)
{
    nvgTextLetterSpacing(vg, spacing);
}

void textSize(float size)
{
    nvgFontSize(vg, size);
}

void textBlur(float blur)
{
    nvgFontBlur(vg, blur);
}

float textWidth(const char* string)
{
    return nvgTextBounds(vg, 0, 0, string, NULL, NULL);
}

static GLboolean isInsideShape = GL_FALSE;
static GLboolean isFirstVertex = GL_TRUE;

void beginShape()
{
    if (isInsideShape)
    {
        printf("WARNING: endShape() is not called before beginShape().\n");
    }
    nvgBeginPath(vg);

    isInsideShape = GL_TRUE;
    isFirstVertex = GL_TRUE;
}

void endShape()
{
    if (!isInsideShape)
    {
        printf("WARNING: beginShape() is not called before endShape().\n");
    }

    if (isFill) nvgFill(vg);
    if (isStroke) nvgStroke(vg);

    isInsideShape = GL_FALSE;
}

void endShapeClose()
{
    nvgClosePath(vg);
    endShape();
}

void beginContour()
{

}

void endContour()
{

}

static GLboolean _checkFirstVertex(float x, float y)
{
    if (isFirstVertex)
    {
        isFirstVertex = GL_FALSE;
        nvgMoveTo(vg, x, y);

        return GL_TRUE;
    }
    return GL_FALSE;
}

void bezierVertex(float c1x, float c1y, float c2x, float c2y, float x, float y)
{
    if (!_checkFirstVertex(x, y))
    {
        nvgBezierTo(vg, c1x, c1y, c2x, c2y, x, y);
    }
}

void curveVertex(float x, float y)
{

}

void quadraticVertex(float cx, float cy, float x, float y)
{
    if (!_checkFirstVertex(x, y))
    {
        nvgQuadTo(vg, cx, cy, x, y);
    }
}

void vertex(float x, float y)
{
    if (!_checkFirstVertex(x, y))
    {
        nvgLineTo(vg, x, y);
    }
}

float noise(float x, float y, float z)
{
    return stb_perlin_noise3(x, y, z, 0, 0, 0);
}

#endif // SKETCH_2D_IMPLEMENTATION
