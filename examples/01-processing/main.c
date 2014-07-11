#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

PImage img;

void setup()
{
    size(1024, 768);
    noCursor();
    img = loadImage("../3rdparty/nanovg/example/screenshot-01.png");
}

void draw()
{
    if (mousePressed)
    {
        image(img, mouseX, mouseY, img.width, img.height);
    }
    else
    {
        image(img, mouseX, mouseY, img.width / 2, img.height / 2);
    }

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
}

void shutdown()
{
    deleteImage(img);
}

