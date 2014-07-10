#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

void setup()
{
    size(600, 600);
    noCursor();
}

void draw()
{
    if (keyPressed && key == GLFW_KEY_ESCAPE)
    {
        ellipse(mouseX, mouseY, 100, 100);
    }

    if (mousePressed)
    {
        fill(color(mouseButton == LEFT ? 255 : 0, 0, 0));
    }
    else
    {
        noFill();
    }
    ellipse(mouseX, mouseY, 10, 10);
}

void shutdown()
{

}

