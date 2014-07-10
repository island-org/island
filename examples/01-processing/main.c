#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

void setup()
{
    size(600, 600);
    noCursor();
}

void draw()
{
    ellipse(mouseX, mouseY, 10, 10);
}

void shutdown()
{

}

