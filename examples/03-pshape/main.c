#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

#define PSHAPE_IMPLEMENTATION
#include "PShape.h"

PShape shp;

void setup()
{
    size(displayWidth, displayHeight);
    noCursor();
    shp = loadShape("../media/cerberus/Cerberus.obj");
}

void draw()
{
    background(gray(122));

    shape(shp);
}

void teardown()
{
}
