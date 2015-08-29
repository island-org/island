#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

PImage img1, img2;
PFont font;

void setup()
{
    size(displayWidth, displayHeight);
    noCursor();
    img1 = loadImage("../3rdparty/nanovg/example/images/image9.jpg");
    img2 = loadImage("../3rdparty/nanovg/example/images/image10.jpg");
    font = loadFont("../3rdparty/nanovg/example/Roboto-Regular.ttf");
}

void draw()
{
    background(gray(122));

    if (mousePressed)
    {
        image(img1, mouseX, mouseY, img1.width, img1.height);
    }
    else
    {
        image(img2, mouseX, mouseY, img2.width, img2.height);
    }

    textFont(font);
    textAlign(NVG_ALIGN_CENTER);
    textSize(30);
    textLeading(5);
    text("test everything here", width/2, height/2);
}

void teardown()
{
}
