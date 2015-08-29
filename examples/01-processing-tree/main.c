// Ported from processing-2.2/modes/java/examples/Topics/Fractals and L-Systems/Tree/Tree.pde
// with minimal code changes.

#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

float theta;   

void setup()
{
    size(640, 360);
}

void branch(float h)
{
    // Each branch will be 2/3rds the size of the previous one
    h *= 0.66;

    // All recursive functions must have an exit condition!!!!
    // Here, ours is when the length of the branch is 2 pixels or less
    if (h > 2)
    {
        pushMatrix();    // Save the current state of transformation (i.e. where are we now)
        rotate(theta);   // Rotate by theta
        line(0, 0, 0, -h);  // Draw the branch
        translate(0, -h); // Move to the end of the branch
        branch(h);       // Ok, now call myself to draw two new branches!!
        popMatrix();     // Whenever we get back here, we "pop" in order to restore the previous matrix state

        // Repeat the same thing, only branch off to the "left" this time!
        pushMatrix();
        rotate(-theta);
        line(0, 0, 0, -h);
        translate(0, -h);
        branch(h);
        popMatrix();
    }
}

void draw()
{
    background(gray(0));
    stroke(gray(255));
    // Convert it to radians
    theta = radians(mouseX / width * 90);
    // Start the tree from the bottom of the screen
    translate(width/2,height);
    // Draw a line 120 pixels
    line(0,0,0,-120);
    // Move to the end of that line
    translate(0,-120);
    // Start the recursive branching!
    branch(120);
}

void teardown()
{

}