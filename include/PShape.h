#ifndef PSHAPE_H
#define PSHAPE_H


typedef struct
{

} PShape;

PShape loadShape(const char* filename);
PShape createShape();
void shape(PShape shp);

#endif // PSHAPE_H

#ifdef PSHAPE_IMPLEMENTATION

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobjloader-c/tinyobj_loader_c.h"

PShape loadShape(const char* filename)
{
    PShape shp;

    return shp;
}

PShape createShape()
{
    PShape shp;

    return shp;
}

void shape(PShape pshape)
{
    
}

#endif // PSHAPE_IMPLEMENTATION

