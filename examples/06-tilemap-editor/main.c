#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

PImage spriteSheet;

void STBTE_DRAW_RECT(int x0, int y0, int x1, int y1, unsigned int clr);
void STBTE_DRAW_TILE(int x0, int y0,
                     unsigned short id, int highlight, float *data);

#define STB_TILEMAP_EDITOR_IMPLEMENTATION
#include "stb/stb_tilemap_editor.h"

stbte_tilemap* tilemap;
double lastMillis;

const float SPRITE_SHEET_GRID = 22.5;

void setup()
{
    int i;
    size(800, 800);
#define MAP_W 200
#define MAP_H 200
#define MAP_LAYERS 2
#define MAX_TILES 10
    tilemap = stbte_create_map(MAP_W, MAP_H, MAP_LAYERS, SPRITE_SHEET_GRID, SPRITE_SHEET_GRID, MAX_TILES);
    if (tilemap == NULL)
    {
        quit();
        return;
    }

    for (i = 0; i < MAX_TILES; i++)
    {
        stbte_define_tile(tilemap, i, 0xff, "brick");
    }

    spriteSheet = loadImage("../media/Net_Hack_Sprite_sheet_v1_1_by_Chubbs_99.jpg");
    stbte_set_display(0, 0, width / 2, height / 2);
    lastMillis = millis();
}

void draw()
{
    int shifted = (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT);

    if (key == GLFW_KEY_ESCAPE) quit();

    stbte_mouse_move(tilemap, mouseX, mouseY, shifted, 0);
    stbte_mouse_button(tilemap, mouseX, mouseY, mouseButton == RIGHT, mousePressed, shifted, 0);
    stbte_mouse_wheel(tilemap, mouseX, mouseY, 0);

    background(gray(122));
    stbte_draw(tilemap);
    stbte_tick(tilemap, (millis() - lastMillis) * 0.001);
    lastMillis = millis();
}

void teardown()
{
    free(tilemap);
}

NVGcolor fromR8G8B8(unsigned int clr)
{
    unsigned char r = clr >> 16;
    unsigned char g = clr >> 8;
    unsigned char b = clr;
    return color(r, g, b);
}

void STBTE_DRAW_RECT(int x0, int y0, int x1, int y1, unsigned int clr)
{
    // this must draw a filled rectangle (exclusive on right/bottom)
    // color = (r<<16)|(g<<8)|(b)
    noStroke();
    fill(fromR8G8B8(clr));
    rect(x0, y0, x1 - x0, y1 - y0);
}

// this draws the tile image identified by 'id' in one of several
// highlight modes (see STBTE_drawmode_* in the header section);
// if 'data' is NULL, it's drawing the tile in the palette; if 'data'
// is not NULL, it's drawing a tile on the map, and that is the data
// associated with that map tile
void STBTE_DRAW_TILE(int x0, int y0,
                     unsigned short id, int highlight, float *data)
{
    Rect src = {id * SPRITE_SHEET_GRID, 0, spriteSheet.width, spriteSheet.height};
    Rect dst = {x0, y0, SPRITE_SHEET_GRID, SPRITE_SHEET_GRID};

    if (highlight == STBTE_drawmode_emphasize)
    {

    }

    if (data == NULL)
    {

    }

    imageEx(spriteSheet, src, dst);
}
