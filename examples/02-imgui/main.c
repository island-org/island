#define SKETCH_2D_IMPLEMENTATION
#include "../01-processing/sketch2d.h"
#include <imgui.h>

PImage img1, img2;
PFont font;

void imguiRenderGLDraw(int width, int height)
{
#if 0
    const imguiGfxCmd* q = imguiGetRenderQueue();
    int nq = imguiGetRenderQueueSize();

    const float s = 1.0f/8.0f;

    glViewport(0, 0, width, height);
    glUseProgram(g_program);
    glActiveTexture(GL_TEXTURE0);
    glUniform2f(g_programViewportLocation, (float) width, (float) height);
    glUniform1i(g_programTextureLocation, 0);

    glDisable(GL_SCISSOR_TEST);
    for (int i = 0; i < nq; ++i)
    {
        const imguiGfxCmd& cmd = q[i];
        if (cmd.type == IMGUI_GFXCMD_RECT)
        {
            if (cmd.rect.r == 0)
            {
                drawRect((float)cmd.rect.x*s+0.5f, (float)cmd.rect.y*s+0.5f,
                    (float)cmd.rect.w*s-1, (float)cmd.rect.h*s-1,
                    1.0f, cmd.col);
            }
            else
            {
                drawRoundedRect((float)cmd.rect.x*s+0.5f, (float)cmd.rect.y*s+0.5f,
                    (float)cmd.rect.w*s-1, (float)cmd.rect.h*s-1,
                    (float)cmd.rect.r*s, 1.0f, cmd.col);
            }
        }
        else if (cmd.type == IMGUI_GFXCMD_LINE)
        {
            drawLine(cmd.line.x0*s, cmd.line.y0*s, cmd.line.x1*s, cmd.line.y1*s, cmd.line.r*s, 1.0f, cmd.col);
        }
        else if (cmd.type == IMGUI_GFXCMD_TRIANGLE)
        {
            if (cmd.flags == 1)
            {
                const float verts[3*2] =
                {
                    (float)cmd.rect.x*s+0.5f, (float)cmd.rect.y*s+0.5f,
                    (float)cmd.rect.x*s+0.5f+(float)cmd.rect.w*s-1, (float)cmd.rect.y*s+0.5f+(float)cmd.rect.h*s/2-0.5f,
                    (float)cmd.rect.x*s+0.5f, (float)cmd.rect.y*s+0.5f+(float)cmd.rect.h*s-1,
                };
                drawPolygon(verts, 3, 1.0f, cmd.col);
            }
            if (cmd.flags == 2)
            {
                const float verts[3*2] =
                {
                    (float)cmd.rect.x*s+0.5f, (float)cmd.rect.y*s+0.5f+(float)cmd.rect.h*s-1,
                    (float)cmd.rect.x*s+0.5f+(float)cmd.rect.w*s/2-0.5f, (float)cmd.rect.y*s+0.5f,
                    (float)cmd.rect.x*s+0.5f+(float)cmd.rect.w*s-1, (float)cmd.rect.y*s+0.5f+(float)cmd.rect.h*s-1,
                };
                drawPolygon(verts, 3, 1.0f, cmd.col);
            }
        }
        else if (cmd.type == IMGUI_GFXCMD_TEXT)
        {
            drawText(cmd.text.x, cmd.text.y, cmd.text.text, cmd.text.align, cmd.col);
        }
        else if (cmd.type == IMGUI_GFXCMD_SCISSOR)
        {
            if (cmd.flags)
            {
                glEnable(GL_SCISSOR_TEST);
                glScissor(cmd.rect.x, cmd.rect.y, cmd.rect.w, cmd.rect.h);
            }
            else
            {
                glDisable(GL_SCISSOR_TEST);
            }
        }
    }
    glDisable(GL_SCISSOR_TEST);
#endif
}

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
    unsigned char mbut = 0;
    int scrollarea1 = 0;

    background(gray(122));

    if (mousePressed)
    {
        image(img1, mouseX, mouseY, img1.width, img1.height);
    }
    else
    {
        image(img2, mouseX, mouseY, img2.width, img2.height);
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

    textFont(font);
    textAlign(NVG_ALIGN_CENTER);
    textSize(30);
    textLeading(5);
    text("test everything here", width/2, height/2);

    if (mousePressed && mouseButton == LEFT) mbut |= IMGUI_MBUT_LEFT;
    imguiBeginFrame(mouseX, mouseY, mbut, 0);

    imguiBeginScrollArea("Scroll area", 10, 10, width / 5, height - 20, &scrollarea1);
    imguiSeparatorLine();
    imguiSeparator();

    imguiButton("Button", GL_TRUE);
    imguiButton("Disabled button", GL_FALSE);
    imguiItem("Item", GL_TRUE);
    imguiItem("Disabled item", GL_FALSE);

    imguiEndScrollArea();

    imguiEndFrame();

    imguiRenderGLDraw(width, height); 
}

void shutdown()
{
}
