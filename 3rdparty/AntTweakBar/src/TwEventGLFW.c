//  ---------------------------------------------------------------------------
//
//  @file       TwEventGLFW.c
//  @brief      Helper: 
//              translate and re-send mouse and keyboard events 
//              from GLFW event callbacks to AntTweakBar
//  
//  @author     Philippe Decaudin
//  @license    This file is part of the AntTweakBar library.
//              For conditions of distribution and use, see License.txt
//
//  ---------------------------------------------------------------------------

// #include <GL/glfw.h>
#include "GLFW/glfw3.h" // a subset of GLFW.h needed to compile TwEventGLFW.c
// note: AntTweakBar.dll does not need to link with GLFW, 
// it just needs some definitions for its helper functions.

#include <AntTweakBar.h>


void TwEventMouseButtonGLFW(GLFWwindow* window, int glfwButton, int glfwAction, int glfwMods)
{
    int handled = 0;
    TwMouseAction action = (glfwAction==GLFW_PRESS) ? TW_MOUSE_PRESSED : TW_MOUSE_RELEASED;

    if( glfwButton==GLFW_MOUSE_BUTTON_LEFT )
        handled = TwMouseButton(action, TW_MOUSE_LEFT);
    else if( glfwButton==GLFW_MOUSE_BUTTON_RIGHT )
        handled = TwMouseButton(action, TW_MOUSE_RIGHT);
    else if( glfwButton==GLFW_MOUSE_BUTTON_MIDDLE )
        handled = TwMouseButton(action, TW_MOUSE_MIDDLE);

    //return handled;
}


int g_KMod = 0;


void TwEventKeyGLFW(GLFWwindow* window, int glfwKey, int glfwScancode, int glfwAction, int glfwMods)
{
    int handled = 0;

    // Register of modifiers state
    if( glfwAction==GLFW_PRESS )
    {
        switch( glfwKey )
        {
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            g_KMod |= TW_KMOD_SHIFT;
            break;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
            g_KMod |= TW_KMOD_CTRL;
            break;
        case GLFW_KEY_LEFT_ALT:
        case GLFW_KEY_RIGHT_ALT:
            g_KMod |= TW_KMOD_ALT;
            break;
        }
    }
    else
    {
        switch( glfwKey )
        {
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            g_KMod &= ~TW_KMOD_SHIFT;
            break;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
            g_KMod &= ~TW_KMOD_CTRL;
            break;
        case GLFW_KEY_LEFT_ALT:
        case GLFW_KEY_RIGHT_ALT:
            g_KMod &= ~TW_KMOD_ALT;
            break;
        }
    }

    // Process key pressed
    if( glfwAction==GLFW_PRESS )
    {
        int mod = g_KMod;
        int testkp = ((mod&TW_KMOD_CTRL) || (mod&TW_KMOD_ALT)) ? 1 : 0;

        if( (mod&TW_KMOD_CTRL) && glfwKey>0 && glfwKey<GLFW_KEY_ESCAPE )   // CTRL cases
            handled = TwKeyPressed(glfwKey, mod);
        else if( glfwKey>=GLFW_KEY_ESCAPE )
        {
            int k = 0;

            if( glfwKey>=GLFW_KEY_F1 && glfwKey<=GLFW_KEY_F15 )
                k = TW_KEY_F1 + (glfwKey-GLFW_KEY_F1);
            else if( testkp && glfwKey>=GLFW_KEY_KP_0 && glfwKey<=GLFW_KEY_KP_9 )
                k = '0' + (glfwKey-GLFW_KEY_KP_0);
            else
            {
                switch( glfwKey )
                {
                case GLFW_KEY_ESCAPE:
                    k = TW_KEY_ESCAPE;
                    break;
                case GLFW_KEY_UP:
                    k = TW_KEY_UP;
                    break;
                case GLFW_KEY_DOWN:
                    k = TW_KEY_DOWN;
                    break;
                case GLFW_KEY_LEFT:
                    k = TW_KEY_LEFT;
                    break;
                case GLFW_KEY_RIGHT:
                    k = TW_KEY_RIGHT;
                    break;
                case GLFW_KEY_TAB:
                    k = TW_KEY_TAB;
                    break;
                case GLFW_KEY_ENTER:
                    k = TW_KEY_RETURN;
                    break;
                case GLFW_KEY_BACKSPACE:
                    k = TW_KEY_BACKSPACE;
                    break;
                case GLFW_KEY_INSERT:
                    k = TW_KEY_INSERT;
                    break;
                case GLFW_KEY_DELETE:
                    k = TW_KEY_DELETE;
                    break;
                case GLFW_KEY_PAGE_UP:
                    k = TW_KEY_PAGE_UP;
                    break;
                case GLFW_KEY_PAGE_DOWN:
                    k = TW_KEY_PAGE_DOWN;
                    break;
                case GLFW_KEY_HOME:
                    k = TW_KEY_HOME;
                    break;
                case GLFW_KEY_END:
                    k = TW_KEY_END;
                    break;
                case GLFW_KEY_KP_ENTER:
                    k = TW_KEY_RETURN;
                    break;
                case GLFW_KEY_KP_DIVIDE:
                    if( testkp )
                        k = '/';
                    break;
                case GLFW_KEY_KP_MULTIPLY:
                    if( testkp )
                        k = '*';
                    break;
                case GLFW_KEY_KP_SUBTRACT:
                    if( testkp )
                        k = '-';
                    break;
                case GLFW_KEY_KP_ADD:
                    if( testkp )
                        k = '+';
                    break;
                case GLFW_KEY_KP_DECIMAL:
                    if( testkp )
                        k = '.';
                    break;
                case GLFW_KEY_KP_EQUAL:
                    if( testkp )
                        k = '=';
                    break;
                }
            }

            if( k>0 )
                handled = TwKeyPressed(k, mod);
        }
    }

    //return handled;
}


void TwEventCharGLFW(GLFWwindow* window, unsigned int glfwCodepoint)
{
    if((glfwCodepoint & 0xff00)==0 )
        TwKeyPressed(glfwCodepoint, g_KMod);
}

void TwEvenCursorPosGLFW(GLFWwindow* window, double mouseX, double mouseY)
{
    TwMouseMotion((int)mouseX, (int)mouseY);
}

void TwEventScrollGLFW(GLFWwindow* window, double offsetX, double offsetY)
{
    TwMouseWheel((int)offsetY);
}
