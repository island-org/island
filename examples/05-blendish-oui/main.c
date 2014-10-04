#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

#include <blendish.h>
#include <oui.h>

////////////////////////////////////////////////////////////////////////////////

typedef enum
{
    // label
    ST_LABEL = 0,
    // button
    ST_BUTTON = 1,
    // radio button
    ST_RADIO = 2,
    // progress slider 
    ST_SLIDER = 3,
    // column
    ST_COLUMN = 4,
    // row
    ST_ROW = 5,
    // check button
    ST_CHECK = 6,
    // panel
    ST_PANEL = 7,
    // text
    ST_TEXT = 8,
} SubType;

typedef struct
{
    int subtype;
} UIData;

typedef struct
{
    UIData head;
    int iconid;
    const char *label;
} UIButtonData;

typedef struct
{
    UIData head;
    const char *label;
    int *option;
} UICheckData;

typedef struct
{
    UIData head;
    int iconid;
    const char *label;
    int *value;
} UIRadioData;

typedef struct
{
    UIData head;
    const char *label;
    float *progress;
} UISliderData;

typedef struct {
    UIData head;
    char *text;
    int maxsize;
} UITextData;

// calculate which corners are sharp for an item, depending on whether
// the container the item is in has negative spacing, and the item
// is first or last element in a sequence of 2 or more elements.
int cornerFlags(int item) 
{
    int parent = uiParent(item);
    int numkids = uiGetChildCount(parent);
    const UIData *head;

    if (numkids < 2) return BND_CORNER_NONE;
    head = (const UIData *)uiGetData(parent);
    if (head) 
    {
        int numid = uiGetChildId(item);
        switch(head->subtype) 
        {
        case ST_COLUMN:
            {
                if (!numid) return BND_CORNER_DOWN;
                else if (numid == numkids-1) return BND_CORNER_TOP;
                else return BND_CORNER_ALL;

            } break;
        case ST_ROW: 
            {
                if (!numid) return BND_CORNER_RIGHT;
                else if (numid == numkids-1) return BND_CORNER_LEFT;
                else return BND_CORNER_ALL;

            } break;
        default: break;
        }
    }
    return BND_CORNER_NONE;
}

void testrect(NVGcontext *vg, UIrect rect) 
{
#if 0
    nvgBeginPath(vg);
    nvgRect(vg,rect.x+0.5,rect.y+0.5,rect.w-1,rect.h-1);
    nvgStrokeColor(vg,nvgRGBf(1,0,0));
    nvgStrokeWidth(vg,1);
    nvgStroke(vg);
#endif
}

void drawUI(NVGcontext *vg, int item, int x, int y)
{
    const UIData *head = (const UIData *)uiGetData(item);
    UIrect rect = uiGetRect(item);
    rect.x += x;
    rect.y += y; 
    if (uiGetState(item) == UI_FROZEN) 
    {
        nvgGlobalAlpha(vg, 0.5);
    }
    if (head)
    {
        switch (head->subtype) 
        {
        case ST_PANEL: 
            {
                bndBevel(vg,rect.x,rect.y,rect.w,rect.h);
            } break;
        case ST_LABEL: 
            {
                const UIButtonData *data = (UIButtonData*)head;
                bndLabel(vg,rect.x,rect.y,rect.w,rect.h,
                    data->iconid,data->label);
            } break;
        case ST_BUTTON: 
            {
                const UIButtonData *data = (UIButtonData*)head;
                bndToolButton(vg,rect.x,rect.y,rect.w,rect.h,
                    cornerFlags(item),(BNDwidgetState)uiGetState(item),
                    data->iconid,data->label);                            
            } break;
        case ST_CHECK: 
            {
                const UICheckData *data = (UICheckData*)head;
                BNDwidgetState state = (BNDwidgetState)uiGetState(item);
                if (*data->option)
                    state = BND_ACTIVE;
                bndOptionButton(vg,rect.x,rect.y,rect.w,rect.h, state,
                    data->label);
            } break;
        case ST_RADIO:
            {
                const UIRadioData *data = (UIRadioData*)head;
                BNDwidgetState state = (BNDwidgetState)uiGetState(item);
                if (*data->value == uiGetChildId(item))
                    state = BND_ACTIVE;
                bndRadioButton(vg,rect.x,rect.y,rect.w,rect.h,
                    cornerFlags(item),state,
                    data->iconid,data->label);
            } break;
        case ST_SLIDER:
            {
                const UISliderData *data = (UISliderData*)head;
                BNDwidgetState state = (BNDwidgetState)uiGetState(item);
                static char value[32];
                sprintf(value,"%.0f%%", (*data->progress)*100.0f);
                bndSlider(vg,rect.x,rect.y,rect.w,rect.h,
                    cornerFlags(item),state,
                    *data->progress,data->label,value);
            } break;
        case ST_TEXT: {
            const UITextData *data = (UITextData*)head;
            BNDwidgetState state = (BNDwidgetState)uiGetState(item);
            int idx = strlen(data->text);
            bndTextField(vg,rect.x,rect.y,rect.w,rect.h,
                cornerFlags(item),state, -1, data->text, idx, idx);
                      } break;

        default: 
            {
                testrect(vg,rect);
            } break;
        }
    } 
    else
    {
        testrect(vg,rect);
    }

    {
        // Recursively parse all the children
        int kid = uiFirstChild(item);
        while (kid > 0)
        {
            drawUI(vg, kid, rect.x, rect.y);
            kid = uiNextSibling(kid);
        }
    }

    if (uiGetState(item) == UI_FROZEN)
    {
        nvgGlobalAlpha(vg, 1.0);
    }
}

int label(int parent, int iconid, const char *label) 
{    
    int item = uiItem();
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    {
        UIButtonData *data = (UIButtonData *)uiAllocData(item, sizeof(UIButtonData));
        data->head.subtype = ST_LABEL;
        data->iconid = iconid;
        data->label = label;
    }
    uiAppend(parent, item);
    return item;
}

void demohandler(int item, UIevent event) 
{
    const UIButtonData *data = (const UIButtonData *)uiGetData(item);
    printf("clicked: %lld %s\n", uiGetHandle(item), data->label);
}

int button(int parent, UIhandle handle, int iconid, const char *label, UIhandler handler) 
{
    // create new ui item
    int item = uiItem(); 
    // set persistent handle for item that is used
    // to track activity over time
    uiSetHandle(item, handle);
    // set size of wiget; horizontal size is dynamic, vertical is fixed
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    // attach event handler e.g. demohandler above
    uiSetHandler(item, handler, UI_BUTTON0_HOT_UP);
    {
        // store some custom data with the button that we use for styling
        UIButtonData *data = (UIButtonData *)uiAllocData(item, sizeof(UIButtonData));
        data->head.subtype = ST_BUTTON;
        data->iconid = iconid;
        data->label = label;
    }
    uiAppend(parent, item);
    return item;
}

void checkhandler(int item, UIevent event)
{
    const UICheckData *data = (const UICheckData *)uiGetData(item);
    *data->option = !(*data->option);
}

int check(int parent, UIhandle handle, const char *label, int *option)
{
    // create new ui item
    int item = uiItem(); 
    // set persistent handle for item that is used
    // to track activity over time
    uiSetHandle(item, handle);
    // set size of wiget; horizontal size is dynamic, vertical is fixed
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    // attach event handler e.g. demohandler above
    uiSetHandler(item, checkhandler, UI_BUTTON0_DOWN);
    {
        // store some custom data with the button that we use for styling
        UICheckData *data = (UICheckData *)uiAllocData(item, sizeof(UICheckData));
        data->head.subtype = ST_CHECK;
        data->label = label;
        data->option = option;
    }
    uiAppend(parent, item);
    return item;
}

// simple logic for a slider

// starting offset of the currently active slider
static float sliderstart = 0.0;

// event handler for slider (same handler for all sliders)
void sliderhandler(int item, UIevent event) 
{
    // retrieve the custom data we saved with the slider
    UISliderData *data = (UISliderData *)uiGetData(item);
    switch(event) {
        case UI_BUTTON0_DOWN: 
            {
                // button was pressed for the first time; capture initial
                // slider value.
                sliderstart = *data->progress;                              
            } break;
        case UI_BUTTON0_CAPTURE: 
            {
                // called for every frame that the button is pressed.
                // get the delta between the click point and the current
                // mouse position
                UIvec2 pos = uiGetCursorStartDelta();
                // get the items layouted rectangle
                UIrect rc = uiGetRect(item);
                // calculate our new offset and clamp
                float value = sliderstart + ((float)pos.x / (float)rc.w);
                value = (value<0)?0:(value>1)?1:value;
                // assign the new value
                *data->progress = value;                                 
            } break;
        default: break;
    }
}

int slider(int parent, UIhandle handle, const char *label, float *progress)
{
    // create new ui item
    int item = uiItem();
    // set persistent handle for item that is used
    // to track activity over time
    uiSetHandle(item, handle);
    // set size of wiget; horizontal size is dynamic, vertical is fixed
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    // attach our slider event handler and capture two classes of events
    uiSetHandler(item, sliderhandler, UI_BUTTON0_DOWN | UI_BUTTON0_CAPTURE);
    {
        // store some custom data with the button that we use for styling
        // and logic, e.g. the pointer to the data we want to alter.
        UISliderData *data = (UISliderData *)uiAllocData(item, sizeof(UISliderData));
        data->head.subtype = ST_SLIDER;
        data->label = label;
        data->progress = progress;
    }
    uiAppend(parent, item);
    return item;
}

void textboxhandler(int item, UIevent event) {
    UITextData *data = (UITextData *)uiGetData(item);
    switch(event) {
        default: break;
        case UI_BUTTON0_DOWN: {
            uiFocus(item);
                              } break;
        case UI_KEY_DOWN: {
            unsigned int key = uiGetKey();
            switch(key) {
        default: break;
        case GLFW_KEY_BACKSPACE: {
            int size = strlen(data->text);
            if (!size) return;
            data->text[size-1] = 0;
                                 } break;
        case GLFW_KEY_ENTER: {
            uiFocus(-1);
                             } break;
            }
                          } break;
        case UI_CHAR: {
            unsigned int key = uiGetKey();
            int size;
            if ((key > 255)||(key < 32)) return;
            size = strlen(data->text);
            if (size >= (data->maxsize-1)) return;
            data->text[size] = (char)key;
                      } break;
    }
}

int textbox(int parent, UIhandle handle, char *text, int maxsize) {
    int item = uiItem();
    UITextData *data;
    uiSetHandle(item, handle);
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    uiSetHandler(item, textboxhandler, 
        UI_BUTTON0_DOWN | UI_KEY_DOWN | UI_CHAR);
    // store some custom data with the button that we use for styling
    // and logic, e.g. the pointer to the data we want to alter.
    data = (UITextData *)uiAllocData(item, sizeof(UITextData));
    data->head.subtype = ST_TEXT;
    data->text = text;
    data->maxsize = maxsize;
    uiAppend(parent, item);
    return item;
}

// simple logic for a radio button
void radiohandler(int item, UIevent event)
{
    UIRadioData *data = (UIRadioData *)uiGetData(item);
    *data->value = uiGetChildId(item);
}

int radio(int parent, UIhandle handle, int iconid, const char *label, int *value)
{
    int item = uiItem();
    uiSetHandle(item, handle);
    uiSetSize(item, label?0:BND_TOOL_WIDTH, BND_WIDGET_HEIGHT);
    {
        UIRadioData *data = (UIRadioData *)uiAllocData(item, sizeof(UIRadioData));
        data->head.subtype = ST_RADIO;
        data->iconid = iconid;
        data->label = label;
        data->value = value;
    }
    uiSetHandler(item, radiohandler, UI_BUTTON0_DOWN);
    uiAppend(parent, item);
    return item;
}

void columnhandler(int parent, UIevent event)
{
    int item = uiLastChild(parent);
    int last = uiPrevSibling(item);
    // mark the new item as positioned under the previous item
    uiSetRelToTop(item, last);
    // fill parent horizontally, anchor to previous item vertically
    uiSetLayout(item, UI_HFILL|UI_TOP);
    // if not the first item, add a margin of 1
    uiSetMargins(item, 0, (last < 0)?0:1, 0, 0);
}

int column(int parent)
{
    int item = uiItem();
    uiSetHandler(item, columnhandler, UI_APPEND);
    uiAppend(parent, item);
    return item;
}

void vgrouphandler(int parent, UIevent event)
{
    int item = uiLastChild(parent);
    int last = uiPrevSibling(item);
    // mark the new item as positioned under the previous item
    uiSetRelToTop(item, last);
    // fill parent horizontally, anchor to previous item vertically
    uiSetLayout(item, UI_HFILL|UI_TOP);
    // if not the first item, add a margin
    uiSetMargins(item, 0, (last < 0)?0:-2, 0, 0);
}

int vgroup(int parent) {
    int item = uiItem();
    UIData *data = (UIData *)uiAllocData(item, sizeof(UIData));
    data->subtype = ST_COLUMN;
    uiSetHandler(item, vgrouphandler, UI_APPEND);
    uiAppend(parent, item);
    return item;
}

void hgrouphandler(int parent, UIevent event)
{
    int item = uiLastChild(parent);
    int last = uiPrevSibling(item);
    uiSetRelToLeft(item, last);
    if (last > 0)
        uiSetRelToRight(last, item);
    uiSetLayout(item, UI_LEFT|UI_RIGHT);
    uiSetMargins(item, (last < 0)?0:-1, 0, 0, 0);
}

int hgroup(int parent)
{
    int item = uiItem();
    UIData *data = (UIData *)uiAllocData(item, sizeof(UIData));
    data->subtype = ST_ROW;
    uiSetHandler(item, hgrouphandler, UI_APPEND);
    uiAppend(parent, item);
    return item;
}

void rowhandler(int parent, UIevent event)
{
    int item = uiLastChild(parent);
    int last = uiPrevSibling(item);
    uiSetRelToLeft(item, last);
    if (last > 0)
        uiSetRelToRight(last, item);
    uiSetLayout(item, UI_LEFT|UI_RIGHT);
    uiSetMargins(item, (last < 0)?0:8, 0, 0, 0);
}

int row(int parent)
{
    int item = uiItem();
    uiSetHandler(item, rowhandler, UI_APPEND);
    uiAppend(parent, item);
    return item;
}

static int enum1 = 0;
static float progress1 = 0.25f;
static float progress2 = 0.75f;
static int option1 = 1;
static int option2 = 0;
static int option3 = 0;
static char textbuffer[32] = "click and edit";

void createUI(NVGcontext *vg, float w, float h)
{
    int col;

    uiClear();

    {
        int root = uiItem(); 
        // position root element
        uiSetLayout(root,UI_LEFT|UI_TOP);
        uiSetMargins(root,50,50,0,0);
        uiSetSize(root,250,400);
    }

    col = column(0);
    uiSetLayout(col, UI_TOP|UI_HFILL);

    button(col, __LINE__, BND_ICONID(6,3), "Item 1", demohandler);
    button(col, __LINE__, BND_ICONID(6,3), "Item 2", demohandler);

    {
        int h = hgroup(col);
        radio(h, __LINE__, BND_ICONID(6,3), "Item 3.0", &enum1);
        radio(h, __LINE__, BND_ICONID(0,10), NULL, &enum1);
        radio(h, __LINE__, BND_ICONID(1,10), NULL, &enum1);
        radio(h, __LINE__, BND_ICONID(6,3), "Item 3.3", &enum1);
    }

    {
        int colr;
        int rows = row(col);
        int coll = vgroup(rows);
        label(coll, -1, "Items 4.0:");
        coll = vgroup(coll);
        button(coll, __LINE__, BND_ICONID(6,3), "Item 4.0.0", demohandler);
        button(coll, __LINE__, BND_ICONID(6,3), "Item 4.0.1", demohandler);
        colr = vgroup(rows);
        uiSetFrozen(colr, option1);
        label(colr, -1, "Items 4.1:");
        colr = vgroup(colr);
        slider(colr, __LINE__, "Item 4.1.0", &progress1);
        slider(colr,__LINE__, "Item 4.1.1", &progress2);
    }

    button(col, __LINE__, BND_ICONID(6,3), "Item 5", NULL);

    check(col, __LINE__, "Frozen", &option1);
    check(col, __LINE__, "Item 7", &option2);
    check(col, __LINE__, "Item 8", &option3);

    textbox(col, (UIhandle)textbuffer, textbuffer, 32);

    uiLayout();
}

UIcontext *uictx;

static void charevent(GLFWwindow *window, unsigned int value)
{
    NVG_NOTUSED(window);
    uiSetChar(value);
}

static void keyevent(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    NVG_NOTUSED(scancode);
    NVG_NOTUSED(mods);
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    uiSetKey(key, mods, action);
}

void setup()
{
    size(640, 320);
    glfwSetKeyCallback(window, keyevent);
    glfwSetCharCallback(window, charevent);

    uictx = uiCreateContext();
    uiMakeCurrent(uictx);

    bndSetFont(nvgCreateFont(vg, "system", "../3rdparty/blendish/DejaVuSans.ttf"));
    bndSetIconImage(nvgCreateImage(vg, "../3rdparty/blendish/blender_icons16.png", NVG_IMAGE_GENERATE_MIPMAPS));
}

void draw()
{
    unsigned char mbut = 0;
    int scrollarea1 = 0;
    int i;

    background(gray(122));

    uiSetCursor((int)mouseX,(int)mouseY);
    for (i=0; i<3; i++) uiSetButton(i, 0);
    if (mousePressed) uiSetButton(mouseButton, 1);
    createUI(vg, width, height);
    uiProcess();
    drawUI(vg, 0, 0, 0);
    {
        int x = mouseX - width/2;
        int y = mouseY - height/2;
        if (abs(x) > width/4) {
            bndJoinAreaOverlay(vg, 0, 0, width, height, 0, (x > 0));
        } else if (abs(y) > height/4) {
            bndJoinAreaOverlay(vg, 0, 0, width, height, 1, (y > 0));
        }
    }
}

void shutdown()
{
    uiDestroyContext(uictx);
}
