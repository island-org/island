//
// based on NanoVG's example code by Mikko Mononen

#include <stdio.h>
#ifdef NANOVG_GLEW
#	include <GL/glew.h>
#endif
#ifdef __APPLE__
#	define GLFW_INCLUDE_GLCOREARB
#endif
#include <GLFW/glfw3.h>
#include "nanovg.h"
#define NANOVG_GL3_IMPLEMENTATION
#include "nanovg_gl.h"

#define BLENDISH_IMPLEMENTATION
#include "blendish.h"

#define OUI_IMPLEMENTATION
#include "oui.h"

////////////////////////////////////////////////////////////////////////////////

typedef enum {
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

typedef struct {
    int subtype;
} UIData;

typedef struct {
    UIData head;
    int iconid;
    const char *label;
} UIButtonData;

typedef struct {
    UIData head;
    const char *label;
    int *option;
} UICheckData;

typedef struct {
    UIData head;
    int iconid;
    const char *label;
    int *value;
} UIRadioData;

typedef struct {
    UIData head;
    const char *label;
    float *progress;
} UISliderData;

typedef struct {
    UIData head;
    char *text;
    int maxsize;
} UITextData;

////////////////////////////////////////////////////////////////////////////////

void init(NVGcontext *vg) {
    bndSetFont(nvgCreateFont(vg, "system", "../DejaVuSans.ttf"));
    bndSetIconImage(nvgCreateImage(vg, "../blender_icons16.png"));
}

// calculate which corners are sharp for an item, depending on whether
// the container the item is in has negative spacing, and the item
// is first or last element in a sequence of 2 or more elements.
int cornerFlags(int item) {
    int parent = uiParent(item);
    int numkids = uiGetChildCount(parent);
    if (numkids < 2) return BND_CORNER_NONE;
    const UIData *head = (const UIData *)uiGetData(parent);
    if (head) {
        int numid = uiGetChildId(item);
        switch(head->subtype) {
        case ST_COLUMN: {
                if (!numid) return BND_CORNER_DOWN;
                else if (numid == numkids-1) return BND_CORNER_TOP;
                else return BND_CORNER_ALL;
        } break;
        case ST_ROW: {
                if (!numid) return BND_CORNER_RIGHT;
                else if (numid == numkids-1) return BND_CORNER_LEFT;
                else return BND_CORNER_ALL;
        } break;
        default: break;
        }
    }
    return BND_CORNER_NONE;
}

void testrect(NVGcontext *vg, UIrect rect) {
#if 0    
    nvgBeginPath(vg);
    nvgRect(vg,rect.x+0.5,rect.y+0.5,rect.w-1,rect.h-1);
    nvgStrokeColor(vg,nvgRGBf(1,0,0));
    nvgStrokeWidth(vg,1);
    nvgStroke(vg);    
#endif    
}

void drawUI(NVGcontext *vg, int item, int x, int y) {
    const UIData *head = (const UIData *)uiGetData(item);
    UIrect rect = uiGetRect(item);
    rect.x += x;
    rect.y += y; 
    if (uiGetState(item) == UI_FROZEN) {
        nvgGlobalAlpha(vg, BND_DISABLED_ALPHA);
    }
    if (head) {
        switch(head->subtype) {
            default: {
                testrect(vg,rect);
            } break;
            case ST_PANEL: {
                bndBevel(vg,rect.x,rect.y,rect.w,rect.h);
            } break;
            case ST_LABEL: {
                assert(head);
                const UIButtonData *data = (UIButtonData*)head;
                bndLabel(vg,rect.x,rect.y,rect.w,rect.h,
                    data->iconid,data->label);
            } break;
            case ST_BUTTON: {
                const UIButtonData *data = (UIButtonData*)head;
                bndToolButton(vg,rect.x,rect.y,rect.w,rect.h,
                    cornerFlags(item),(BNDwidgetState)uiGetState(item),
                    data->iconid,data->label);
            } break;
            case ST_CHECK: {
                const UICheckData *data = (UICheckData*)head;
                BNDwidgetState state = (BNDwidgetState)uiGetState(item);
                if (*data->option)
                    state = BND_ACTIVE;
                bndOptionButton(vg,rect.x,rect.y,rect.w,rect.h, state,
                    data->label);
            } break;
            case ST_RADIO:{
                const UIRadioData *data = (UIRadioData*)head;
                BNDwidgetState state = (BNDwidgetState)uiGetState(item);
                if (*data->value == uiGetChildId(item))
                    state = BND_ACTIVE;
                bndRadioButton(vg,rect.x,rect.y,rect.w,rect.h,
                    cornerFlags(item),state,
                    data->iconid,data->label);
            } break;
            case ST_SLIDER:{
                const UISliderData *data = (UISliderData*)head;
                BNDwidgetState state = (BNDwidgetState)uiGetState(item);
                static char value[32];
                sprintf(value,"%.0f%%",(*data->progress)*100.0f);
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
        }
    } else {
        testrect(vg,rect);
    }
    
    int kid = uiFirstChild(item);
    while (kid > 0) {
        drawUI(vg, kid, rect.x, rect.y);
        kid = uiNextSibling(kid);
    }
    if (uiGetState(item) == UI_FROZEN) {
        nvgGlobalAlpha(vg, 1.0);
    }
}

int label(int parent, int iconid, const char *label) {    
    int item = uiItem();
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    UIButtonData *data = (UIButtonData *)uiAllocData(item, sizeof(UIButtonData));
    data->head.subtype = ST_LABEL;
    data->iconid = iconid;
    data->label = label;
    uiAppend(parent, item);
    return item;
}

void demohandler(int item, UIevent event) {
    const UIButtonData *data = (const UIButtonData *)uiGetData(item);
    printf("clicked: %lld %s\n", uiGetHandle(item), data->label);
}

int button(int parent, UIhandle handle, int iconid, const char *label, 
    UIhandler handler) {
    // create new ui item
    int item = uiItem(); 
    // set persistent handle for item that is used
    // to track activity over time
    uiSetHandle(item, handle);
    // set size of wiget; horizontal size is dynamic, vertical is fixed
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    // attach event handler e.g. demohandler above
    uiSetHandler(item, handler, UI_BUTTON0_HOT_UP);
    // store some custom data with the button that we use for styling
    UIButtonData *data = (UIButtonData *)uiAllocData(item, sizeof(UIButtonData));
    data->head.subtype = ST_BUTTON;
    data->iconid = iconid;
    data->label = label;
    uiAppend(parent, item);
    return item;
}

void checkhandler(int item, UIevent event) {
    const UICheckData *data = (const UICheckData *)uiGetData(item);
    *data->option = !(*data->option);
}

int check(int parent, UIhandle handle, const char *label, int *option) {
    // create new ui item
    int item = uiItem(); 
    // set persistent handle for item that is used
    // to track activity over time
    uiSetHandle(item, handle);
    // set size of wiget; horizontal size is dynamic, vertical is fixed
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    // attach event handler e.g. demohandler above
    uiSetHandler(item, checkhandler, UI_BUTTON0_DOWN);
    // store some custom data with the button that we use for styling
    UICheckData *data = (UICheckData *)uiAllocData(item, sizeof(UICheckData));
    data->head.subtype = ST_CHECK;
    data->label = label;
    data->option = option;
    uiAppend(parent, item);
    return item;
}

// simple logic for a slider

// starting offset of the currently active slider
static float sliderstart = 0.0;

// event handler for slider (same handler for all sliders)
void sliderhandler(int item, UIevent event) {
    // retrieve the custom data we saved with the slider
    UISliderData *data = (UISliderData *)uiGetData(item);
    switch(event) {
        default: break;
        case UI_BUTTON0_DOWN: {
            // button was pressed for the first time; capture initial
            // slider value.
            sliderstart = *data->progress;
        } break;
        case UI_BUTTON0_CAPTURE: {
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
    }
}

int slider(int parent, UIhandle handle, const char *label, float *progress) {
    // create new ui item
    int item = uiItem();
    // set persistent handle for item that is used
    // to track activity over time
    uiSetHandle(item, handle);
    // set size of wiget; horizontal size is dynamic, vertical is fixed
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    // attach our slider event handler and capture two classes of events
    uiSetHandler(item, sliderhandler, 
        UI_BUTTON0_DOWN | UI_BUTTON0_CAPTURE);
    // store some custom data with the button that we use for styling
    // and logic, e.g. the pointer to the data we want to alter.
    UISliderData *data = (UISliderData *)uiAllocData(item, sizeof(UISliderData));
    data->head.subtype = ST_SLIDER;
    data->label = label;
    data->progress = progress;
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
            if ((key > 255)||(key < 32)) return;
            int size = strlen(data->text);
            if (size >= (data->maxsize-1)) return;
            data->text[size] = (char)key;
        } break;
    }
}

int textbox(int parent, UIhandle handle, char *text, int maxsize) {
    int item = uiItem();
    uiSetHandle(item, handle);
    uiSetSize(item, 0, BND_WIDGET_HEIGHT);
    uiSetHandler(item, textboxhandler, 
        UI_BUTTON0_DOWN | UI_KEY_DOWN | UI_CHAR);
    // store some custom data with the button that we use for styling
    // and logic, e.g. the pointer to the data we want to alter.
    UITextData *data = (UITextData *)uiAllocData(item, sizeof(UITextData));
    data->head.subtype = ST_TEXT;
    data->text = text;
    data->maxsize = maxsize;
    uiAppend(parent, item);
    return item;
}

// simple logic for a radio button
void radiohandler(int item, UIevent event) {
    UIRadioData *data = (UIRadioData *)uiGetData(item);
    *data->value = uiGetChildId(item);
}

int radio(int parent, UIhandle handle, int iconid, const char *label, int *value) {
    int item = uiItem();
    uiSetHandle(item, handle);
    uiSetSize(item, label?0:BND_TOOL_WIDTH, BND_WIDGET_HEIGHT);
    UIRadioData *data = (UIRadioData *)uiAllocData(item, sizeof(UIRadioData));
    data->head.subtype = ST_RADIO;
    data->iconid = iconid;
    data->label = label;
    data->value = value;
    uiSetHandler(item, radiohandler, UI_BUTTON0_DOWN);
    uiAppend(parent, item);
    return item;
}

void columnhandler(int parent, UIevent event) {
    int item = uiLastChild(parent);
    int last = uiPrevSibling(item);
    // mark the new item as positioned under the previous item
    uiSetRelToTop(item, last);
    // fill parent horizontally, anchor to previous item vertically
    uiSetLayout(item, UI_HFILL|UI_TOP);
    // if not the first item, add a margin of 1
    uiSetMargins(item, 0, (last < 0)?0:1, 0, 0);
}

int panel() {
    int item = uiItem();
    UIData *data = (UIData *)uiAllocData(item, sizeof(UIData));
    data->subtype = ST_PANEL;
    return item;
}

int column(int parent) {
    int item = uiItem();
    uiSetHandler(item, columnhandler, UI_APPEND);
    uiAppend(parent, item);
    return item;
}

void vgrouphandler(int parent, UIevent event) {
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

void hgrouphandler(int parent, UIevent event) {
    int item = uiLastChild(parent);
    int last = uiPrevSibling(item);
    uiSetRelToLeft(item, last);
    if (last > 0)
        uiSetRelToRight(last, item);
    uiSetLayout(item, UI_LEFT|UI_RIGHT);
    uiSetMargins(item, (last < 0)?0:-1, 0, 0, 0);
}

int hgroup(int parent) {
    int item = uiItem();
    UIData *data = (UIData *)uiAllocData(item, sizeof(UIData));
    data->subtype = ST_ROW;
    uiSetHandler(item, hgrouphandler, UI_APPEND);
    uiAppend(parent, item);
    return item;
}

void rowhandler(int parent, UIevent event) {
    int item = uiLastChild(parent);
    int last = uiPrevSibling(item);
    uiSetRelToLeft(item, last);
    if (last > 0)
        uiSetRelToRight(last, item);
    uiSetLayout(item, UI_LEFT|UI_RIGHT);
    uiSetMargins(item, (last < 0)?0:8, 0, 0, 0);
}

int row(int parent) {
    int item = uiItem();
    uiSetHandler(item, rowhandler, UI_APPEND);
    uiAppend(parent, item);
    return item;
}

void draw_noodles(NVGcontext *vg, int x, int y) {
    int w = 200;
    int s = 70;

    bndNodeBackground(vg, x+w, y-50, 100, 200, BND_DEFAULT, BND_ICONID(6,3),
        "Default", nvgRGBf(0.392f,0.392f,0.392f));
    bndNodeBackground(vg, x+w+120, y-50, 100, 200, BND_HOVER, BND_ICONID(6,3),
        "Hover", nvgRGBf(0.392f,0.392f,0.392f));
    bndNodeBackground(vg, x+w+240, y-50, 100, 200, BND_ACTIVE, BND_ICONID(6,3),
        "Active", nvgRGBf(0.392f,0.392f,0.392f));
    
    for (int i = 0; i < 9; ++i) {
        int a = i%3;
        int b = i/3;
        bndNodeWire(vg, x, y+s*a, x+w, y+s*b, (BNDwidgetState)a, (BNDwidgetState)b);
    }
    
    bndNodePort(vg, x, y, BND_DEFAULT, nvgRGBf(0.5f, 0.5f, 0.5f));
    bndNodePort(vg, x+w, y, BND_DEFAULT, nvgRGBf(0.5f, 0.5f, 0.5f));
    bndNodePort(vg, x, y+s, BND_HOVER, nvgRGBf(0.5f, 0.5f, 0.5f));
    bndNodePort(vg, x+w, y+s, BND_HOVER, nvgRGBf(0.5f, 0.5f, 0.5f));
    bndNodePort(vg, x, y+2*s, BND_ACTIVE, nvgRGBf(0.5f, 0.5f, 0.5f));
    bndNodePort(vg, x+w, y+2*s, BND_ACTIVE, nvgRGBf(0.5f, 0.5f, 0.5f));
}

void draw(NVGcontext *vg, float w, float h) {
    bndBackground(vg, 0, 0, w, h);
    
    bndSplitterWidgets(vg, 0, 0, w, h);
    
    int x = 10;
    int y = 10;
    
    bndToolButton(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_DEFAULT,
        BND_ICONID(6,3),"Default");
    y += 25;
    bndToolButton(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_HOVER,
        BND_ICONID(6,3),"Hovered");
    y += 25;   
    bndToolButton(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_ACTIVE,
        BND_ICONID(6,3),"Active");
    
    y += 40;
    bndRadioButton(vg,x,y,80,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_DEFAULT,
        -1,"Default");
    y += 25;
    bndRadioButton(vg,x,y,80,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_HOVER,
        -1,"Hovered");
    y += 25;
    bndRadioButton(vg,x,y,80,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_ACTIVE,
        -1,"Active");

    y += 25;
    bndLabel(vg,x,y,120,BND_WIDGET_HEIGHT,-1,"Label:");
    y += BND_WIDGET_HEIGHT;
    bndChoiceButton(vg,x,y,80,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_DEFAULT,
        -1, "Default");
    y += 25;
    bndChoiceButton(vg,x,y,80,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_HOVER,
        -1, "Hovered");
    y += 25;
    bndChoiceButton(vg,x,y,80,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_ACTIVE,
        -1, "Active");
        
    y += 25;
    int ry = y;
    int rx = x;
    
    y = 10;
    x += 130;
    bndOptionButton(vg,x,y,120,BND_WIDGET_HEIGHT,BND_DEFAULT,"Default");
    y += 25;
    bndOptionButton(vg,x,y,120,BND_WIDGET_HEIGHT,BND_HOVER,"Hovered");
    y += 25;
    bndOptionButton(vg,x,y,120,BND_WIDGET_HEIGHT,BND_ACTIVE,"Active");

    y += 40;
    bndNumberField(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_DOWN,BND_DEFAULT,
        "Top","100");
    y += BND_WIDGET_HEIGHT-2;
    bndNumberField(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_ALL,BND_DEFAULT,
        "Center","100");
    y += BND_WIDGET_HEIGHT-2;
    bndNumberField(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_TOP,BND_DEFAULT,
        "Bottom","100");

    int mx = x-30;
    int my = y-12;
    int mw = 120;
    bndMenuBackground(vg,mx,my,mw,120,BND_CORNER_TOP);
    bndMenuLabel(vg,mx,my,mw,BND_WIDGET_HEIGHT,-1,"Menu Title");
    my += BND_WIDGET_HEIGHT-2;
    bndMenuItem(vg,mx,my,mw,BND_WIDGET_HEIGHT,BND_DEFAULT,
        BND_ICONID(17,3),"Default");
    my += BND_WIDGET_HEIGHT-2;
    bndMenuItem(vg,mx,my,mw,BND_WIDGET_HEIGHT,BND_HOVER,
        BND_ICONID(18,3),"Hovered");
    my += BND_WIDGET_HEIGHT-2;
    bndMenuItem(vg,mx,my,mw,BND_WIDGET_HEIGHT,BND_ACTIVE,
        BND_ICONID(19,3),"Active");
    
    y = 10;
    x += 130;
    int ox = x;
    bndNumberField(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_DEFAULT,
        "Default","100");
    y += 25;
    bndNumberField(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_HOVER,
        "Hovered","100");
    y += 25;
    bndNumberField(vg,x,y,120,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_ACTIVE,
        "Active","100");
    
    y += 40;
    bndRadioButton(vg,x,y,60,BND_WIDGET_HEIGHT,BND_CORNER_RIGHT,BND_DEFAULT,
        -1,"One");
    x += 60-1;
    bndRadioButton(vg,x,y,60,BND_WIDGET_HEIGHT,BND_CORNER_ALL,BND_DEFAULT,
        -1,"Two");
    x += 60-1;
    bndRadioButton(vg,x,y,60,BND_WIDGET_HEIGHT,BND_CORNER_ALL,BND_DEFAULT,
        -1,"Three");
    x += 60-1;
    bndRadioButton(vg,x,y,60,BND_WIDGET_HEIGHT,BND_CORNER_LEFT,BND_ACTIVE,
        -1,"Butts");
    
    x = ox;
    y += 40;
    float progress_value = fmodf(glfwGetTime()/10.0,1.0);
    char progress_label[32];
    sprintf(progress_label, "%d%%", int(progress_value*100+0.5f));
    bndSlider(vg,x,y,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_DEFAULT,
        progress_value,"Default",progress_label);
    y += 25;
    bndSlider(vg,x,y,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_HOVER,
        progress_value,"Hovered",progress_label);
    y += 25;
    bndSlider(vg,x,y,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_ACTIVE,
        progress_value,"Active",progress_label);

    int rw = x+240-rx;
    float s_offset = sinf(glfwGetTime()/2.0)*0.5+0.5;
    float s_size = cosf(glfwGetTime()/3.11)*0.5+0.5;
    
    bndScrollBar(vg,rx,ry,rw,BND_SCROLLBAR_HEIGHT,BND_DEFAULT,s_offset,s_size);
    ry += 20;
    bndScrollBar(vg,rx,ry,rw,BND_SCROLLBAR_HEIGHT,BND_HOVER,s_offset,s_size);
    ry += 20;
    bndScrollBar(vg,rx,ry,rw,BND_SCROLLBAR_HEIGHT,BND_ACTIVE,s_offset,s_size);
    
    const char edit_text[] = "The quick brown fox";
    int textlen = strlen(edit_text)+1;
    int t = int(glfwGetTime()*2);
    int idx1 = (t/textlen)%textlen;
    int idx2 = idx1 + (t%(textlen-idx1));
    
    ry += 25;
    bndTextField(vg,rx,ry,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_DEFAULT,
        -1, edit_text, idx1, idx2);
    ry += 25;
    bndTextField(vg,rx,ry,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_HOVER,
        -1, edit_text, idx1, idx2);
    ry += 25;
    bndTextField(vg,rx,ry,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_ACTIVE,
        -1, edit_text, idx1, idx2);
        
    draw_noodles(vg, 20, ry+50);
    
    rx += rw + 20;
    ry = 10;
    bndScrollBar(vg,rx,ry,BND_SCROLLBAR_WIDTH,240,BND_DEFAULT,s_offset,s_size);
    rx += 20;
    bndScrollBar(vg,rx,ry,BND_SCROLLBAR_WIDTH,240,BND_HOVER,s_offset,s_size);
    rx += 20;
    bndScrollBar(vg,rx,ry,BND_SCROLLBAR_WIDTH,240,BND_ACTIVE,s_offset,s_size);
    
    x = ox;
    y += 40;
    bndToolButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_RIGHT,
        BND_DEFAULT,BND_ICONID(0,10),NULL);
    x += BND_TOOL_WIDTH-1;
    bndToolButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_ALL,
        BND_DEFAULT,BND_ICONID(1,10),NULL);
    x += BND_TOOL_WIDTH-1;
    bndToolButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_ALL,
        BND_DEFAULT,BND_ICONID(2,10),NULL);
    x += BND_TOOL_WIDTH-1;
    bndToolButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_ALL,
        BND_DEFAULT,BND_ICONID(3,10),NULL);
    x += BND_TOOL_WIDTH-1;
    bndToolButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_ALL,
        BND_DEFAULT,BND_ICONID(4,10),NULL);
    x += BND_TOOL_WIDTH-1;
    bndToolButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_LEFT,
        BND_DEFAULT,BND_ICONID(5,10),NULL);
    x += BND_TOOL_WIDTH-1;
    x += 5;
    bndRadioButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_RIGHT,
        BND_DEFAULT,BND_ICONID(0,11),NULL);
    x += BND_TOOL_WIDTH-1;
    bndRadioButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_ALL,
        BND_DEFAULT,BND_ICONID(1,11),NULL);
    x += BND_TOOL_WIDTH-1;
    bndRadioButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_ALL,
        BND_DEFAULT,BND_ICONID(2,11),NULL);
    x += BND_TOOL_WIDTH-1;
    bndRadioButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_ALL,
        BND_DEFAULT,BND_ICONID(3,11),NULL);
    x += BND_TOOL_WIDTH-1;
    bndRadioButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_ALL,
        BND_ACTIVE,BND_ICONID(4,11),NULL);
    x += BND_TOOL_WIDTH-1;
    bndRadioButton(vg,x,y,BND_TOOL_WIDTH,BND_WIDGET_HEIGHT,BND_CORNER_LEFT,
        BND_DEFAULT,BND_ICONID(5,11),NULL);
    
    // some OUI stuff

    // some persistent variables for demonstration
    static int enum1 = 0;
    static float progress1 = 0.25f;
    static float progress2 = 0.75f;
    static int option1 = 1;
    static int option2 = 0;
    static int option3 = 0;
    
    uiClear();
    
    int root = panel();
    // position root element
    uiSetLayout(0,UI_LEFT|UI_TOP);
    uiSetMargins(0,600,10,0,0);
    uiSetSize(0,250,400);
    
    int col = column(root);
    uiSetMargins(col, 10, 10, 10, 10);
    uiSetLayout(col, UI_TOP|UI_HFILL);
    
    button(col, 1, BND_ICONID(6,3), "Item 1", demohandler);
    button(col, 2, BND_ICONID(6,3), "Item 2", demohandler);
    
    {
        int h = hgroup(col);
        radio(h, 3, BND_ICONID(6,3), "Item 3.0", &enum1);
        radio(h, 4, BND_ICONID(0,10), NULL, &enum1);
        radio(h, 5, BND_ICONID(1,10), NULL, &enum1);
        radio(h, 6, BND_ICONID(6,3), "Item 3.3", &enum1);
    }
    
    {
        int rows = row(col);
        int coll = vgroup(rows);
        label(coll, -1, "Items 4.0:");
        coll = vgroup(coll);
        button(coll, 7, BND_ICONID(6,3), "Item 4.0.0", demohandler);
        button(coll, 8, BND_ICONID(6,3), "Item 4.0.1", demohandler);
        int colr = vgroup(rows);
        uiSetFrozen(colr, option1);
        label(colr, -1, "Items 4.1:");
        colr = vgroup(colr);
        slider(colr, 9, "Item 4.1.0", &progress1);
        slider(colr,10, "Item 4.1.1", &progress2);
    }
    
    button(col, 11, BND_ICONID(6,3), "Item 5", NULL);
    
    check(col, 12, "Frozen", &option1);
    check(col, 13, "Item 7", &option2);
    check(col, 14, "Item 8", &option3);
    
    static char textbuffer[32] = "click and edit";
    
    textbox(col, (UIhandle)textbuffer, textbuffer, 32);
    
    uiLayout();
    drawUI(vg, 0, 0, 0);
    
    UIvec2 cursor = uiGetCursor();
    cursor.x -= w/2;
    cursor.y -= h/2;
    if (abs(cursor.x) > (w/4)) {
        bndJoinAreaOverlay(vg, 0, 0, w, h, 0, (cursor.x > 0));
    } else if (abs(cursor.y) > (h/4)) {
        bndJoinAreaOverlay(vg, 0, 0, w, h, 1, (cursor.y > 0));
    }
    
    uiProcess();
}

////////////////////////////////////////////////////////////////////////////////

void errorcb(int error, const char* desc)
{
	printf("GLFW error %d: %s\n", error, desc);
}

static void mousebutton(GLFWwindow *window, int button, int action, int mods) {
	NVG_NOTUSED(window);
	NVG_NOTUSED(mods);
    uiSetButton(button, (action==GLFW_PRESS)?1:0);
}

static void cursorpos(GLFWwindow *window, double x, double y) {
	NVG_NOTUSED(window);
    uiSetCursor((int)x,(int)y);
}

static void charevent(GLFWwindow *window, unsigned int value) {
	NVG_NOTUSED(window);
    uiSetChar(value);
}

static void key(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	NVG_NOTUSED(scancode);
	NVG_NOTUSED(mods);
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
    uiSetKey(key, mods, action);
}

int main()
{
	GLFWwindow* window;
	struct NVGcontext* vg = NULL;
    UIcontext *uictx;
    
    uictx = uiCreateContext();
    uiMakeCurrent(uictx);

	if (!glfwInit()) {
		printf("Failed to init GLFW.");
		return -1;
	}

	glfwSetErrorCallback(errorcb);
#ifndef _WIN32 // don't require this on win32, and works with more cards
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);

	window = glfwCreateWindow(1000, 600, "OUI Blendish Demo", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	glfwSetKeyCallback(window, key);
    glfwSetCharCallback(window, charevent);
    glfwSetCursorPosCallback(window, cursorpos);
    glfwSetMouseButtonCallback(window, mousebutton);    

	glfwMakeContextCurrent(window);
#ifdef NANOVG_GLEW
	glewExperimental = GL_TRUE;
	if(glewInit() != GLEW_OK) {
		printf("Could not init glew.\n");
		return -1;
	}
	// GLEW generates GL error because it calls glGetString(GL_EXTENSIONS), we'll consume it here.
	glGetError();
#endif

	vg = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
	if (vg == NULL) {
		printf("Could not init nanovg.\n");
		return -1;
	}
	
	init(vg);

	glfwSwapInterval(0);

	glfwSetTime(0);

	while (!glfwWindowShouldClose(window))
	{
		double mx, my;
		int winWidth, winHeight;
		int fbWidth, fbHeight;
		float pxRatio;

		glfwGetCursorPos(window, &mx, &my);
		glfwGetWindowSize(window, &winWidth, &winHeight);
		glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
		// Calculate pixel ration for hi-dpi devices.
		pxRatio = (float)fbWidth / (float)winWidth;

		// Update and render
		glViewport(0, 0, fbWidth, fbHeight);
		glClearColor(0,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

		nvgBeginFrame(vg, winWidth, winHeight, pxRatio);

        draw(vg, winWidth, winHeight);

		nvgEndFrame(vg);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

    uiDestroyContext(uictx);

	nvgDeleteGL3(vg);

	glfwTerminate();
	return 0;
}
