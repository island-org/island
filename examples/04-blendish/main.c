#define SKETCH_2D_IMPLEMENTATION
#include "../01-processing/sketch2d.h"
#include <blendish.h>
#include <oui.h>

PImage img1, img2;
PFont font;


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

////////////////////////////////////////////////////////////////////////////////

// calculate which corners are sharp for an item, depending on whether
// the container the item is in has negative spacing, and the item
// is first or last element in a sequence of 2 or more elements.
int cornerFlags(int item) {
    int parent = uiParent(item);
    int numkids = uiGetChildCount(parent);
    const UIData *head;

    if (numkids < 2) return BND_CORNER_NONE;
    head = (const UIData *)uiGetData(parent);
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
        nvgGlobalAlpha(vg, 0.5);
    }
    if (head) {
        switch(head->subtype) {
            default: {
                testrect(vg,rect);
                     } break;
            case ST_LABEL: {
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
                sprintf(value,"%.0f%%", (int)(*data->progress)*100.0f);
                bndSlider(vg,rect.x,rect.y,rect.w,rect.h,
                    cornerFlags(item),state,
                    *data->progress,data->label,value);
                           } break;
        }
    } else {
        testrect(vg,rect);
    }

    {
        int kid = uiFirstChild(item);
        while (kid > 0) {
            drawUI(vg, kid, rect.x, rect.y);
            kid = uiNextSibling(kid);
        }
    }

    if (uiGetState(item) == UI_FROZEN) {
        nvgGlobalAlpha(vg, 1.0);
    }
}

int label(int parent, int iconid, const char *label) {    
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

// simple logic for a radio button
void radiohandler(int item, UIevent event) {
    UIRadioData *data = (UIRadioData *)uiGetData(item);
    *data->value = uiGetChildId(item);
}

int radio(int parent, UIhandle handle, int iconid, const char *label, int *value) {
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

void drawBlendish(NVGcontext *vg, float w, float h) {
    int rx, ry, rw;
    int mx, my, mw;
    int ox;
    float s_offset, s_size;
    int col;

    // some persistent variables for demonstration
    static int enum1 = 0;
    static float progress1 = 0.25f;
    static float progress2 = 0.75f;
    static int option1 = 1;
    static int option2 = 0;
    static int option3 = 0;

    int x = 10;
    int y = 10;

    bndBackground(vg, 0, 0, w, h);

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
    ry = y;
    rx = x;

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

    mx = x-30;
    my = y-12;
    mw = 120;
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
    ox = x;
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
    {
        float progress_value;
        char progress_label[32];

        progress_value = fmodf(glfwGetTime()/10.0,1.0);
        sprintf(progress_label, "%d%%", (int)(progress_value*100+0.5f));
        bndSlider(vg,x,y,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_DEFAULT,
            progress_value,"Default",progress_label);
        y += 25;
        bndSlider(vg,x,y,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_HOVER,
            progress_value,"Hovered",progress_label);
        y += 25;
        bndSlider(vg,x,y,240,BND_WIDGET_HEIGHT,BND_CORNER_NONE,BND_ACTIVE,
            progress_value,"Active",progress_label);
    }


    rw = x+240-rx;
    s_offset = sinf(glfwGetTime()/2.0)*0.5+0.5;
    s_size = cosf(glfwGetTime()/3.11)*0.5+0.5;

    bndScrollBar(vg,rx,ry,rw,BND_SCROLLBAR_HEIGHT,BND_DEFAULT,s_offset,s_size);
    ry += 20;
    bndScrollBar(vg,rx,ry,rw,BND_SCROLLBAR_HEIGHT,BND_HOVER,s_offset,s_size);
    ry += 20;
    bndScrollBar(vg,rx,ry,rw,BND_SCROLLBAR_HEIGHT,BND_ACTIVE,s_offset,s_size);

    {
        const char edit_text[] = "The quick brown fox";
        int textlen = strlen(edit_text)+1;
        int t = glfwGetTime()*2;
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
    }

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

    uiClear();

    {
        int root = uiItem(); 
        // position root element
        uiSetLayout(0,UI_LEFT|UI_TOP);
        uiSetMargins(0,600,10,0,0);
        uiSetSize(0,250,400);
    }


    col = column(0);
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
        int colr;
        int rows = row(col);
        int coll = vgroup(rows);
        label(coll, -1, "Items 4.0:");
        coll = vgroup(coll);
        button(coll, 7, BND_ICONID(6,3), "Item 4.0.0", demohandler);
        button(coll, 8, BND_ICONID(6,3), "Item 4.0.1", demohandler);
        colr = vgroup(rows);
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

    uiProcess();

    drawUI(vg, 0, 0, 0);
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

static void keyCb(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    NVG_NOTUSED(scancode);
    NVG_NOTUSED(mods);
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

UIcontext *uictx;

void setup()
{
    size(1024, 768);
    img1 = loadImage("../3rdparty/nanovg/example/images/image9.jpg");
    img2 = loadImage("../3rdparty/nanovg/example/images/image10.jpg");
    font = loadFont("../3rdparty/nanovg/example/Roboto-Regular.ttf");

    uictx = uiCreateContext();
    uiMakeCurrent(uictx);

    glfwSetKeyCallback(window, keyCb);
    glfwSetCursorPosCallback(window, cursorpos);
    glfwSetMouseButtonCallback(window, mousebutton);

    bndSetFont(nvgCreateFont(vg, "system", "../3rdparty/blendish/DejaVuSans.ttf"));
    bndSetIconImage(nvgCreateImage(vg, "../3rdparty/blendish/blender_icons16.png", NVG_IMAGE_GENERATE_MIPMAPS));
}

void draw()
{
    unsigned char mbut = 0;
    int scrollarea1 = 0;

    background(gray(122));

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

    drawBlendish(vg, width, height);
}

void shutdown()
{
    uiDestroyContext(uictx);
}
