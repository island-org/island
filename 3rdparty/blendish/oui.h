/*
OUI - A minimal semi-immediate GUI handling & layouting library

Copyright (c) 2014 Leonard Ritter <leonard.ritter@duangle.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _OUI_H_
#define _OUI_H_

/*
Revision 2 (2014-07-13)

OUI (short for "Open UI", spoken like the french "oui" for "yes") is a
platform agnostic single-header C library for layouting GUI elements and
handling related user input. Together with a set of widget drawing and logic 
routines it can be used to build complex user interfaces.

OUI is a semi-immediate GUI. Widget declarations are persistent for the duration
of the setup and evaluation, but do not need to be kept around longer than one
frame.

OUI has no widget types; instead, it provides only one kind of element, "Items",
which can be taylored to the application by the user and expanded with custom
buffers and event handlers to behave as containers, buttons, sliders, radio
buttons, and so on.

OUI also does not draw anything; Instead it provides a set of functions to
iterate and query the layouted items in order to allow client code to render
each widget with its current state using a preferred graphics library.

A basic setup for OUI usage looks like this:

void app_main(...) {
    UIcontext *context = uiCreateContext();
    uiMakeCurrent(context);

    while (app_running()) {
        // update position of mouse cursor; the ui can also be updated
        // from received events.
        uiSetCursor(app_get_mouse_x(), app_get_mouse_y());
        // update button state
        for (int i = 0; i < 3; ++i)
            uiSetButton(i, app_get_button_state(i));
        
        // begin new UI declarations
        uiClear();
        
        // - UI setup code goes here -
        app_setup_ui();

        // layout UI
        uiLayout();

        // draw UI
        app_draw_ui(render_context,0,0,0);
        
        // update states and fire handlers
        uiProcess();
    }

    uiDestroyContext(context);
}

Here's an example setup for a checkbox control:

typedef struct CheckBoxData {
    int type;
    const char *label;
    bool *checked;
} CheckBoxData;

// called when the item is clicked (see checkbox())
void app_checkbox_handler(int item, UIevent event) {
    
    // retrieve custom data (see checkbox())
    const CheckBoxData *data = (const CheckBoxData *)uiGetData(item);
    
    // toggle value
    *data->checked = !(*data->checked);
}

// creates a checkbox control for a pointer to a boolean and attaches it to 
// a parent item.
int checkbox(int parent, UIhandle handle, const char *label, bool *checked) {
    
    // create new ui item
    int item = uiItem(); 
    
    // set persistent handle for item that is used
    // to track activity over time
    uiSetHandle(item, handle);
    
    // set size of wiget; horizontal size is dynamic, vertical is fixed
    uiSetSize(item, 0, APP_WIDGET_HEIGHT);
    
    // attach checkbox handler, set to fire as soon as the left button is
    // pressed; UI_BUTTON0_HOT_UP is also a popular alternative.
    uiSetHandler(item, app_checkbox_handler, UI_BUTTON0_DOWN);
    
    // store some custom data with the checkbox that we use for rendering
    // and value changes.
    CheckBoxData *data = (CheckBoxData *)uiAllocData(item, sizeof(CheckBoxData));
    // assign a custom typeid to the data so the renderer knows how to
    // render this control.
    data->type = APP_WIDGET_CHECKBOX;
    data->label = label;
    data->checked = checked;
    
    // append to parent
    uiAppend(parent, item);
    
    return item;
}

A simple recursive drawing routine can look like this:

void app_draw_ui(AppRenderContext *ctx, int item, int x, int y) {
    // retrieve custom data and cast it to an int; we assume the first member
    // of every widget data item to be an "int type" field.
    const int *type = (const int *)uiGetData(item);
    
    // get the widgets relative rectangle and offset by the parents
    // absolute position.
    UIrect rect = uiGetRect(item);
    rect.x += x;
    rect.y += y; 
    
    // if a type is set, this is a specialized widget
    if (type) {
        switch(*type) {
            default: break;
            case APP_WIDGET_LABEL: {
                // ...
            } break;
            case APP_WIDGET_BUTTON: {
                // ...
            } break;
            case APP_WIDGET_CHECKBOX: {
                // cast to the full data type
                const CheckBoxData *data = (CheckBoxData*)type;
                
                // get the widgets current state
                int state = uiGetState(item);
                
                // if the value is set, the state is always active
                if (*data->checked)
                    state = UI_ACTIVE;
                
                // draw the checkbox
                app_draw_checkbox(ctx, rect, state, data->label);
            } break;
        }
    }
    
    // iterate through all children and draw
    int kid = uiFirstChild(item);
    while (kid >= 0) {
        app_draw_ui(ctx, kid, rect.x, rect.y);
        kid = uiNextSibling(kid);
    }
}

See example.cpp in the repository for a full usage example.

*/

// limits

// maximum number of items that may be added (must be power of 2)
#define UI_MAX_ITEMS 4096
// maximum size in bytes reserved for storage of application dependent data
// as passed to uiAllocData().
#define UI_MAX_BUFFERSIZE 1048576
// maximum size in bytes of a single data buffer passed to uiAllocData().
#define UI_MAX_DATASIZE 4096
// maximum depth of nested containers
#define UI_MAX_DEPTH 64
// maximum number of buffered input events
#define UI_MAX_INPUT_EVENTS 64

typedef unsigned int UIuint;

// opaque UI context
typedef struct UIcontext UIcontext;

// application defined context handle
typedef unsigned long long UIhandle;

// item states as returned by uiGetState()

typedef enum UIitemState {
    // the item is inactive
    UI_COLD = 0,
    // the item is inactive, but the cursor is hovering over this item
    UI_HOT = 1,
    // the item is toggled, activated, focused (depends on item kind)
    UI_ACTIVE = 2,
    // the item is unresponsive
    UI_FROZEN = 3,
} UIitemState;

// layout flags
typedef enum UIlayoutFlags {
    // anchor to left item or left side of parent
    UI_LEFT = 1,
    // anchor to top item or top side of parent
    UI_TOP = 2,
    // anchor to right item or right side of parent
    UI_RIGHT = 4,
    // anchor to bottom item or bottom side of parent
    UI_DOWN = 8,
    // anchor to both left and right item or parent borders
    UI_HFILL = 5,
    // anchor to both top and bottom item or parent borders
    UI_VFILL = 10,
    // center horizontally, with left margin as offset
    UI_HCENTER = 0,
    // center vertically, with top margin as offset
    UI_VCENTER = 0,
    // center in both directions, with left/top margin as offset
    UI_CENTER = 0,
    // anchor to all four directions
    UI_FILL = 15,
} UIlayoutFlags;

// event flags
typedef enum UIevent {
    // on button 0 down
    UI_BUTTON0_DOWN = 0x0001,
    // on button 0 up
    // when this event has a handler, uiGetState() will return UI_ACTIVE as
    // long as button 0 is down.
    UI_BUTTON0_UP = 0x0002,
    // on button 0 up while item is hovered
    // when this event has a handler, uiGetState() will return UI_ACTIVE
    // when the cursor is hovering the items rectangle; this is the
    // behavior expected for buttons.
    UI_BUTTON0_HOT_UP = 0x0004,
    // item is being captured (button 0 constantly pressed); 
    // when this event has a handler, uiGetState() will return UI_ACTIVE as
    // long as button 0 is down.
    UI_BUTTON0_CAPTURE = 0x0008,
    // on button 2 down (right mouse button, usually triggers context menu)
    UI_BUTTON2_DOWN = 0x0010,
    // item has received a new child
    // this can be used to allow container items to configure child items
    // as they appear.
    UI_APPEND = 0x0100,
    // item is focused and has received a key-down event
    // the respective key can be queried using uiGetKey() and uiGetModifier()
    UI_KEY_DOWN = 0x0200,
    // item is focused and has received a key-up event
    // the respective key can be queried using uiGetKey() and uiGetModifier()
    UI_KEY_UP = 0x0400,
    // item is focused and has received a character event
    // the respective character can be queried using uiGetKey()
    UI_CHAR = 0x0800,
    // item is requested to fill a container with additional items,
    // usually menuitems for a context menu.
    // the respective container can be queried using uiGetExtendItem()
    UI_EXTEND = 0x1000,
} UIevent;

// handler callback; event is one of UI_EVENT_*
typedef void (*UIhandler)(int item, UIevent event);

// for cursor positions, mainly
typedef struct UIvec2 {
    union {
        int v[2];
        struct { int x, y; };
    };
} UIvec2;

// layout rectangle
typedef struct UIrect {
    union {
        int v[4];
        struct { int x, y, w, h; };
    };
} UIrect;

// unless declared otherwise, all operations have the complexity O(1).

// Context Management
// ------------------

// create a new UI context; call uiMakeCurrent() to make this context the
// current context. The context is managed by the client and must be released
// using uiDestroyContext()
UIcontext *uiCreateContext();

// select an UI context as the current context; a context must always be 
// selected before using any of the other UI functions
void uiMakeCurrent(UIcontext *ctx);

// release the memory of an UI context created with uiCreateContext(); if the
// context is the current context, the current context will be set to NULL
void uiDestroyContext(UIcontext *ctx);

// Input Control
// -------------

// sets the current cursor position (usually belonging to a mouse) to the
// screen coordinates at (x,y)
void uiSetCursor(int x, int y);

// returns the current cursor position in screen coordinates as set by 
// uiSetCursor()
UIvec2 uiGetCursor();

// returns the offset of the cursor relative to the last call to uiProcess()
UIvec2 uiGetCursorDelta();

// returns the beginning point of a drag operation.
UIvec2 uiGetCursorStart();

// returns the offset of the cursor relative to the beginning point of a drag
// operation.
UIvec2 uiGetCursorStartDelta();

// sets a mouse or gamepad button as pressed/released
// button is in the range 0..63 and maps to an application defined input
// source.
// enabled is 1 for pressed, 0 for released
void uiSetButton(int button, int enabled);

// returns the current state of an application dependent input button
// as set by uiSetButton().
// the function returns 1 if the button has been set to pressed, 0 for released.
int uiGetButton(int button);

// sets a key as down/up; the key can be any application defined keycode
// mod is an application defined set of flags for modifier keys
// enabled is 1 for key down, 0 for key up
// all key events are being buffered until the next call to uiProcess()
void uiSetKey(unsigned int key, unsigned int mod, int enabled);

// sends a single character for text input; the character is usually in the
// unicode range, but can be application defined.
// all char events are being buffered until the next call to uiProcess()
void uiSetChar(unsigned int value);


// Stages
// ------

// clear the item buffer; uiClear() should be called before the first 
// UI declaration for this frame to avoid concatenation of the same UI multiple 
// times.
// After the call, all previously declared item IDs are invalid, and all
// application dependent context data has been freed.
void uiClear();

// layout all added items starting from the root item 0.
// after calling uiLayout(), no further modifications to the item tree should
// be done until the next call to uiClear().
// It is safe to immediately draw the items after a call to uiLayout().
// this is an O(N) operation for N = number of declared items.
void uiLayout();

// update the internal state according to the current cursor position and 
// button states, and call all registered handlers.
// after calling uiProcess(), no further modifications to the item tree should
// be done until the next call to uiClear().
// Items should be drawn before a call to uiProcess()
// this is an O(N) operation for N = number of declared items.
void uiProcess();

// UI Declaration
// --------------

// create a new UI item and return the new items ID.
int uiItem();

// set an items state to frozen; the UI will not recurse into frozen items
// when searching for hot or active items; subsequently, frozen items and
// their child items will not cause mouse event notifications.
// The frozen state is not applied recursively; uiGetState() will report
// UI_COLD for child items. Upon encountering a frozen item, the drawing
// routine needs to handle rendering of child items appropriately.
// see example.cpp for a demonstration.
void uiSetFrozen(int item, int enable);

// set the application-dependent handle of an item.
// handle is an application defined 64-bit handle. If handle is 0, the item
// will not be interactive.
void uiSetHandle(int item, UIhandle handle);

// assigns the items own address as handle; this may cause glitches
// when the order of items changes while theitem is captured
void uiSetSelfHandle(int item);

// allocate space for application-dependent context data and return the pointer
// if successful. If no data has been allocated, a new pointer is returned. 
// Otherwise, an assertion is thrown.
// The memory of the pointer is managed by the UI context.
void *uiAllocData(int item, int size);

// set the handler callback for an interactive item. 
// flags is a combination of UI_EVENT_* and designates for which events the 
// handler should be called. 
void uiSetHandler(int item, UIhandler handler, int flags);

// assign an item to a container.
// an item ID of 0 refers to the root item.
// if child is already assigned to a parent, an assertion will be thrown.
// the function returns the child item ID
int uiAppend(int item, int child);

// set the size of the item; a size of 0 indicates the dimension to be 
// dynamic; if the size is set, the item can not expand beyond that size.
void uiSetSize(int item, int w, int h);

// set the anchoring behavior of the item to one or multiple UIlayoutFlags
void uiSetLayout(int item, int flags);

// set the left, top, right and bottom margins of an item; when the item is
// anchored to the parent or another item, the margin controls the distance
// from the neighboring element.
void uiSetMargins(int item, int l, int t, int r, int b);

// anchor the item to another sibling within the same container, so that the
// sibling is left to this item.
void uiSetRelToLeft(int item, int other);
// anchor the item to another sibling within the same container, so that the
// sibling is above this item.
void uiSetRelToTop(int item, int other);
// anchor the item to another sibling within the same container, so that the
// sibling is right to this item.
void uiSetRelToRight(int item, int other);
// anchor the item to another sibling within the same container, so that the
// sibling is below this item.
void uiSetRelToDown(int item, int other);

// set item as recipient of all keyboard events; the item must have a handle
// assigned; if item is -1, no item will be focused.
void uiFocus(int item);

// Iteration
// ---------

// returns the first child item of a container item. If the item is not
// a container or does not contain any items, -1 is returned.
// if item is 0, the first child item of the root item will be returned.
int uiFirstChild(int item);

// returns the last child item of a container item. If the item is not
// a container or does not contain any items, -1 is returned.
// if item is 0, the last child item of the root item will be returned.
int uiLastChild(int item);

// returns an items parent container item.
// if item is 0, -1 will be returned.
int uiParent(int item);

// returns an items next sibling in the list of the parent containers children.
// if item is 0 or the item is the last child item, -1 will be returned.
int uiNextSibling(int item);
// returns an items previous sibling in the list of the parent containers
// children.
// if item is 0 or the item is the first child item, -1 will be returned.
int uiPrevSibling(int item);

// Querying
// --------

// return the current state of the item. This state is only valid after
// a call to uiProcess().
// The returned value is one of UI_COLD, UI_HOT, UI_ACTIVE, UI_FROZEN.
UIitemState uiGetState(int item);

// return the application-dependent handle of the item as passed to uiSetHandle().
UIhandle uiGetHandle(int item);

// return the item with the given application-dependent handle as assigned by
// uiSetHandle() or -1 if unsuccessful.
int uiGetItem(UIhandle handle);

// return the item that is currently under the cursor or -1 for none
int uiGetHotItem();

// return the item that is currently focused or -1 for none
int uiGetFocusedItem();

// return the application-dependent context data for an item as passed to
// uiAllocData(). The memory of the pointer is managed by the UI context
// and should not be altered.
const void *uiGetData(int item);

// return the handler callback for an item as passed to uiSetHandler()
UIhandler uiGetHandler(int item);
// return the handler flags for an item as passed to uiSetHandler()
int uiGetHandlerFlags(int item);
// when handling a KEY_DOWN/KEY_UP event: the key that triggered this event
unsigned int uiGetKey();
// when handling a KEY_DOWN/KEY_UP event: the key that triggered this event
unsigned int uiGetModifier();
// when handling an EXTEND event; the container which to add items to
int uiGetExtendItem();

// returns the number of child items a container item contains. If the item 
// is not a container or does not contain any items, 0 is returned.
// if item is 0, the child item count of the root item will be returned.
int uiGetChildCount(int item);

// returns an items child index relative to its parent. If the item is the
// first item, the return value is 0; If the item is the last item, the return
// value is equivalent to uiGetChildCount(uiParent(item))-1.
// if item is 0, 0 will be returned.
int uiGetChildId(int item);

// returns the items layout rectangle relative to the parent. If uiGetRect()
// is called before uiLayout(), the values of the returned rectangle are
// undefined.
UIrect uiGetRect(int item);

// returns the items layout rectangle in absolute coordinates. If 
// uiGetAbsoluteRect() is called before uiLayout(), the values of the returned
// rectangle are undefined.
// This function has complexity O(N) for N parents
UIrect uiGetAbsoluteRect(int item);

// returns 1 if an items absolute rectangle contains a given coordinate
// otherwise 0
int uiContains(int item, int x, int y);

// when called from an input event handler, returns the active items absolute
// layout rectangle. If uiGetActiveRect() is called outside of a handler,
// the values of the returned rectangle are undefined.
UIrect uiGetActiveRect();

// return the width of the item as set by uiSetSize()
int uiGetWidth(int item);
// return the height of the item as set by uiSetSize()
int uiGetHeight(int item);

// return the anchoring behavior as set by uiSetLayout()
int uiGetLayout(int item);

// return the left margin of the item as set with uiSetMargins()
int uiGetMarginLeft(int item);
// return the top margin of the item as set with uiSetMargins()
int uiGetMarginTop(int item);
// return the right margin of the item as set with uiSetMargins()
int uiGetMarginRight(int item);
// return the bottom margin of the item as set with uiSetMargins()
int uiGetMarginDown(int item);

// return the items anchored sibling as assigned with uiSetRelToLeft() 
// or -1 if not set.
int uiGetRelToLeft(int item);
// return the items anchored sibling as assigned with uiSetRelToTop() 
// or -1 if not set.
int uiGetRelToTop(int item);
// return the items anchored sibling as assigned with uiSetRelToRight() 
// or -1 if not set.
int uiGetRelToRight(int item);
// return the items anchored sibling as assigned with uiSetRelToBottom() 
// or -1 if not set.
int uiGetRelToDown(int item);

#endif // _OUI_H_

#ifdef OUI_IMPLEMENTATION

#include <assert.h>

#ifdef _MSC_VER
	#pragma warning (disable: 4996) // Switch off security warnings
	#pragma warning (disable: 4100) // Switch off unreferenced formal parameter warnings
	#ifdef __cplusplus
	#define UI_INLINE inline
	#else
	#define UI_INLINE
	#endif
#else
	#define UI_INLINE inline
#endif

#define UI_MAX_KIND 16

#define UI_ANY_INPUT (UI_BUTTON0_DOWN \
    |UI_BUTTON0_UP \
    |UI_BUTTON0_HOT_UP \
    |UI_BUTTON0_CAPTURE \
    |UI_BUTTON2_DOWN \
    |UI_KEY_DOWN \
    |UI_KEY_UP \
    |UI_CHAR)

typedef struct UIitem {
    // declaration independent unique handle (for persistence)
    UIhandle handle;
    // handler
    UIhandler handler;
    
    // container structure
    
    // number of kids
    int numkids;
    // index of first kid
    int firstkid;
    // index of last kid
    int lastkid;
    
    // child structure
    
    // parent item
    int parent;
    // index of kid relative to parent
    int kidid;
    // index of next sibling with same parent
    int nextitem;
    // index of previous sibling with same parent
    int previtem;
    
    // one or multiple of UIlayoutFlags
    int layout_flags;
    // size
    UIvec2 size;
    // visited flags for layouting
    int visited;
    // margin offsets, interpretation depends on flags
    int margins[4];
    // neighbors to position borders to
    int relto[4];
    
    // computed size
    UIvec2 computed_size;
    // relative rect
    UIrect rect;
    
    // attributes
    
    int frozen;
    // index of data or -1 for none
    int data;
    // size of data
    int datasize;
    // a combination of UIevents
    int event_flags;
} UIitem;

typedef enum UIstate {
    UI_STATE_IDLE = 0,
    UI_STATE_CAPTURE,
} UIstate;

typedef struct UIhandleEntry {
    unsigned int key;
    int item;
} UIhandleEntry;

typedef struct UIinputEvent {
    unsigned int key;
    unsigned int mod;
    UIevent event;
} UIinputEvent;

struct UIcontext {
    // button state in this frame
    unsigned long long buttons;
    // button state in the previous frame
    unsigned long long last_buttons;
    
    // where the cursor was at the beginning of the active state
    UIvec2 start_cursor;
    // where the cursor was last frame
    UIvec2 last_cursor;
    // where the cursor is currently
    UIvec2 cursor;
    
    UIhandle hot_handle;
    UIhandle active_handle;
    UIhandle focus_handle;
    UIrect hot_rect;
    UIrect active_rect;
    UIstate state;
    int hot_item;
    unsigned int active_key;
    unsigned int active_modifier;
    int extend_item;
    
    int count;    
    int datasize;
    int eventcount;
    
    UIitem items[UI_MAX_ITEMS];    
    unsigned char data[UI_MAX_BUFFERSIZE];
    UIhandleEntry handles[UI_MAX_ITEMS];    
    UIinputEvent events[UI_MAX_INPUT_EVENTS];
};

UI_INLINE int ui_max(int a, int b) {
    return (a>b)?a:b;
}

UI_INLINE int ui_min(int a, int b) {
    return (a<b)?a:b;
}

static UIcontext *ui_context = NULL;

UIcontext *uiCreateContext() {
    UIcontext *ctx = (UIcontext *)malloc(sizeof(UIcontext));
    memset(ctx, 0, sizeof(UIcontext));
    return ctx;
}

void uiMakeCurrent(UIcontext *ctx) {
    ui_context = ctx;
    if (ui_context)
        uiClear();
}

void uiDestroyContext(UIcontext *ctx) {
    if (ui_context == ctx)
        uiMakeCurrent(NULL);
    free(ctx);
}

UI_INLINE unsigned int uiHashHandle(UIhandle handle) {
    handle = (handle+(handle>>32)) & 0xffffffff;
    unsigned int x = (unsigned int)handle;
    x += (x>>6)+(x>>19);
    x += x<<16;
    x ^= x<<3;
    x += x>>5;
    x ^= x<<2;
    x += x>>15;
    x ^= x<<10;
    return x?x:1; // must not be zero
}

UI_INLINE unsigned int uiHashProbeDistance(unsigned int key, unsigned int slot_index) {
    unsigned int pos = key & (UI_MAX_ITEMS-1);
    return (slot_index + UI_MAX_ITEMS - pos) & (UI_MAX_ITEMS-1);
}

UI_INLINE UIhandleEntry *uiHashLookupHandle(unsigned int key) {
    assert(ui_context);
    int pos = key & (UI_MAX_ITEMS-1);
    unsigned int dist = 0;
    for (;;) {
        UIhandleEntry *entry = ui_context->handles + pos;
        unsigned int pos_key = entry->key;
        if (!pos_key) return NULL;
        else if (entry->key == key)
            return entry;
        else if (dist > uiHashProbeDistance(pos_key, pos))
            return NULL;
        pos = (pos+1) & (UI_MAX_ITEMS-1);
        ++dist;
    }
}

int uiGetItem(UIhandle handle) {
    unsigned int key = uiHashHandle(handle);
    UIhandleEntry *e = uiHashLookupHandle(key);
    return e?(e->item):-1;
}

static void uiHashInsertHandle(UIhandle handle, int item) {
    unsigned int key = uiHashHandle(handle);
    UIhandleEntry *e = uiHashLookupHandle(key);
    if (e) { // update
        e->item = item;
        return;
    }

    int pos = key & (UI_MAX_ITEMS-1);
    unsigned int dist = 0;
    for (unsigned int i = 0; i < UI_MAX_ITEMS; ++i) {
        int index = (pos + i) & (UI_MAX_ITEMS-1);
        unsigned int pos_key = ui_context->handles[index].key;
        if (!pos_key) {
            ui_context->handles[index].key = key;
            ui_context->handles[index].item = item;
            break;
        } else {
            unsigned int probe_distance = uiHashProbeDistance(pos_key, index);
            if (dist > probe_distance) {
                unsigned int oldkey = ui_context->handles[index].key;
                unsigned int olditem = ui_context->handles[index].item;                
                ui_context->handles[index].key = key;
                ui_context->handles[index].item = item;
                key = oldkey;
                item = olditem;
                dist = probe_distance;
            }
        }
        ++dist;
    }
}

void uiSetButton(int button, int enabled) {
    assert(ui_context);
    unsigned long long mask = 1ull<<button;
    // set new bit
    ui_context->buttons = (enabled)?
        (ui_context->buttons | mask):
        (ui_context->buttons & ~mask);
}

static void uiAddInputEvent(UIinputEvent event) {
    assert(ui_context);
    if (ui_context->eventcount == UI_MAX_INPUT_EVENTS) return;
    ui_context->events[ui_context->eventcount++] = event;
}

static void uiClearInputEvents() {
    assert(ui_context);
    ui_context->eventcount = 0;
}

void uiSetKey(unsigned int key, unsigned int mod, int enabled) {
    assert(ui_context);
    UIinputEvent event = { key, mod, enabled?UI_KEY_DOWN:UI_KEY_UP };
    uiAddInputEvent(event);
}

void uiSetChar(unsigned int value) {
    assert(ui_context);
    UIinputEvent event = { value, 0, UI_CHAR };
    uiAddInputEvent(event);
}

int uiGetLastButton(int button) {
    assert(ui_context);
    return (ui_context->last_buttons & (1ull<<button))?1:0;
}

int uiGetButton(int button) {
    assert(ui_context);
    return (ui_context->buttons & (1ull<<button))?1:0;
}

int uiButtonPressed(int button) {
    assert(ui_context);
    return !uiGetLastButton(button) && uiGetButton(button);
}

int uiButtonReleased(int button) {
    assert(ui_context);
    return uiGetLastButton(button) && !uiGetButton(button);
}

void uiSetCursor(int x, int y) {
    assert(ui_context);
    ui_context->cursor.x = x;
    ui_context->cursor.y = y;
}

UIvec2 uiGetCursor() {
    assert(ui_context);
    return ui_context->cursor;
}

UIvec2 uiGetCursorStart() {
    assert(ui_context);
    return ui_context->start_cursor;
}

UIvec2 uiGetCursorDelta() {
    assert(ui_context);
    UIvec2 result = {{{
        ui_context->cursor.x - ui_context->last_cursor.x,
        ui_context->cursor.y - ui_context->last_cursor.y
    }}};
    return result;
}

UIvec2 uiGetCursorStartDelta() {
    assert(ui_context);
    UIvec2 result = {{{
        ui_context->cursor.x - ui_context->start_cursor.x,
        ui_context->cursor.y - ui_context->start_cursor.y
    }}};
    return result;
}

unsigned int uiGetKey() {
    assert(ui_context);
    return ui_context->active_key;
}

unsigned int uiGetModifier() {
    assert(ui_context);
    return ui_context->active_modifier;
}

int uiGetExtendItem() {
    assert(ui_context);
    return ui_context->extend_item;
}

UIitem *uiItemPtr(int item) {
    assert(ui_context && (item >= 0) && (item < ui_context->count));
    return ui_context->items + item;
}

int uiGetHotItem() {
    assert(ui_context);
    return ui_context->hot_item;
}

void uiFocus(int item) {
    assert(ui_context && (item >= -1) && (item < ui_context->count));
    ui_context->focus_handle = (item < 0)?0:uiGetHandle(item);
}

int uiGetFocusedItem() {
    assert(ui_context);
    return ui_context->focus_handle?uiGetItem(ui_context->focus_handle):-1;
}

void uiClear() {
    assert(ui_context);
    ui_context->count = 0;
    ui_context->datasize = 0;
    ui_context->hot_item = -1;
    memset(ui_context->handles, 0, sizeof(ui_context->handles));
}

int uiItem() {
    assert(ui_context);
    assert(ui_context->count < UI_MAX_ITEMS);
    int idx = ui_context->count++;
    UIitem *item = uiItemPtr(idx);
    memset(item, 0, sizeof(UIitem));
    item->parent = -1;
    item->firstkid = -1;
    item->lastkid = -1;
    item->nextitem = -1;
    item->previtem = -1;
    item->data = -1;
    for (int i = 0; i < 4; ++i)
        item->relto[i] = -1;
    return idx;
}

void uiNotifyItem(int item, UIevent event) {
    UIitem *pitem = uiItemPtr(item);
    if (pitem->handler && (pitem->event_flags & event)) {
        pitem->handler(item, event);
    }
}

void uiExtend(int parent, int from_item) {
    assert(ui_context);
    ui_context->extend_item = parent;
    uiNotifyItem(from_item, UI_EXTEND);
}

int uiAppend(int item, int child) {
    assert(child > 0);
    assert(uiParent(child) == -1);
    UIitem *pitem = uiItemPtr(child);
    UIitem *pparent = uiItemPtr(item);
    pitem->parent = item;
    pitem->kidid = pparent->numkids++;
    if (pparent->lastkid < 0) {
        pparent->firstkid = child;
        pparent->lastkid = child;
    } else {
        pitem->previtem = pparent->lastkid;
        uiItemPtr(pparent->lastkid)->nextitem = child;
        pparent->lastkid = child;
    }
    uiNotifyItem(item, UI_APPEND);
    return child;
}

void uiSetFrozen(int item, int enable) {
    UIitem *pitem = uiItemPtr(item);
    pitem->frozen = enable;
}

void uiSetSize(int item, int w, int h) {
    UIitem *pitem = uiItemPtr(item);
    pitem->size.x = w;
    pitem->size.y = h;
}

int uiGetWidth(int item) {
    return uiItemPtr(item)->size.x;
}

int uiGetHeight(int item) {
    return uiItemPtr(item)->size.y;
}

void uiSetLayout(int item, int flags) {
    uiItemPtr(item)->layout_flags = flags;
}

int uiGetLayout(int item) {
    return uiItemPtr(item)->layout_flags;
}

void uiSetMargins(int item, int l, int t, int r, int b) {
    UIitem *pitem = uiItemPtr(item);
    pitem->margins[0] = l;
    pitem->margins[1] = t;
    pitem->margins[2] = r;
    pitem->margins[3] = b;
}

int uiGetMarginLeft(int item) {
    return uiItemPtr(item)->margins[0];
}
int uiGetMarginTop(int item) {
    return uiItemPtr(item)->margins[1];
}
int uiGetMarginRight(int item) {
    return uiItemPtr(item)->margins[2];
}
int uiGetMarginDown(int item) {
    return uiItemPtr(item)->margins[3];
}


void uiSetRelToLeft(int item, int other) {
    assert((other < 0) || (uiParent(other) == uiParent(item)));
    uiItemPtr(item)->relto[0] = other;
}

int uiGetRelToLeft(int item) {
    return uiItemPtr(item)->relto[0];
}

void uiSetRelToTop(int item, int other) {
    assert((other < 0) || (uiParent(other) == uiParent(item)));
    uiItemPtr(item)->relto[1] = other;
}
int uiGetRelToTop(int item) {
    return uiItemPtr(item)->relto[1];
}

void uiSetRelToRight(int item, int other) {
    assert((other < 0) || (uiParent(other) == uiParent(item)));
    uiItemPtr(item)->relto[2] = other;
}
int uiGetRelToRight(int item) {
    return uiItemPtr(item)->relto[2];
}

void uiSetRelToDown(int item, int other) {
    assert((other < 0) || (uiParent(other) == uiParent(item)));
    uiItemPtr(item)->relto[3] = other;
}
int uiGetRelToDown(int item) {
    return uiItemPtr(item)->relto[3];
}


UI_INLINE void uiComputeChainSize(UIitem *pkid, 
    int *need_size, int *hard_size, int dim) {
    UIitem *pitem = pkid;
    int wdim = dim+2;
    int size = pitem->rect.v[wdim] + pitem->margins[dim] + pitem->margins[wdim];
    *need_size = size;
    *hard_size = pitem->size.v[dim]?size:0;
    
    int it = 0;
    pitem->visited |= 1<<dim;
    // traverse along left neighbors
    while ((pitem->layout_flags>>dim) & UI_LEFT) {
        if (pitem->relto[dim] < 0) break;
        pitem = uiItemPtr(pitem->relto[dim]);
        pitem->visited |= 1<<dim;
        size = pitem->rect.v[wdim] + pitem->margins[dim] + pitem->margins[wdim];
        *need_size = (*need_size) + size;
        *hard_size = (*hard_size) + (pitem->size.v[dim]?size:0);
        it++;
        assert(it<1000000); // infinite loop
    }
    // traverse along right neighbors
    pitem = pkid;
    it = 0;
    while ((pitem->layout_flags>>dim) & UI_RIGHT) {
        if (pitem->relto[wdim] < 0) break;
        pitem = uiItemPtr(pitem->relto[wdim]);
        pitem->visited |= 1<<dim;
        size = pitem->rect.v[wdim] + pitem->margins[dim] + pitem->margins[wdim];
        *need_size = (*need_size) + size;
        *hard_size = (*hard_size) + (pitem->size.v[dim]?size:0);
        it++;
        assert(it<1000000); // infinite loop
    }
}

UI_INLINE void uiComputeSizeDim(UIitem *pitem, int dim) {
    int wdim = dim+2;
    int need_size = 0;
    int hard_size = 0;
    int kid = pitem->firstkid;
    while (kid >= 0) {
        UIitem *pkid = uiItemPtr(kid);
        if (!(pkid->visited & (1<<dim))) {
            int ns,hs;
            uiComputeChainSize(pkid, &ns, &hs, dim);
            need_size = ui_max(need_size, ns);
            hard_size = ui_max(hard_size, hs);
        }
        kid = uiNextSibling(kid);
    }
    pitem->computed_size.v[dim] = hard_size;
    
    if (pitem->size.v[dim]) {
        pitem->rect.v[wdim] = pitem->size.v[dim];
    } else {
        pitem->rect.v[wdim] = need_size;
    }
}

static void uiComputeBestSize(int item, int dim) {
    UIitem *pitem = uiItemPtr(item);
    pitem->visited = 0;
    // children expand the size
    int kid = uiFirstChild(item);
    while (kid >= 0) {
        uiComputeBestSize(kid, dim);
        kid = uiNextSibling(kid);
    }
    
    uiComputeSizeDim(pitem, dim);
}

static void uiLayoutChildItem(UIitem *pparent, UIitem *pitem, 
    int *dyncount, int *consumed_space, int dim) {
    if (pitem->visited & (4<<dim)) return;
    pitem->visited |= (4<<dim);
    
    int wdim = dim+2;
    
    int x = 0;
    int s = pparent->rect.v[wdim];
    
    int flags = pitem->layout_flags>>dim;
    int hasl = (flags & UI_LEFT) && (pitem->relto[dim] >= 0);
    int hasr = (flags & UI_RIGHT) && (pitem->relto[wdim] >= 0);
    
    if ((flags & UI_HFILL) != UI_HFILL) {
        *consumed_space = (*consumed_space)
            + pitem->rect.v[wdim]
            + pitem->margins[wdim]
            + pitem->margins[dim];
    } else if (!pitem->size.v[dim]) {
        *dyncount = (*dyncount)+1;
    }
    
    if (hasl) {
        UIitem *pl = uiItemPtr(pitem->relto[dim]);
        uiLayoutChildItem(pparent, pl, dyncount, consumed_space, dim);
        x = pl->rect.v[dim]+pl->rect.v[wdim]+pl->margins[wdim];
        s -= x;
    }
    if (hasr) {
        UIitem *pl = uiItemPtr(pitem->relto[wdim]);
        uiLayoutChildItem(pparent, pl, dyncount, consumed_space, dim);
        s = pl->rect.v[dim]-pl->margins[dim]-x;
    }

    switch(flags & UI_HFILL) {
    default:
    case UI_HCENTER: {
        pitem->rect.v[dim] = x+(s-pitem->rect.v[wdim])/2+pitem->margins[dim];
    } break;
    case UI_LEFT: {
        pitem->rect.v[dim] = x+pitem->margins[dim];
    } break;
    case UI_RIGHT: {
        pitem->rect.v[dim] = x+s-pitem->rect.v[wdim]-pitem->margins[wdim];
    } break;
    case UI_HFILL: {
        if (pitem->size.v[dim]) { // hard maximum size; can't stretch
            if (!hasl)
                pitem->rect.v[dim] = x+pitem->margins[dim];
            else
                pitem->rect.v[dim] = x+s-pitem->rect.v[wdim]-pitem->margins[wdim];
        } else {
            if (1) { //!pitem->rect.v[wdim]) {
                //int width = (pparent->rect.v[wdim] - pparent->computed_size.v[dim]);
                int width = (pparent->rect.v[wdim] - (*consumed_space));
                int space = width / (*dyncount);
                //int rest = width - space*(*dyncount);
                if (!hasl) {
                    pitem->rect.v[dim] = x+pitem->margins[dim];
                    pitem->rect.v[wdim] = s-pitem->margins[dim]-pitem->margins[wdim];
                } else {
                    pitem->rect.v[wdim] = space-pitem->margins[dim]-pitem->margins[wdim];
                    pitem->rect.v[dim] = x+s-pitem->rect.v[wdim]-pitem->margins[wdim];
                }
            } else {
                pitem->rect.v[dim] = x+pitem->margins[dim];
                pitem->rect.v[wdim] = s-pitem->margins[dim]-pitem->margins[wdim];
            }
        }
    } break;
    }    
}

UI_INLINE void uiLayoutItemDim(UIitem *pitem, int dim) {
    int wdim = dim+2;
    int kid = pitem->firstkid;
    int consumed_space = 0;
    int dyncount = 0;
    while (kid >= 0) {
        UIitem *pkid = uiItemPtr(kid);
        uiLayoutChildItem(pitem, pkid, &dyncount, &consumed_space, dim);
        kid = uiNextSibling(kid);
    }
}

static void uiLayoutItem(int item, int dim) {
    UIitem *pitem = uiItemPtr(item);
    
    uiLayoutItemDim(pitem, dim);
    
    int kid = uiFirstChild(item);
    while (kid >= 0) {
        uiLayoutItem(kid, dim);
        kid = uiNextSibling(kid);
    }
}

UIrect uiGetRect(int item) {
    return uiItemPtr(item)->rect;
}

UIrect uiGetActiveRect() {
    assert(ui_context);
    return ui_context->active_rect;
}

int uiFirstChild(int item) {
    return uiItemPtr(item)->firstkid;
}

int uiLastChild(int item) {
    return uiItemPtr(item)->lastkid;
}

int uiNextSibling(int item) {
    return uiItemPtr(item)->nextitem;
}

int uiPrevSibling(int item) {
    return uiItemPtr(item)->previtem;
}

int uiParent(int item) {
    return uiItemPtr(item)->parent;
}

const void *uiGetData(int item) {
    UIitem *pitem = uiItemPtr(item);
    if (pitem->data < 0) return NULL;
    return ui_context->data + pitem->data;
}

void *uiAllocData(int item, int size) {
    assert((size > 0) && (size < UI_MAX_DATASIZE));
    UIitem *pitem = uiItemPtr(item);
    assert(pitem->data < 0);
    assert((ui_context->datasize+size) <= UI_MAX_BUFFERSIZE);
    pitem->data = ui_context->datasize;
    ui_context->datasize += size;
    return ui_context->data + pitem->data;
}

void uiSetHandle(int item, UIhandle handle) {
    uiItemPtr(item)->handle = handle;
    if (handle) {
        uiHashInsertHandle(handle, item);
    }
}

void uiSetSelfHandle(int item) {
    UIitem *pitem = uiItemPtr(item);
    pitem->handle = (UIhandle)pitem;
    uiHashInsertHandle((UIhandle)pitem, item);
}

UIhandle uiGetHandle(int item) {
    return uiItemPtr(item)->handle;
}

void uiSetHandler(int item, UIhandler handler, int flags) {
    UIitem *pitem = uiItemPtr(item);
    pitem->handler = handler;
    pitem->event_flags = flags;
}

UIhandler uiGetHandler(int item) {
    return uiItemPtr(item)->handler;
}

int uiGetHandlerFlags(int item) {
    return uiItemPtr(item)->event_flags;
}

int uiGetChildId(int item) {
    return uiItemPtr(item)->kidid;
}

int uiGetChildCount(int item) {
    return uiItemPtr(item)->numkids;
}

UIrect uiGetAbsoluteRect(int item) {
    UIrect rect = uiGetRect(item);
    item = uiParent(item);
    while (item >= 0) {
        rect.x += uiItemPtr(item)->rect.x;
        rect.y += uiItemPtr(item)->rect.y;
        item = uiParent(item);
    }    
    return rect;
}

int uiContains(int item, int x, int y) {
    UIrect rect = uiGetAbsoluteRect(item);
    x -= rect.x;
    y -= rect.y;
    if ((x>=0)
     && (y>=0)
     && (x<rect.w)
     && (y<rect.h)) return 1;
    return 0;
}

int uiFindItem(int item, int x, int y, int ox, int oy) {
    UIitem *pitem = uiItemPtr(item);
    if (pitem->frozen) return -1;
    UIrect rect = pitem->rect;
    x -= rect.x;
    y -= rect.y;
    ox += rect.x;
    oy += rect.y;
    if ((x>=0)
     && (y>=0)
     && (x<rect.w)
     && (y<rect.h)) {
        int kid = uiLastChild(item);
        while (kid >= 0) {
            int best_hit = uiFindItem(kid,x,y,ox,oy);
            if (best_hit >= 0) return best_hit;
            kid = uiPrevSibling(kid);
        }
        // click-through if the item has no handler for input events
        if (pitem->event_flags & UI_ANY_INPUT) {
            rect.x = ox;
            rect.y = oy;
            ui_context->hot_rect = rect;
            return item;
        }
    }
    return -1;
}

void uiLayout() {
    assert(ui_context);
    if (!ui_context->count) return;

    // compute widths
    uiComputeBestSize(0,0);
    // position root element rect
    uiItemPtr(0)->rect.x = uiItemPtr(0)->margins[0];
    uiLayoutItem(0,0);
    
    // compute heights
    uiComputeBestSize(0,1);
    // position root element rect
    uiItemPtr(0)->rect.y = uiItemPtr(0)->margins[1];    
    uiLayoutItem(0,1);

    // drawing routines may require this to be set already
    int hot = uiFindItem(0, 
        ui_context->cursor.x, ui_context->cursor.y, 0, 0);
    ui_context->hot_item = hot;
}

void uiProcess() {
    assert(ui_context);
    if (!ui_context->count) {
        uiClearInputEvents();
        return;
    }
    
    int hot_item = uiGetItem(ui_context->hot_handle);
    int active_item = uiGetItem(ui_context->active_handle);
    int focus_item = uiGetItem(ui_context->focus_handle);

    // send all keyboard events
    if (focus_item >= 0) {
        for (int i = 0; i < ui_context->eventcount; ++i) {
            ui_context->active_key = ui_context->events[i].key;
            ui_context->active_modifier = ui_context->events[i].mod;
            uiNotifyItem(focus_item, 
                ui_context->events[i].event);
        }
    } else {
        ui_context->focus_handle = 0;
    }
    uiClearInputEvents();

    int hot = ui_context->hot_item;

    switch(ui_context->state) {
    default:
    case UI_STATE_IDLE: {
        ui_context->start_cursor = ui_context->cursor;
        if (uiGetButton(0)) {
            hot_item = -1;
            active_item = hot;
            ui_context->active_rect = ui_context->hot_rect;
            
            if (active_item != focus_item) {
                focus_item = -1;
                ui_context->focus_handle = 0;
            }
            
            if (active_item >= 0) {
                uiNotifyItem(active_item, UI_BUTTON0_DOWN);
            }            
            ui_context->state = UI_STATE_CAPTURE;            
        } else if (uiGetButton(2) && !uiGetLastButton(2) && (hot >= 0)) {
            hot_item = -1;
            ui_context->active_rect = ui_context->hot_rect;
            uiNotifyItem(hot, UI_BUTTON2_DOWN);
        } else {
            hot_item = hot;
        }
    } break;
    case UI_STATE_CAPTURE: {
        if (!uiGetButton(0)) {
            if (active_item >= 0) {
                uiNotifyItem(active_item, UI_BUTTON0_UP);
                if (active_item == hot) {
                    uiNotifyItem(active_item, UI_BUTTON0_HOT_UP);
                }
            }
            active_item = -1;
            ui_context->state = UI_STATE_IDLE;
        } else {
            if (active_item >= 0) {
                uiNotifyItem(active_item, UI_BUTTON0_CAPTURE);
            }            
            if (hot == active_item)
                hot_item = hot;
            else
                hot_item = -1;
        }
    } break;
    }
    
    ui_context->last_cursor = ui_context->cursor;
    ui_context->hot_handle = (hot_item>=0)?
        uiGetHandle(hot_item):0;
    ui_context->active_handle = (active_item>=0)?
        uiGetHandle(active_item):0;
}

static int uiIsActive(int item) {
    assert(ui_context);
    return (ui_context->active_handle)&&(uiGetHandle(item) == ui_context->active_handle);
}

static int uiIsHot(int item) {
    assert(ui_context);
    return (ui_context->hot_handle)&&(uiGetHandle(item) == ui_context->hot_handle);
}

static int uiIsFocused(int item) {
    assert(ui_context);
    return (ui_context->focus_handle)&&(uiGetHandle(item) == ui_context->focus_handle);
}

UIitemState uiGetState(int item) {
    UIitem *pitem = uiItemPtr(item);
    if (pitem->frozen) return UI_FROZEN;
    if (uiIsFocused(item)) {
        if (pitem->event_flags & (UI_KEY_DOWN|UI_CHAR|UI_KEY_UP)) return UI_ACTIVE;
    }
    if (uiIsActive(item)) {
        if (pitem->event_flags & (UI_BUTTON0_CAPTURE|UI_BUTTON0_UP)) return UI_ACTIVE;
        if ((pitem->event_flags & UI_BUTTON0_HOT_UP)
            && uiIsHot(item)) return UI_ACTIVE;
        return UI_COLD;
    } else if (uiIsHot(item)) {
        return UI_HOT;
    }
    return UI_COLD;
}

#endif // OUI_IMPLEMENTATION
