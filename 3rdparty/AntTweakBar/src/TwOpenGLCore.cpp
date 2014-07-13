//  ---------------------------------------------------------------------------
//
//  @file       TwOpenGLCore.cpp
//  @author     Philippe Decaudin
//  @license    This file is part of the AntTweakBar library.
//              For conditions of distribution and use, see License.txt
//
//  ---------------------------------------------------------------------------

/*
#pragma warning GL3             //// used for development
#define GL3_PROTOTYPES 1        ////
#include <GL3/gl3.h>            ////
#define ANT_OGL_HEADER_INCLUDED ////
*/

#if defined ANT_OSX
#   include <OpenGL/gl3.h>
#   define ANT_OGL_HEADER_INCLUDED
#endif
#include "TwPrecomp.h"
#include "TwOpenGLCore.h"
#include "TwMgr.h"

using namespace std;

extern const char *g_ErrCantLoadOGL;
extern const char *g_ErrCantUnloadOGL;

//  ---------------------------------------------------------------------------

#ifdef _DEBUG
    static void CheckGLCoreError(const char *file, int line, const char *func)
    {
        int err=0;
        char msg[256];
        while( (err=glGetError())!=0 )
        {
            sprintf(msg, "%s(%d) : [%s] GL_CORE_ERROR=0x%x\n", file, line, func, err);
            #ifdef ANT_WINDOWS
                OutputDebugStringA(msg);
            #endif
            fprintf(stderr, msg);
        }
    }
#   ifdef __FUNCTION__
#       define CHECK_GL_ERROR CheckGLCoreError(__FILE__, __LINE__, __FUNCTION__)
#   else
#       define CHECK_GL_ERROR CheckGLCoreError(__FILE__, __LINE__, "")
#   endif
#else
#   define CHECK_GL_ERROR ((void)(0))
#endif

//  ---------------------------------------------------------------------------

static GLuint BindFont(const CTexFont *_Font)
{
    GLuint TexID = 0;
    glGenTextures(1, &TexID);
    glBindTexture(GL_TEXTURE_2D, TexID);
    glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);
    glPixelStorei(GL_UNPACK_LSB_FIRST, GL_FALSE);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, _Font->m_TexWidth, _Font->m_TexHeight, 0, GL_RED, GL_UNSIGNED_BYTE, _Font->m_TexBytes);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    return TexID;
}

static void UnbindFont(GLuint _FontTexID)
{
    if( _FontTexID>0 )
        glDeleteTextures(1, &_FontTexID);
}

//  ---------------------------------------------------------------------------

static GLuint CompileShader(GLuint shader)
{
    glCompileShader(shader); CHECK_GL_ERROR;

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status); CHECK_GL_ERROR;
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength); CHECK_GL_ERROR;

        GLchar strInfoLog[256];
        glGetShaderInfoLog(shader, sizeof(strInfoLog), NULL, strInfoLog); CHECK_GL_ERROR;
#ifdef ANT_WINDOWS
        OutputDebugStringA("Compile failure: ");
        OutputDebugStringA(strInfoLog);
        OutputDebugStringA("\n");
#endif
        fprintf(stderr, "Compile failure: %s\n", strInfoLog);
        shader = 0;
    }

    return shader;
}

static GLuint LinkProgram(GLuint program)
{
    glLinkProgram(program); CHECK_GL_ERROR;

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status); CHECK_GL_ERROR;
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength); CHECK_GL_ERROR;

        GLchar strInfoLog[256];
        glGetProgramInfoLog(program, sizeof(strInfoLog), NULL, strInfoLog); CHECK_GL_ERROR;
#ifdef ANT_WINDOWS
        OutputDebugStringA("Linker failure: ");
        OutputDebugStringA(strInfoLog);
        OutputDebugStringA("\n");
#endif
        fprintf(stderr, "Linker failure: %s\n", strInfoLog);
        program = 0;
    }

    return program;
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::ResizeTriBuffers(size_t _NewSize)
{
    m_TriBufferSize = _NewSize;

    glBindVertexArray(m_TriVArray);

    glBindBuffer(GL_ARRAY_BUFFER, m_TriVertices);
    glBufferData(GL_ARRAY_BUFFER, m_TriBufferSize*sizeof(Vec2), 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, m_TriUVs);
    glBufferData(GL_ARRAY_BUFFER, m_TriBufferSize*sizeof(Vec2), 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, m_TriColors);
    glBufferData(GL_ARRAY_BUFFER, m_TriBufferSize*sizeof(color32), 0, GL_DYNAMIC_DRAW);

    CHECK_GL_ERROR;
}

//  ---------------------------------------------------------------------------

int CTwGraphOpenGLCore::Init()
{
    m_Drawing = false;
    m_FontTexID = 0;
    m_FontTex = NULL;

    if( glewInit() != GLEW_OK )
    {
        g_TwMgr->SetLastError(g_ErrCantLoadOGL);
        return 0;
    }

    // Create line/rect shaders
    const GLchar *lineRectVS[] = {
        "#version 150 core\n"
        "in vec3 vertex;"
        "in vec4 color;"
        "out vec4 fcolor;"
        "void main() { gl_Position = vec4(vertex, 1); fcolor = color; }"
    };
    m_LineRectVS = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_LineRectVS, 1, lineRectVS, NULL);
    CompileShader(m_LineRectVS);

    const GLchar *lineRectFS[] = {
        "#version 150 core\n"
        "precision highp float;"
        "in vec4 fcolor;"
        "out vec4 outColor;"
        "void main() { outColor = fcolor; }"
    };
    m_LineRectFS = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(m_LineRectFS, 1, lineRectFS, NULL);
    CompileShader(m_LineRectFS);

    m_LineRectProgram = glCreateProgram();
    glAttachShader(m_LineRectProgram, m_LineRectVS);
    glAttachShader(m_LineRectProgram, m_LineRectFS);
    glBindAttribLocation(m_LineRectProgram, 0, "vertex");
    glBindAttribLocation(m_LineRectProgram, 1, "color");
    LinkProgram(m_LineRectProgram);

    // Create line/rect vertex buffer
    const GLfloat lineRectInitVertices[] = { 0,0,0, 0,0,0, 0,0,0, 0,0,0 };
    const color32 lineRectInitColors[] = { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff };
    glGenVertexArrays(1, &m_LineRectVArray);
    glBindVertexArray(m_LineRectVArray);
    glGenBuffers(1, &m_LineRectVertices);
    glBindBuffer(GL_ARRAY_BUFFER, m_LineRectVertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lineRectInitVertices), lineRectInitVertices, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &m_LineRectColors);
    glBindBuffer(GL_ARRAY_BUFFER, m_LineRectColors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lineRectInitColors), lineRectInitColors, GL_DYNAMIC_DRAW);

    // Create triangles shaders
    const GLchar *triVS[] = {
        "#version 150 core\n"
        "uniform vec2 offset;"
        "uniform vec2 wndSize;"
        "in vec2 vertex;"
        "in vec4 color;"
        "out vec4 fcolor;"
        "void main() { gl_Position = vec4(2.0*(vertex.x+offset.x-0.5)/wndSize.x - 1.0, 1.0 - 2.0*(vertex.y+offset.y-0.5)/wndSize.y, 0, 1); fcolor = color; }"
    };
    m_TriVS = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_TriVS, 1, triVS, NULL);
    CompileShader(m_TriVS);

    const GLchar *triUniVS[] = {
        "#version 150 core\n"
        "uniform vec2 offset;"
        "uniform vec2 wndSize;"
        "uniform vec4 color;"
        "in vec2 vertex;"
        "out vec4 fcolor;"
        "void main() { gl_Position = vec4(2.0*(vertex.x+offset.x-0.5)/wndSize.x - 1.0, 1.0 - 2.0*(vertex.y+offset.y-0.5)/wndSize.y, 0, 1); fcolor = color; }"
    };
    m_TriUniVS = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_TriUniVS, 1, triUniVS, NULL);
    CompileShader(m_TriUniVS);

    m_TriFS = m_TriUniFS = m_LineRectFS;

    m_TriProgram = glCreateProgram();
    glAttachShader(m_TriProgram, m_TriVS);
    glAttachShader(m_TriProgram, m_TriFS);
    glBindAttribLocation(m_TriProgram, 0, "vertex");
    glBindAttribLocation(m_TriProgram, 1, "color");
    LinkProgram(m_TriProgram);
    m_TriLocationOffset = glGetUniformLocation(m_TriProgram, "offset");
    m_TriLocationWndSize = glGetUniformLocation(m_TriProgram, "wndSize");

    m_TriUniProgram = glCreateProgram();
    glAttachShader(m_TriUniProgram, m_TriUniVS);
    glAttachShader(m_TriUniProgram, m_TriUniFS);
    glBindAttribLocation(m_TriUniProgram, 0, "vertex");
    glBindAttribLocation(m_TriUniProgram, 1, "color");
    LinkProgram(m_TriUniProgram);
    m_TriUniLocationOffset = glGetUniformLocation(m_TriUniProgram, "offset");
    m_TriUniLocationWndSize = glGetUniformLocation(m_TriUniProgram, "wndSize");
    m_TriUniLocationColor = glGetUniformLocation(m_TriUniProgram, "color");

    const GLchar *triTexFS[] = {
        "#version 150 core\n"
        "precision highp float;"
        "uniform sampler2D tex;"
        "in vec2 fuv;"
        "in vec4 fcolor;"
        "out vec4 outColor;"
// texture2D is deprecated and replaced by texture with GLSL 3.30 but it seems 
// that on Mac Lion backward compatibility is not ensured.
#if defined(ANT_OSX) && (MAC_OS_X_VERSION_MAX_ALLOWED >= 1070)
        "void main() { outColor.rgb = fcolor.bgr; outColor.a = fcolor.a * texture(tex, fuv).r; }"
#else
        "void main() { outColor.rgb = fcolor.bgr; outColor.a = fcolor.a * texture2D(tex, fuv).r; }"
#endif
    };
    m_TriTexFS = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(m_TriTexFS, 1, triTexFS, NULL);
    CompileShader(m_TriTexFS);

    const GLchar *triTexVS[] = {
        "#version 150 core\n"
        "uniform vec2 offset;"
        "uniform vec2 wndSize;"
        "in vec2 vertex;"
        "in vec2 uv;"
        "in vec4 color;"
        "out vec2 fuv;"
        "out vec4 fcolor;"
        "void main() { gl_Position = vec4(2.0*(vertex.x+offset.x-0.5)/wndSize.x - 1.0, 1.0 - 2.0*(vertex.y+offset.y-0.5)/wndSize.y, 0, 1); fuv = uv; fcolor = color; }"
    };
    m_TriTexVS = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_TriTexVS, 1, triTexVS, NULL);
    CompileShader(m_TriTexVS);

    const GLchar *triTexUniVS[] = {
        "#version 150 core\n"
        "uniform vec2 offset;"
        "uniform vec2 wndSize;"
        "uniform vec4 color;"
        "in vec2 vertex;"
        "in vec2 uv;"
        "out vec4 fcolor;"
        "out vec2 fuv;"
        "void main() { gl_Position = vec4(2.0*(vertex.x+offset.x-0.5)/wndSize.x - 1.0, 1.0 - 2.0*(vertex.y+offset.y-0.5)/wndSize.y, 0, 1); fuv = uv; fcolor = color; }"
    };
    m_TriTexUniVS = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_TriTexUniVS, 1, triTexUniVS, NULL);
    CompileShader(m_TriTexUniVS);

    m_TriTexUniFS = m_TriTexFS;

    m_TriTexProgram = glCreateProgram();
    glAttachShader(m_TriTexProgram, m_TriTexVS);
    glAttachShader(m_TriTexProgram, m_TriTexFS);
    glBindAttribLocation(m_TriTexProgram, 0, "vertex");
    glBindAttribLocation(m_TriTexProgram, 1, "uv");
    glBindAttribLocation(m_TriTexProgram, 2, "color");
    LinkProgram(m_TriTexProgram);
    m_TriTexLocationOffset = glGetUniformLocation(m_TriTexProgram, "offset");
    m_TriTexLocationWndSize = glGetUniformLocation(m_TriTexProgram, "wndSize");
    m_TriTexLocationTexture = glGetUniformLocation(m_TriTexProgram, "tex");

    m_TriTexUniProgram = glCreateProgram();
    glAttachShader(m_TriTexUniProgram, m_TriTexUniVS);
    glAttachShader(m_TriTexUniProgram, m_TriTexUniFS);
    glBindAttribLocation(m_TriTexUniProgram, 0, "vertex");
    glBindAttribLocation(m_TriTexUniProgram, 1, "uv");
    glBindAttribLocation(m_TriTexUniProgram, 2, "color");
    LinkProgram(m_TriTexUniProgram);
    m_TriTexUniLocationOffset = glGetUniformLocation(m_TriTexUniProgram, "offset");
    m_TriTexUniLocationWndSize = glGetUniformLocation(m_TriTexUniProgram, "wndSize");
    m_TriTexUniLocationColor = glGetUniformLocation(m_TriTexUniProgram, "color");
    m_TriTexUniLocationTexture = glGetUniformLocation(m_TriTexUniProgram, "tex");

    // Create tri vertex buffer
    glGenVertexArrays(1, &m_TriVArray);
    glGenBuffers(1, &m_TriVertices);
    glGenBuffers(1, &m_TriUVs);
    glGenBuffers(1, &m_TriColors);
    ResizeTriBuffers(16384); // set initial size

    CHECK_GL_ERROR;
    return 1;
}

//  ---------------------------------------------------------------------------

int CTwGraphOpenGLCore::Shut()
{
    assert(m_Drawing==false);

    UnbindFont(m_FontTexID);

    CHECK_GL_ERROR;

    glDeleteProgram(m_LineRectProgram); m_LineRectProgram = 0;
    glDeleteShader(m_LineRectVS); m_LineRectVS = 0;
    glDeleteShader(m_LineRectFS); m_LineRectFS = 0;

    glDeleteProgram(m_TriProgram); m_TriProgram = 0;
    glDeleteShader(m_TriVS); m_TriVS = 0;

    glDeleteProgram(m_TriUniProgram); m_TriUniProgram = 0;
    glDeleteShader(m_TriUniVS); m_TriUniVS = 0;

    glDeleteProgram(m_TriTexProgram); m_TriTexProgram = 0;
    glDeleteShader(m_TriTexVS); m_TriTexVS = 0;
    glDeleteShader(m_TriTexFS); m_TriTexFS = 0;

    glDeleteProgram(m_TriTexUniProgram); m_TriTexUniProgram = 0;
    glDeleteShader(m_TriTexUniVS); m_TriTexUniVS = 0;

    glDeleteBuffers(1, &m_LineRectVertices); m_LineRectVertices = 0;
    glDeleteBuffers(1, &m_LineRectColors); m_LineRectColors = 0;
    glDeleteVertexArrays(1, &m_LineRectVArray); m_LineRectVArray = 0;

    glDeleteBuffers(1, &m_TriVertices); m_TriVertices = 0;
    glDeleteBuffers(1, &m_TriColors); m_TriColors = 0;
    glDeleteBuffers(1, &m_TriUVs); m_TriUVs = 0;
    glDeleteVertexArrays(1, &m_TriVArray); m_TriVArray = 0;

    CHECK_GL_ERROR;

    int Res = 1;
    //if( glewInit()==0 )
    //{
    //    g_TwMgr->SetLastError(g_ErrCantUnloadOGL);
    //    Res = 0;
    //}

    return Res;
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::BeginDraw(int _WndWidth, int _WndHeight)
{
    CHECK_GL_ERROR;
    assert(m_Drawing==false && _WndWidth>0 && _WndHeight>0);
    m_Drawing = true;
    m_WndWidth = _WndWidth;
    m_WndHeight = _WndHeight;
    m_OffsetX = 0;
    m_OffsetY = 0;

    glGetIntegerv(GL_VIEWPORT, m_PrevViewport); CHECK_GL_ERROR;
    if( _WndWidth>0 && _WndHeight>0 )
    {
        GLint Vp[4];
        Vp[0] = 0;
        Vp[1] = 0;
        Vp[2] = _WndWidth-1;
        Vp[3] = _WndHeight-1;
        glViewport(Vp[0], Vp[1], Vp[2], Vp[3]);
    }

    m_PrevVArray = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, (GLint*)&m_PrevVArray); CHECK_GL_ERROR;
    glBindVertexArray(0); CHECK_GL_ERROR;

    m_PrevLineWidth = 1;
    glGetFloatv(GL_LINE_WIDTH, &m_PrevLineWidth); CHECK_GL_ERROR;
    glLineWidth(1); CHECK_GL_ERROR;

    m_PrevLineSmooth = glIsEnabled(GL_LINE_SMOOTH);
    glDisable(GL_LINE_SMOOTH); CHECK_GL_ERROR;

    m_PrevCullFace = glIsEnabled(GL_CULL_FACE);
    glDisable(GL_CULL_FACE); CHECK_GL_ERROR;
    
    m_PrevDepthTest = glIsEnabled(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST); CHECK_GL_ERROR;

    m_PrevBlend = glIsEnabled(GL_BLEND);
    glEnable(GL_BLEND); CHECK_GL_ERROR;

    m_PrevScissorTest = glIsEnabled(GL_SCISSOR_TEST);
    glDisable(GL_SCISSOR_TEST); CHECK_GL_ERROR;

    glGetIntegerv(GL_SCISSOR_BOX, m_PrevScissorBox); CHECK_GL_ERROR;

    glGetIntegerv(GL_BLEND_SRC, &m_PrevSrcBlend); CHECK_GL_ERROR;
    glGetIntegerv(GL_BLEND_DST, &m_PrevDstBlend); CHECK_GL_ERROR;
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); CHECK_GL_ERROR;

    m_PrevTexture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &m_PrevTexture); CHECK_GL_ERROR;
    glBindTexture(GL_TEXTURE_2D, 0); CHECK_GL_ERROR;

    m_PrevProgramObject = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*)&m_PrevProgramObject); CHECK_GL_ERROR;
    glBindVertexArray(0); CHECK_GL_ERROR;
    glUseProgram(0); CHECK_GL_ERROR;  

    m_PrevActiveTexture = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, (GLint*)&m_PrevActiveTexture); CHECK_GL_ERROR;
    glActiveTexture(GL_TEXTURE0);

    CHECK_GL_ERROR;
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::EndDraw()
{
    assert(m_Drawing==true);
    m_Drawing = false;

    glLineWidth(m_PrevLineWidth); CHECK_GL_ERROR;

    if( m_PrevLineSmooth )
    {
      glEnable(GL_LINE_SMOOTH); CHECK_GL_ERROR;
    }
    else
    {
      glDisable(GL_LINE_SMOOTH); CHECK_GL_ERROR;      
    }

    if( m_PrevCullFace )
    {
      glEnable(GL_CULL_FACE); CHECK_GL_ERROR;
    }
    else
    {
      glDisable(GL_CULL_FACE); CHECK_GL_ERROR;      
    }

    if( m_PrevDepthTest )
    {
      glEnable(GL_DEPTH_TEST); CHECK_GL_ERROR;
    }
    else
    {
      glDisable(GL_DEPTH_TEST); CHECK_GL_ERROR;      
    }

    if( m_PrevBlend )
    {
      glEnable(GL_BLEND); CHECK_GL_ERROR;
    }
    else
    {
      glDisable(GL_BLEND); CHECK_GL_ERROR;      
    }

    if( m_PrevScissorTest )
    {
      glEnable(GL_SCISSOR_TEST); CHECK_GL_ERROR;
    }
    else
    {
      glDisable(GL_SCISSOR_TEST); CHECK_GL_ERROR;      
    }

    glScissor(m_PrevScissorBox[0], m_PrevScissorBox[1], m_PrevScissorBox[2], m_PrevScissorBox[3]); CHECK_GL_ERROR;

    glBlendFunc(m_PrevSrcBlend, m_PrevDstBlend); CHECK_GL_ERROR;

    glBindTexture(GL_TEXTURE_2D, m_PrevTexture); CHECK_GL_ERROR;

    glUseProgram(m_PrevProgramObject); CHECK_GL_ERROR;

    glBindBuffer(GL_ARRAY_BUFFER, 0); CHECK_GL_ERROR;
    
    glBindVertexArray(m_PrevVArray); CHECK_GL_ERROR;

    glViewport(m_PrevViewport[0], m_PrevViewport[1], m_PrevViewport[2], m_PrevViewport[3]); CHECK_GL_ERROR;

    CHECK_GL_ERROR;
}

//  ---------------------------------------------------------------------------

bool CTwGraphOpenGLCore::IsDrawing()
{
    return m_Drawing;
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::Restore()
{
    UnbindFont(m_FontTexID);
    m_FontTexID = 0;
    m_FontTex = NULL;
}

//  ---------------------------------------------------------------------------

static inline float ToNormScreenX(float x, int wndWidth)
{
    return 2.0f*((float)x-0.5f)/wndWidth - 1.0f;
}

static inline float ToNormScreenY(float y, int wndHeight)
{
    return 1.0f - 2.0f*((float)y-0.5f)/wndHeight;
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::DrawLine(int _X0, int _Y0, int _X1, int _Y1, color32 _Color0, color32 _Color1, bool _AntiAliased)
{
    CHECK_GL_ERROR;
    assert(m_Drawing==true);

    //const GLfloat dx = +0.0f;
    const GLfloat dx = 0;
    //GLfloat dy = -0.2f;
    const GLfloat dy = -0.5f;
    if( _AntiAliased )
        glEnable(GL_LINE_SMOOTH);
    else
        glDisable(GL_LINE_SMOOTH);

    glBindVertexArray(m_LineRectVArray);

    GLfloat x0 = ToNormScreenX(_X0+dx + m_OffsetX, m_WndWidth);
    GLfloat y0 = ToNormScreenY(_Y0+dy + m_OffsetY, m_WndHeight);
    GLfloat x1 = ToNormScreenX(_X1+dx + m_OffsetX, m_WndWidth);
    GLfloat y1 = ToNormScreenY(_Y1+dy + m_OffsetY, m_WndHeight);
    GLfloat vertices[] = { x0,y0,0,  x1,y1,0 };
    glBindBuffer(GL_ARRAY_BUFFER, m_LineRectVertices);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_TRUE, 0, NULL);
    glEnableVertexAttribArray(0);

    color32 colors[] = { _Color0, _Color1 };
    glBindBuffer(GL_ARRAY_BUFFER, m_LineRectColors);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(colors), colors);
    glVertexAttribPointer(1, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
    glEnableVertexAttribArray(1);

    glUseProgram(m_LineRectProgram);
    glDrawArrays(GL_LINES, 0, 2);

    if( _AntiAliased )
        glDisable(GL_LINE_SMOOTH);

    CHECK_GL_ERROR;
}
  
//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::DrawRect(int _X0, int _Y0, int _X1, int _Y1, color32 _Color00, color32 _Color10, color32 _Color01, color32 _Color11)
{
    CHECK_GL_ERROR;
    assert(m_Drawing==true);

    // border adjustment
    if(_X0<_X1)
        ++_X1;
    else if(_X0>_X1)
        ++_X0;
    if(_Y0<_Y1)
        --_Y0;
    else if(_Y0>_Y1)
        --_Y1;

    glBindVertexArray(m_LineRectVArray);

    GLfloat x0 = ToNormScreenX((float)_X0 + m_OffsetX, m_WndWidth);
    GLfloat y0 = ToNormScreenY((float)_Y0 + m_OffsetY, m_WndHeight);
    GLfloat x1 = ToNormScreenX((float)_X1 + m_OffsetX, m_WndWidth);
    GLfloat y1 = ToNormScreenY((float)_Y1 + m_OffsetY, m_WndHeight);
    GLfloat vertices[] = { x0,y0,0, x1,y0,0, x0,y1,0, x1,y1,0 };
    glBindBuffer(GL_ARRAY_BUFFER, m_LineRectVertices);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_TRUE, 0, NULL);
    glEnableVertexAttribArray(0);

    GLuint colors[] = { _Color00, _Color10, _Color01, _Color11 };
    glBindBuffer(GL_ARRAY_BUFFER, m_LineRectColors);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(colors), colors);
    glVertexAttribPointer(1, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
    glEnableVertexAttribArray(1);

    glUseProgram(m_LineRectProgram);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    CHECK_GL_ERROR;
}

//  ---------------------------------------------------------------------------

void *CTwGraphOpenGLCore::NewTextObj()
{
    return new CTextObj;
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::DeleteTextObj(void *_TextObj)
{
    assert(_TextObj!=NULL);
    delete static_cast<CTextObj *>(_TextObj);
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::BuildText(void *_TextObj, const std::string *_TextLines, color32 *_LineColors, color32 *_LineBgColors, int _NbLines, const CTexFont *_Font, int _Sep, int _BgWidth)
{
    assert(m_Drawing==true);
    assert(_TextObj!=NULL);
    assert(_Font!=NULL);

    if( _Font != m_FontTex )
    {
        UnbindFont(m_FontTexID);
        m_FontTexID = BindFont(_Font);
        m_FontTex = _Font;
    }
    CTextObj *TextObj = static_cast<CTextObj *>(_TextObj);
    TextObj->m_TextVerts.resize(0);
    TextObj->m_TextUVs.resize(0);
    TextObj->m_BgVerts.resize(0);
    TextObj->m_Colors.resize(0);
    TextObj->m_BgColors.resize(0);

    int x, x1, y, y1, i, Len;
    unsigned char ch;
    const unsigned char *Text;
    color32 LineColor = COLOR32_RED;
    for( int Line=0; Line<_NbLines; ++Line )
    {
        x = 0;
        y = Line * (_Font->m_CharHeight+_Sep);
        y1 = y+_Font->m_CharHeight;
        Len = (int)_TextLines[Line].length();
        Text = (const unsigned char *)(_TextLines[Line].c_str());
        if( _LineColors!=NULL )
            LineColor = (_LineColors[Line]&0xff00ff00) | GLubyte(_LineColors[Line]>>16) | (GLubyte(_LineColors[Line])<<16);

        for( i=0; i<Len; ++i )
        {
            ch = Text[i];
            x1 = x + _Font->m_CharWidth[ch];

            TextObj->m_TextVerts.push_back(Vec2(x , y ));
            TextObj->m_TextVerts.push_back(Vec2(x1, y ));
            TextObj->m_TextVerts.push_back(Vec2(x , y1));
            TextObj->m_TextVerts.push_back(Vec2(x1, y ));
            TextObj->m_TextVerts.push_back(Vec2(x1, y1));
            TextObj->m_TextVerts.push_back(Vec2(x , y1));

            TextObj->m_TextUVs.push_back(Vec2(_Font->m_CharU0[ch], _Font->m_CharV0[ch]));
            TextObj->m_TextUVs.push_back(Vec2(_Font->m_CharU1[ch], _Font->m_CharV0[ch]));
            TextObj->m_TextUVs.push_back(Vec2(_Font->m_CharU0[ch], _Font->m_CharV1[ch]));
            TextObj->m_TextUVs.push_back(Vec2(_Font->m_CharU1[ch], _Font->m_CharV0[ch]));
            TextObj->m_TextUVs.push_back(Vec2(_Font->m_CharU1[ch], _Font->m_CharV1[ch]));
            TextObj->m_TextUVs.push_back(Vec2(_Font->m_CharU0[ch], _Font->m_CharV1[ch]));

            if( _LineColors!=NULL )
            {
                TextObj->m_Colors.push_back(LineColor);
                TextObj->m_Colors.push_back(LineColor);
                TextObj->m_Colors.push_back(LineColor);
                TextObj->m_Colors.push_back(LineColor);
                TextObj->m_Colors.push_back(LineColor);
                TextObj->m_Colors.push_back(LineColor);
            }

            x = x1;
        }
        if( _BgWidth>0 )
        {
            TextObj->m_BgVerts.push_back(Vec2(-1        , y ));
            TextObj->m_BgVerts.push_back(Vec2(_BgWidth+1, y ));
            TextObj->m_BgVerts.push_back(Vec2(-1        , y1));
            TextObj->m_BgVerts.push_back(Vec2(_BgWidth+1, y ));
            TextObj->m_BgVerts.push_back(Vec2(_BgWidth+1, y1));
            TextObj->m_BgVerts.push_back(Vec2(-1        , y1));

            if( _LineBgColors!=NULL )
            {
                color32 LineBgColor = (_LineBgColors[Line]&0xff00ff00) | GLubyte(_LineBgColors[Line]>>16) | (GLubyte(_LineBgColors[Line])<<16);
                TextObj->m_BgColors.push_back(LineBgColor);
                TextObj->m_BgColors.push_back(LineBgColor);
                TextObj->m_BgColors.push_back(LineBgColor);
                TextObj->m_BgColors.push_back(LineBgColor);
                TextObj->m_BgColors.push_back(LineBgColor);
                TextObj->m_BgColors.push_back(LineBgColor);
            }
        }
    }
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::DrawText(void *_TextObj, int _X, int _Y, color32 _Color, color32 _BgColor)
{
    CHECK_GL_ERROR;
    assert(m_Drawing==true);
    assert(_TextObj!=NULL);
    CTextObj *TextObj = static_cast<CTextObj *>(_TextObj);

    if( TextObj->m_TextVerts.size()<4 && TextObj->m_BgVerts.size()<4 )
        return; // nothing to draw

    // draw character background triangles
    if( (_BgColor!=0 || TextObj->m_BgColors.size()==TextObj->m_BgVerts.size()) && TextObj->m_BgVerts.size()>=4 )
    {
        size_t numBgVerts = TextObj->m_BgVerts.size();
        if( numBgVerts > m_TriBufferSize )
            ResizeTriBuffers(numBgVerts + 2048);
  
        glBindVertexArray(m_TriVArray);

        glBindBuffer(GL_ARRAY_BUFFER, m_TriVertices);
        glBufferSubData(GL_ARRAY_BUFFER, 0, numBgVerts*sizeof(Vec2), &(TextObj->m_BgVerts[0]));
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_TRUE, 0, NULL);
        glEnableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);

        if( TextObj->m_BgColors.size()==TextObj->m_BgVerts.size() && _BgColor==0 )
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_TriColors);
            glBufferSubData(GL_ARRAY_BUFFER, 0, numBgVerts*sizeof(color32), &(TextObj->m_BgColors[0]));
            glVertexAttribPointer(1, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
            glEnableVertexAttribArray(1);

            glUseProgram(m_TriProgram);
            glUniform2f(m_TriLocationOffset, (float)_X, (float)_Y);
            glUniform2f(m_TriLocationWndSize, (float)m_WndWidth, (float)m_WndHeight);
        }
        else
        {
            glUseProgram(m_TriUniProgram);
            glUniform4f(m_TriUniLocationColor, GLfloat((_BgColor>>16)&0xff)/256.0f, GLfloat((_BgColor>>8)&0xff)/256.0f, GLfloat(_BgColor&0xff)/256.0f, GLfloat((_BgColor>>24)&0xff)/256.0f);
            glUniform2f(m_TriUniLocationOffset, (float)_X, (float)_Y);
            glUniform2f(m_TriUniLocationWndSize, (float)m_WndWidth, (float)m_WndHeight);
        }
        
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)TextObj->m_BgVerts.size());
    }

    // draw character triangles
    if( TextObj->m_TextVerts.size()>=4 )
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_FontTexID);
        size_t numTextVerts = TextObj->m_TextVerts.size();
        if( numTextVerts > m_TriBufferSize )
            ResizeTriBuffers(numTextVerts + 2048);
        
        glBindVertexArray(m_TriVArray);
        glDisableVertexAttribArray(2);

        glBindBuffer(GL_ARRAY_BUFFER, m_TriVertices);
        glBufferSubData(GL_ARRAY_BUFFER, 0, numTextVerts*sizeof(Vec2), &(TextObj->m_TextVerts[0]));
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_TRUE, 0, NULL);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, m_TriUVs);
        glBufferSubData(GL_ARRAY_BUFFER, 0, numTextVerts*sizeof(Vec2), &(TextObj->m_TextUVs[0]));
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(1);

        if( TextObj->m_Colors.size()==TextObj->m_TextVerts.size() && _Color==0 )
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_TriColors);
            glBufferSubData(GL_ARRAY_BUFFER, 0, numTextVerts*sizeof(color32), &(TextObj->m_Colors[0]));
            glVertexAttribPointer(2, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
            glEnableVertexAttribArray(2);

            glUseProgram(m_TriTexProgram);
            glUniform2f(m_TriTexLocationOffset, (float)_X, (float)_Y);
            glUniform2f(m_TriTexLocationWndSize, (float)m_WndWidth, (float)m_WndHeight);
            glUniform1i(m_TriTexLocationTexture, 0);
        }
        else
        {
            glUseProgram(m_TriTexUniProgram);
            glUniform4f(m_TriTexUniLocationColor, GLfloat((_Color>>16)&0xff)/256.0f, GLfloat((_Color>>8)&0xff)/256.0f, GLfloat(_Color&0xff)/256.0f, GLfloat((_Color>>24)&0xff)/256.0f);
            glUniform2f(m_TriTexUniLocationOffset, (float)_X, (float)_Y);
            glUniform2f(m_TriTexUniLocationWndSize, (float)m_WndWidth, (float)m_WndHeight);
            glUniform1i(m_TriTexUniLocationTexture, 0);
        }
        
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)TextObj->m_TextVerts.size());
    }

    CHECK_GL_ERROR;
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::ChangeViewport(int _X0, int _Y0, int _Width, int _Height, int _OffsetX, int _OffsetY)
{
    // glViewport impacts the NDC; use glScissor instead
    m_OffsetX = _X0 + _OffsetX;
    m_OffsetY = _Y0 + _OffsetY;
    SetScissor(_X0, _Y0, _Width, _Height);
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::RestoreViewport()
{
    m_OffsetX = m_OffsetY = 0;
    SetScissor(0, 0, 0, 0);
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::SetScissor(int _X0, int _Y0, int _Width, int _Height)
{
    if( _Width>0 && _Height>0 )
    {
        glScissor(_X0-1, m_WndHeight-_Y0-_Height, _Width-1, _Height);
        glEnable(GL_SCISSOR_TEST);
    }
    else
        glDisable(GL_SCISSOR_TEST);
}

//  ---------------------------------------------------------------------------

void CTwGraphOpenGLCore::DrawTriangles(int _NumTriangles, int *_Vertices, color32 *_Colors, Cull _CullMode)
{
    assert(m_Drawing==true);

    const GLfloat dx = +0.0f;
    const GLfloat dy = +0.0f;

    // Backup states
    GLint prevCullFaceMode, prevFrontFace;
    glGetIntegerv(GL_CULL_FACE_MODE, &prevCullFaceMode);
    glGetIntegerv(GL_FRONT_FACE, &prevFrontFace);
    GLboolean prevCullEnable = glIsEnabled(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    if( _CullMode==CULL_CW )
        glFrontFace(GL_CCW);
    else if( _CullMode==CULL_CCW )
        glFrontFace(GL_CW);
    else
        glDisable(GL_CULL_FACE);

    glUseProgram(m_TriProgram);
    glBindVertexArray(m_TriVArray);
    glUniform2f(m_TriLocationOffset, (float)m_OffsetX+dx, (float)m_OffsetY+dy);
    glUniform2f(m_TriLocationWndSize, (float)m_WndWidth, (float)m_WndHeight);
    glDisableVertexAttribArray(2);

    size_t numVerts = 3*_NumTriangles;
    if( numVerts > m_TriBufferSize )
        ResizeTriBuffers(numVerts + 2048);
  
    glBindBuffer(GL_ARRAY_BUFFER, m_TriVertices);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numVerts*2*sizeof(int), _Vertices);
    glVertexAttribPointer(0, 2, GL_INT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, m_TriColors);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numVerts*sizeof(color32), _Colors);
    glVertexAttribPointer(1, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
    glEnableVertexAttribArray(1);
        
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)numVerts);

    // Reset states
    glCullFace(prevCullFaceMode);
    glFrontFace(prevFrontFace);
    if( prevCullEnable )
        glEnable(GL_CULL_FACE);
    else
        glDisable(GL_CULL_FACE);

    CHECK_GL_ERROR;
}

//  ---------------------------------------------------------------------------
