#ifndef PSHAPE_H
#define PSHAPE_H

#include "tinyobjloader-c/tinyobj_loader_c.h"
#include "stb/stb.h"
#include "stb_vec.h"

typedef struct
{
    float bmin[3];
    float bmax[3]; // bounding box
    tinyobj_material_t* materials;
    size_t num_materials;
    tinyobj_shape_t* shapes;
    size_t num_shapes;
    tinyobj_attrib_t attrib;
} PShape;

PShape loadShape(const char* filename);
PShape createShape();
void deleteShape(PShape shp);
void shape(PShape shp);

#endif // PSHAPE_H

#ifdef PSHAPE_IMPLEMENTATION

#include <float.h>
#include <limits.h>

static void CalcNormal(float N[3], float v0[3], float v1[3], float v2[3]) {
    float v10[3];
    float v20[3];
    float len2;
    
    v10[0] = v1[0] - v0[0];
    v10[1] = v1[1] - v0[1];
    v10[2] = v1[2] - v0[2];
    
    v20[0] = v2[0] - v0[0];
    v20[1] = v2[1] - v0[1];
    v20[2] = v2[2] - v0[2];
    
    N[0] = v20[1] * v10[2] - v20[2] * v10[1];
    N[1] = v20[2] * v10[0] - v20[0] * v10[2];
    N[2] = v20[0] * v10[1] - v20[1] * v10[0];
    
    len2 = N[0] * N[0] + N[1] * N[1] + N[2] * N[2];
    if (len2 > 0.0f) {
        float len = (float)sqrt((double)len2);
        
        N[0] /= len;
        N[1] /= len;
    }
}

PShape loadShape(const char* filename)
{
    size_t data_len = 0;
    char* data = stb_file((char*)filename, &data_len);
    if (data == NULL) exit(-1);
    printf("filesize: %d\n", (int)data_len);
    
    PShape shp = {};
    
    {
        unsigned int flags = TINYOBJ_FLAG_TRIANGULATE;
        int ret = tinyobj_parse_obj(&shp.attrib, &shp.shapes, &shp.num_shapes, &shp.materials,
                                    &shp.num_materials, data, data_len, flags);
        free(data);
        
        if (ret != TINYOBJ_SUCCESS) {
            return shp;
        }
        
        printf("# of shapes    = %d\n", (int)shp.num_shapes);
        printf("# of materiasl = %d\n", (int)shp.num_materials);
    }
    
    shp.bmin[0] = shp.bmin[1] = shp.bmin[2] = FLT_MAX;
    shp.bmax[0] = shp.bmax[1] = shp.bmax[2] = -FLT_MAX;
    
    {
        // DrawObject o;
        float* vb;
        /* std::vector<float> vb; //  */
        size_t face_offset = 0;
        size_t i;
        
        /* Assume triangulated face. */
        size_t num_triangles = shp.attrib.num_face_num_verts;
        size_t stride = 9; /* 9 = pos(3float), normal(3float), color(3float) */
        
        vb = (float*)malloc(sizeof(float) * stride * num_triangles * 3);
        
        for (i = 0; i < shp.attrib.num_face_num_verts; i++) {
            size_t f;
            assert(shp.attrib.face_num_verts[i] % 3 ==
                   0); /* assume all triangle faces. */
            for (f = 0; f < (size_t)shp.attrib.face_num_verts[i] / 3; f++) {
                size_t k;
                float v[3][3];
                float n[3][3];
                float c[3];
                float len2;
                
                tinyobj_vertex_index_t idx0 = shp.attrib.faces[face_offset + 3 * f + 0];
                tinyobj_vertex_index_t idx1 = shp.attrib.faces[face_offset + 3 * f + 1];
                tinyobj_vertex_index_t idx2 = shp.attrib.faces[face_offset + 3 * f + 2];
                
                for (k = 0; k < 3; k++) {
                    int f0 = idx0.v_idx;
                    int f1 = idx1.v_idx;
                    int f2 = idx2.v_idx;
                    assert(f0 >= 0);
                    assert(f1 >= 0);
                    assert(f2 >= 0);
                    
                    v[0][k] = shp.attrib.vertices[3 * (size_t)f0 + k];
                    v[1][k] = shp.attrib.vertices[3 * (size_t)f1 + k];
                    v[2][k] = shp.attrib.vertices[3 * (size_t)f2 + k];
                    shp.bmin[k] = (v[0][k] < shp.bmin[k]) ? v[0][k] : shp.bmin[k];
                    shp.bmin[k] = (v[1][k] < shp.bmin[k]) ? v[1][k] : shp.bmin[k];
                    shp.bmin[k] = (v[2][k] < shp.bmin[k]) ? v[2][k] : shp.bmin[k];
                    shp.bmax[k] = (v[0][k] > shp.bmax[k]) ? v[0][k] : shp.bmax[k];
                    shp.bmax[k] = (v[1][k] > shp.bmax[k]) ? v[1][k] : shp.bmax[k];
                    shp.bmax[k] = (v[2][k] > shp.bmax[k]) ? v[2][k] : shp.bmax[k];
                }
                
                if (shp.attrib.num_normals > 0) {
                    int f0 = idx0.vn_idx;
                    int f1 = idx1.vn_idx;
                    int f2 = idx2.vn_idx;
                    if (f0 >= 0 && f1 >= 0 && f2 >= 0) {
                        assert(f0 < (int)shp.attrib.num_normals);
                        assert(f1 < (int)shp.attrib.num_normals);
                        assert(f2 < (int)shp.attrib.num_normals);
                        for (k = 0; k < 3; k++) {
                            n[0][k] = shp.attrib.normals[3 * (size_t)f0 + k];
                            n[1][k] = shp.attrib.normals[3 * (size_t)f1 + k];
                            n[2][k] = shp.attrib.normals[3 * (size_t)f2 + k];
                        }
                    } else { /* normal index is not defined for this face */
                        /* compute geometric normal */
                        CalcNormal(n[0], v[0], v[1], v[2]);
                        n[1][0] = n[0][0];
                        n[1][1] = n[0][1];
                        n[1][2] = n[0][2];
                        n[2][0] = n[0][0];
                        n[2][1] = n[0][1];
                        n[2][2] = n[0][2];
                    }
                } else {
                    /* compute geometric normal */
                    CalcNormal(n[0], v[0], v[1], v[2]);
                    n[1][0] = n[0][0];
                    n[1][1] = n[0][1];
                    n[1][2] = n[0][2];
                    n[2][0] = n[0][0];
                    n[2][1] = n[0][1];
                    n[2][2] = n[0][2];
                }
                
                for (k = 0; k < 3; k++) {
                    vb[(3 * i + k) * stride + 0] = v[k][0];
                    vb[(3 * i + k) * stride + 1] = v[k][1];
                    vb[(3 * i + k) * stride + 2] = v[k][2];
                    vb[(3 * i + k) * stride + 3] = n[k][0];
                    vb[(3 * i + k) * stride + 4] = n[k][1];
                    vb[(3 * i + k) * stride + 5] = n[k][2];
                    
                    /* Use normal as color. */
                    c[0] = n[k][0];
                    c[1] = n[k][1];
                    c[2] = n[k][2];
                    len2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                    if (len2 > 0.0f) {
                        float len = (float)sqrt((double)len2);
                        
                        c[0] /= len;
                        c[1] /= len;
                        c[2] /= len;
                    }
                    
                    vb[(3 * i + k) * stride + 6] = (c[0] * 0.5f + 0.5f);
                    vb[(3 * i + k) * stride + 7] = (c[1] * 0.5f + 0.5f);
                    vb[(3 * i + k) * stride + 8] = (c[2] * 0.5f + 0.5f);
                }
            }
            face_offset += (size_t)shp.attrib.face_num_verts[i];
        }
        
#if 0
        o.vb = 0;
        o.numTriangles = 0;
        if (num_triangles > 0) {
            glGenBuffers(1, &o.vb);
            glBindBuffer(GL_ARRAY_BUFFER, o.vb);
            glBufferData(GL_ARRAY_BUFFER, num_triangles * 3 * stride * sizeof(float),
                         vb, GL_STATIC_DRAW);
            o.numTriangles = (int)num_triangles;
        }
        
        free(vb);
#endif
    }

    return shp;
}

PShape createShape()
{
    PShape shp = {};
    tinyobj_attrib_init(&shp.attrib);

    return shp;
}

void deleteShape(PShape shp)
{
    if (shp.attrib.num_vertices == 0) return;
    
    tinyobj_attrib_free(&shp.attrib);
    tinyobj_shapes_free(shp.shapes, shp.num_shapes);
    tinyobj_materials_free(shp.materials, shp.num_materials);
    
    shp.attrib.num_vertices = 0;
}

void shape(PShape pshape)
{
    
}


#endif // PSHAPE_IMPLEMENTATION

