#define STB_DEFINE
#if _MSC_VER > 1700
typedef unsigned long       DWORD;
#endif
#include "stb/stb.h"
#include "stb_vec.h"

#define STB_DXT_IMPLEMENTATION
#include "stb/stb_dxt.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_PERLIN_IMPLEMENTATION
#include "stb/stb_perlin.h"
