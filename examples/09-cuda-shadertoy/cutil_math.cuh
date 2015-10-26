// 3D vector algebra from cutil_math.h
// add
inline __host__ __device__ float3 operator+(float3 a, float3 b){return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);}
inline __host__ __device__ void operator+=(float3 &a, float3 b){a.x += b.x; a.y += b.y; a.z += b.z;}
inline __host__ __device__ float3 operator+(float3 a, float b){return make_float3(a.x + b, a.y + b, a.z + b);}
inline __host__ __device__ float3 operator+(float b, float3 a){return make_float3(b + a.x, b + a.y, b + a.z);}
inline __host__ __device__ void operator+=(float3 &a, float b){a.x += b; a.y += b; a.z += b;}
// subtract
inline __host__ __device__ float3 operator-(float3 a, float3 b){return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);}
inline __host__ __device__ void operator-=(float3 &a, float3 b){a.x -= b.x; a.y -= b.y; a.z -= b.z;}
inline __host__ __device__ float3 operator-(float3 a, float b){return make_float3(a.x - b, a.y - b, a.z - b);}
inline __host__ __device__ float3 operator-(float b, float3 a){return make_float3(b - a.x, b - a.y, b - a.z);}
inline __host__ __device__ void operator-=(float3 &a, float b){a.x -= b; a.y -= b; a.z -= b;}
// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b){return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);}
inline __host__ __device__ void operator*=(float3 &a, float3 b){a.x *= b.x; a.y *= b.y; a.z *= b.z;}
inline __host__ __device__ float3 operator*(float3 a, float b){return make_float3(a.x * b, a.y * b, a.z * b);}
inline __host__ __device__ float3 operator*(float b, float3 a){return make_float3(b * a.x, b * a.y, b * a.z);}
inline __host__ __device__ void operator*=(float3 &a, float b){a.x *= b; a.y *= b; a.z *= b;}
// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b){return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);}
inline __host__ __device__ void operator/=(float3 &a, float3 b){a.x /= b.x; a.y /= b.y; a.z /= b.z;}
inline __host__ __device__ float3 operator/(float3 a, float b){return make_float3(a.x / b, a.y / b, a.z / b);}
inline __host__ __device__ void operator/=(float3 &a, float b){a.x /= b; a.y /= b; a.z /= b;}
inline __host__ __device__ float3 operator/(float b, float3 a){return make_float3(b / a.x, b / a.y, b / a.z);}
// min
inline __host__ __device__ float3 fminf(float3 a, float3 b){return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));}
// max
inline __host__ __device__ float3 fmaxf(float3 a, float3 b){return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));}
// lerp
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t){return a + t*(b - a);}
// clamp value v between a and b
inline __device__ __host__ float clamp(float f, float a, float b){return fmaxf(a, fminf(f, b));}
inline __device__ __host__ float3 clamp(float3 v, float a, float b){return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b){return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));}
// dot product
inline __host__ __device__ float dot(float3 a, float3 b){return a.x * b.x + a.y * b.y + a.z * b.z;}
// length
inline __host__ __device__ float length(float3 v){return sqrtf(dot(v, v));}
// normalize
inline __host__ __device__ float3 normalize(float3 v){float invLen = rsqrtf(dot(v, v));return v * invLen;}
// floor
inline __host__ __device__ float3 floorf(float3 v){return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));}
// frac
inline __host__ __device__ float fracf(float v){return v - floorf(v);}
inline __host__ __device__ float3 fracf(float3 v){return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));}
// fmod
inline __host__ __device__ float3 fmodf(float3 a, float3 b){return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));}
// absolute value
inline __host__ __device__ float3 fabs(float3 v){return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));}
// reflect 
//returns reflection of incident ray I around surface normal N
// N should be normalized, reflected vector's length is equal to length of I
inline __host__ __device__ float3 reflect(float3 i, float3 n){return i - 2.0f * n * dot(n, i);}
// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b){return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);}
