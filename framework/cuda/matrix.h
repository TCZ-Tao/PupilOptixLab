#pragma once

#include "vec_math.h"

struct mat3x3 {
    float3 r0, r1, r2;
};
CUDA_INLINE CUDA_HOSTDEVICE mat3x3 transpose(const mat3x3 &m) noexcept {
    mat3x3 ret;
    ret.r0 = make_float3(m.r0.x, m.r1.x, m.r2.x);
    ret.r1 = make_float3(m.r0.y, m.r1.y, m.r2.y);
    ret.r2 = make_float3(m.r0.z, m.r1.z, m.r2.z);
    return ret;
}

CUDA_INLINE CUDA_HOSTDEVICE mat3x3 make_mat3x3(const float3 &v) noexcept {
    return mat3x3{ 
        make_float3(v.x, 0.f, 0.f), 
        make_float3(0.f, v.y, 0.f), 
        make_float3(0.f, 0.f, v.z) 
    };
}

CUDA_INLINE CUDA_HOSTDEVICE float3 operator*(const mat3x3 &m, const float3 &v) noexcept {
    return make_float3(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

CUDA_INLINE CUDA_HOSTDEVICE mat3x3 operator*(const mat3x3 &m, const mat3x3 &m2) noexcept {
    mat3x3 ret;
    ret.r0.x = dot(m.r0, make_float3(m2.r0.x, m2.r1.x, m2.r2.x));
    ret.r0.y = dot(m.r0, make_float3(m2.r0.y, m2.r1.y, m2.r2.y));
    ret.r0.z = dot(m.r0, make_float3(m2.r0.z, m2.r1.z, m2.r2.z));
    ret.r1.x = dot(m.r1, make_float3(m2.r0.x, m2.r1.x, m2.r2.x));
    ret.r1.y = dot(m.r1, make_float3(m2.r0.y, m2.r1.y, m2.r2.y));
    ret.r1.z = dot(m.r1, make_float3(m2.r0.z, m2.r1.z, m2.r2.z));
    ret.r2.x = dot(m.r2, make_float3(m2.r0.x, m2.r1.x, m2.r2.x));
    ret.r2.y = dot(m.r2, make_float3(m2.r0.y, m2.r1.y, m2.r2.y));
    ret.r2.z = dot(m.r2, make_float3(m2.r0.z, m2.r1.z, m2.r2.z));
    return ret;
}

struct mat4x4 {
    float4 r0, r1, r2, r3;
};

CUDA_INLINE CUDA_HOSTDEVICE mat4x4 make_mat4x4(const mat3x3 &m) noexcept {
    mat4x4 ret;
    ret.r0 = make_float4(m.r0, 0.f);
    ret.r1 = make_float4(m.r1, 0.f);
    ret.r2 = make_float4(m.r2, 0.f);
    ret.r3 = make_float4(0.f, 0.f, 0.f, 1.f);
    return ret;
}

CUDA_INLINE CUDA_HOSTDEVICE float4 operator*(const mat4x4 &m, const float4 &v) noexcept {
    return make_float4(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v), dot(m.r3, v));
}

CUDA_INLINE CUDA_HOSTDEVICE mat4x4 operator*(const mat4x4 &m, const mat4x4 &m2) noexcept {
    mat4x4 ret;
    ret.r0.x = dot(m.r0, make_float4(m2.r0.x, m2.r1.x, m2.r2.x, m2.r3.x));
    ret.r0.y = dot(m.r0, make_float4(m2.r0.y, m2.r1.y, m2.r2.y, m2.r3.y));
    ret.r0.z = dot(m.r0, make_float4(m2.r0.z, m2.r1.z, m2.r2.z, m2.r3.z));
    ret.r0.w = dot(m.r0, make_float4(m2.r0.w, m2.r1.w, m2.r2.w, m2.r3.w));
    ret.r1.x = dot(m.r1, make_float4(m2.r0.x, m2.r1.x, m2.r2.x, m2.r3.x));
    ret.r1.y = dot(m.r1, make_float4(m2.r0.y, m2.r1.y, m2.r2.y, m2.r3.y));
    ret.r1.z = dot(m.r1, make_float4(m2.r0.z, m2.r1.z, m2.r2.z, m2.r3.z));
    ret.r1.w = dot(m.r1, make_float4(m2.r0.w, m2.r1.w, m2.r2.w, m2.r3.w));
    ret.r2.x = dot(m.r2, make_float4(m2.r0.x, m2.r1.x, m2.r2.x, m2.r3.x));
    ret.r2.y = dot(m.r2, make_float4(m2.r0.y, m2.r1.y, m2.r2.y, m2.r3.y));
    ret.r2.z = dot(m.r2, make_float4(m2.r0.z, m2.r1.z, m2.r2.z, m2.r3.z));
    ret.r2.w = dot(m.r2, make_float4(m2.r0.w, m2.r1.w, m2.r2.w, m2.r3.w));
    ret.r3.x = dot(m.r3, make_float4(m2.r0.x, m2.r1.x, m2.r2.x, m2.r3.x));
    ret.r3.y = dot(m.r3, make_float4(m2.r0.y, m2.r1.y, m2.r2.y, m2.r3.y));
    ret.r3.z = dot(m.r3, make_float4(m2.r0.z, m2.r1.z, m2.r2.z, m2.r3.z));
    ret.r3.w = dot(m.r3, make_float4(m2.r0.w, m2.r1.w, m2.r2.w, m2.r3.w));
    return ret;
}
