#define ENABLE_HLSL

#include <tint/tint.h>
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>
#include <spirv-tools/optimizer.hpp>

#include <iostream>
#include <exception>

struct TintInitializer
{
    TintInitializer()
    {
        tint::Initialize();
    }
    ~TintInitializer()
    {
        tint::Shutdown();
    }
};

struct GlslangInitilizer
{

    GlslangInitilizer()
    {
        glslang::InitializeProcess();
    }
    ~GlslangInitilizer()
    {
        glslang::FinalizeProcess();
    }
};

template <typename... Args>
std::string ConcatenateArgs(Args... args)
{
    std::ostringstream OutputStream;
    (OutputStream << ... << args); // Fold expression to concatenate all arguments
    return OutputStream.str();
}

#define LOG_ERROR_AND_THROW(...)                                        \
    do                                                                  \
    {                                                                   \
        std::string Message = ConcatenateArgs(__VA_ARGS__);             \
        std::cerr << "Error: " << Message << " (in " << __FILE__ << ":" \
                  << __LINE__ << ", function " << __func__ << ")"       \
                  << std::endl;                                         \
        throw std::runtime_error(Message);                              \
    } while (false)

namespace HLSL
{

const std::string GenerateMipsCS = R"(
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author:  James Stanard
//

#ifndef NON_POWER_OF_TWO
#    define NON_POWER_OF_TWO 0
#endif

RWTexture2DArray<float4> OutMip1;
RWTexture2DArray<float4> OutMip2;
RWTexture2DArray<float4> OutMip3;
RWTexture2DArray<float4> OutMip4;
Texture2DArray<float4>   SrcTex;
SamplerState             BilinearClamp;

cbuffer CB
{
    uint   SrcMipLevel;  // Texture level of source mip
    uint   NumMipLevels; // Number of OutMips to write: [1, 4]
    uint   FirstArraySlice;
    uint   Dummy;
    float2 TexelSize; // 1.0 / OutMip1.Dimensions
}

// The reason for separating channels is to reduce bank conflicts in the
// local data memory controller.  A large stride will cause more threads
// to collide on the same memory bank.
groupshared float gs_R[64];
groupshared float gs_G[64];
groupshared float gs_B[64];
groupshared float gs_A[64];

void StoreColor(uint Index, float4 Color)
{
    gs_R[Index] = Color.r;
    gs_G[Index] = Color.g;
    gs_B[Index] = Color.b;
    gs_A[Index] = Color.a;
}

float4 LoadColor(uint Index)
{
    return float4(gs_R[Index], gs_G[Index], gs_B[Index], gs_A[Index]);
}

float3 LinearToSRGB(float3 x)
{
    // This is exactly the sRGB curve
    //return x < 0.0031308 ? 12.92 * x : 1.055 * pow(abs(x), 1.0 / 2.4) - 0.055;

    // This is cheaper but nearly equivalent
    return x < 0.0031308 ? 12.92 * x : 1.13005 * sqrt(abs(x - 0.00228)) - 0.13448 * x + 0.005719;
}

float4 PackColor(float4 Linear)
{
#ifdef CONVERT_TO_SRGB
    return float4(LinearToSRGB(Linear.rgb), Linear.a);
#else
    return Linear;
#endif
}

[numthreads(8, 8, 1)] 
void main(uint GI : SV_GroupIndex, uint3 DTid: SV_DispatchThreadID)
{
    uint2 DstMipSize;
    uint  Elements;
    SrcTex.GetDimensions(DstMipSize.x, DstMipSize.y, Elements);
    DstMipSize >>= SrcMipLevel;
    bool IsValidThread = all(DTid.xy < DstMipSize);
    uint ArraySlice    = FirstArraySlice + DTid.z;

    float4 Src1 = 0;
    if (IsValidThread)
    {
        // One bilinear sample is insufficient when scaling down by more than 2x.
        // You will slightly undersample in the case where the source dimension
        // is odd.  This is why it's a really good idea to only generate mips on
        // power-of-two sized textures.  Trying to handle the undersampling case
        // will force this shader to be slower and more complicated as it will
        // have to take more source texture samples.
#if NON_POWER_OF_TWO == 0
        float2 UV = TexelSize * (DTid.xy + 0.5);
        Src1      = SrcTex.SampleLevel(BilinearClamp, float3(UV, ArraySlice), SrcMipLevel);
#elif NON_POWER_OF_TWO == 1
        // > 2:1 in X dimension
        // Use 2 bilinear samples to guarantee we don't undersample when downsizing by more than 2x
        // horizontally.
        float2 UV1 = TexelSize * (DTid.xy + float2(0.25, 0.5));
        float2 Off = TexelSize * float2(0.5, 0.0);
        Src1       = 0.5 * (SrcTex.SampleLevel(BilinearClamp, float3(UV1, ArraySlice), SrcMipLevel) + SrcTex.SampleLevel(BilinearClamp, float3(UV1 + Off, ArraySlice), SrcMipLevel));
#elif NON_POWER_OF_TWO == 2
        // > 2:1 in Y dimension
        // Use 2 bilinear samples to guarantee we don't undersample when downsizing by more than 2x
        // vertically.
        float2 UV1 = TexelSize * (DTid.xy + float2(0.5, 0.25));
        float2 Off = TexelSize * float2(0.0, 0.5);
        Src1       = 0.5 * (SrcTex.SampleLevel(BilinearClamp, float3(UV1, ArraySlice), SrcMipLevel) + SrcTex.SampleLevel(BilinearClamp, float3(UV1 + Off, ArraySlice), SrcMipLevel));
#elif NON_POWER_OF_TWO == 3
        // > 2:1 in in both dimensions
        // Use 4 bilinear samples to guarantee we don't undersample when downsizing by more than 2x
        // in both directions.
        float2 UV1 = TexelSize * (DTid.xy + float2(0.25, 0.25));
        float2 Off = TexelSize * 0.5;
        Src1 += SrcTex.SampleLevel(BilinearClamp, float3(UV1, ArraySlice), SrcMipLevel);
        Src1 += SrcTex.SampleLevel(BilinearClamp, float3(UV1 + float2(Off.x, 0.0), ArraySlice), SrcMipLevel);
        Src1 += SrcTex.SampleLevel(BilinearClamp, float3(UV1 + float2(0.0, Off.y), ArraySlice), SrcMipLevel);
        Src1 += SrcTex.SampleLevel(BilinearClamp, float3(UV1 + float2(Off.x, Off.y), ArraySlice), SrcMipLevel);
        Src1 *= 0.25;
#endif

        OutMip1[uint3(DTid.xy, ArraySlice)] = PackColor(Src1);
    }

    // A scalar (constant) branch can exit all threads coherently.
    if (NumMipLevels == 1)
        return;

    if (IsValidThread)
    {
        // Without lane swizzle operations, the only way to share data with other
        // threads is through LDS.
        StoreColor(GI, Src1);
    }

    // This guarantees all LDS writes are complete and that all threads have
    // executed all instructions so far (and therefore have issued their LDS
    // write instructions.)
    GroupMemoryBarrierWithGroupSync();

    if (IsValidThread)
    {
        // With low three bits for X and high three bits for Y, this bit mask
        // (binary: 001001) checks that X and Y are even.
        if ((GI & 0x9) == 0)
        {
            float4 Src2 = LoadColor(GI + 0x01);
            float4 Src3 = LoadColor(GI + 0x08);
            float4 Src4 = LoadColor(GI + 0x09);
            Src1        = 0.25 * (Src1 + Src2 + Src3 + Src4);

            OutMip2[uint3(DTid.xy / 2, ArraySlice)] = PackColor(Src1);
            StoreColor(GI, Src1);
        }
    }

    if (NumMipLevels == 2)
        return;

    GroupMemoryBarrierWithGroupSync();

    if (IsValidThread)
    {
        // This bit mask (binary: 011011) checks that X and Y are multiples of four.
        if ((GI & 0x1B) == 0)
        {
            float4 Src2 = LoadColor(GI + 0x02);
            float4 Src3 = LoadColor(GI + 0x10);
            float4 Src4 = LoadColor(GI + 0x12);
            Src1        = 0.25 * (Src1 + Src2 + Src3 + Src4);

            OutMip3[uint3(DTid.xy / 4, ArraySlice)] = PackColor(Src1);
            StoreColor(GI, Src1);
        }
    }

    if (NumMipLevels == 3)
        return;

    GroupMemoryBarrierWithGroupSync();

    if (IsValidThread)
    {
        // This bit mask would be 111111 (X & Y multiples of 8), but only one
        // thread fits that criteria.
        if (GI == 0)
        {
            float4 Src2 = LoadColor(GI + 0x04);
            float4 Src3 = LoadColor(GI + 0x20);
            float4 Src4 = LoadColor(GI + 0x24);
            Src1        = 0.25 * (Src1 + Src2 + Src3 + Src4);

            OutMip4[uint3(DTid.xy / 8, ArraySlice)] = PackColor(Src1);
        }
    }
}

)";
;
} // namespace HLSL


std::vector<uint32_t> OptimizeSPIRV(const std::vector<uint32_t>& SrcSPIRV, spv_target_env TargetEnv)
{
    spvtools::Optimizer SpirvOptimizer(TargetEnv);
    SpirvOptimizer.RegisterLegalizationPasses();
    SpirvOptimizer.RegisterPerformancePasses();

    std::vector<uint32_t> OptimizedSPIRV;
    if (!SpirvOptimizer.Run(SrcSPIRV.data(), SrcSPIRV.size(), &OptimizedSPIRV))
        OptimizedSPIRV.clear();

    return OptimizedSPIRV;
}

std::vector<uint32_t> ConvertHLSLtoSPIRV(const std::string& HLSL)
{
    GlslangInitilizer InitScope{};

    glslang::TShader Shader{EShLangCompute};

    auto* pHLSL = HLSL.c_str();
    Shader.setStrings(&pHLSL, 1);
    Shader.setEntryPoint("main");
    Shader.setEnvInput(glslang::EShSourceHlsl, Shader.getStage(), glslang::EShClientVulkan, 100);
    Shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    Shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    Shader.setEnvTargetHlslFunctionality1();
    Shader.setHlslIoMapping(true);
    Shader.setDxPositionW(true);
    Shader.setAutoMapBindings(true);
    Shader.setAutoMapLocations(true);

    TBuiltInResource Resources;
    Shader.parse(&Resources, 100, false, EShMsgDefault);

    glslang::TProgram Program;
    Program.addShader(&Shader);

    if (!Program.link(EShMsgDefault))
        LOG_ERROR_AND_THROW("Failed to link program: \n", Program.getInfoLog());

    if (!Program.mapIO())
        LOG_ERROR_AND_THROW("Failed to map IO: \n", Program.getInfoLog());

    std::vector<uint32_t> SPIRV;
    glslang::GlslangToSpv(*Program.getIntermediate(Shader.getStage()), SPIRV);

    auto OptimizedSPIRV = OptimizeSPIRV(SPIRV, SPV_ENV_VULKAN_1_0);

    if (!OptimizedSPIRV.empty())
        return OptimizedSPIRV;

    LOG_ERROR_AND_THROW("Failed to optimize SPIR-V.");
}

std::string ConvertSPIRVtoWGSL(const std::vector<uint32_t>& SPIRV)
{
    TintInitializer InitScope{};

    std::string   WGSL;
    tint::Program Program = tint::spirv::reader::Read(SPIRV, {true});

    if (!Program.IsValid())
        LOG_ERROR_AND_THROW("Tint SPIR-V reader failure:\nParser: ", Program.Diagnostics().Str(), "\n");

    auto GenerationResult = tint::wgsl::writer::Generate(Program, {});

    if (GenerationResult != tint::Success)
        LOG_ERROR_AND_THROW("Tint WGSL writer failure:\nGeneate: ", GenerationResult.Failure().reason.Str(), "\n");
    WGSL = std::move(GenerationResult->wgsl);
    return WGSL;
}

int main(int argc, const char* argv[])
{
    try
    {
        auto SPIRV = ConvertHLSLtoSPIRV(HLSL::GenerateMipsCS);
        auto WGSL  = ConvertSPIRVtoWGSL(SPIRV);
        std::cout << WGSL << "\n";
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}
