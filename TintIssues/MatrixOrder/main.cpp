#define ENABLE_HLSL

#include <iostream>
#include <exception>

#include <tint/tint.h>
#include <glslang/Public/ShaderLang.h>

#include <SPIRV/GlslangToSpv.h>

#include <spirv-tools/optimizer.hpp>
#include <spirv-tools/libspirv.hpp>

#include "source/opt/ir_builder.h"
#include "source/opt/build_module.h"
#include "source/opt/def_use_manager.h"


namespace spvtools::opt
{
class Instruction;
class IRContext;
} // namespace spvtools::opt

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

const std::string CubeTextureVS = R"(
cbuffer Constants
{
    float4x4 g_WorldViewProj;
};

struct VSInput
{
    float3 Pos : ATTRIB0;
    float4 Color : ATTRIB1;
};

struct PSInput
{
    float4 Pos : SV_POSITION;
    float4 Color : COLOR0;
};

void main(in VSInput VSIn,
          out PSInput PSIn)
{
    PSIn.Pos = mul(float4(VSIn.Pos, 1.0), g_WorldViewProj);
    PSIn.Color = VSIn.Color;
}
)";
;
} // namespace HLSL


std::vector<uint32_t> OptimizeSPIRV(const std::vector<uint32_t>& SrcSPIRV, spv_target_env TargetEnv)
{
    std::unique_ptr<spvtools::opt::IRContext> Context = spvtools::BuildModule(TargetEnv, {}, SrcSPIRV.data(), SrcSPIRV.size());
    if (!Context)
        LOG_ERROR_AND_THROW("Failed to parse SPIR-V binary");

    for (auto& Instruction : Context->module()->annotations())
    {
        if (Instruction.opcode() == spv::Op::OpMemberDecorate)
        {
            uint32_t TargetIdx  = Instruction.GetSingleWordInOperand(0);
            uint32_t FieldIdx   = Instruction.GetSingleWordInOperand(1);
            uint32_t Decoration = Instruction.GetSingleWordInOperand(2);

            if (Decoration == static_cast<uint32_t>(spv::Decoration::RowMajor))
            {
                Instruction.SetInOperand(2, {static_cast<uint32_t>(spv::Decoration::ColMajor)});

                auto* Definition = Context->get_def_use_mgr()->GetDef(TargetIdx);
                //TODO: Ddd matrix transposition
                std::cout << "Converted RowMajor matrix with ID: " << TargetIdx << " to ColumnMajor." << std::endl;
            }
        }
    }

    std::vector<uint32_t> PatchedSPIRV;
    Context->module()->ToBinary(&PatchedSPIRV, false);

    spvtools::Optimizer SpirvOptimizer(TargetEnv);
    SpirvOptimizer.RegisterLegalizationPasses();
    SpirvOptimizer.RegisterPerformancePasses();

    std::vector<uint32_t> OptimizedSPIRV;
    if (!SpirvOptimizer.Run(PatchedSPIRV.data(), PatchedSPIRV.size(), &OptimizedSPIRV))
        OptimizedSPIRV.clear();

    return OptimizedSPIRV;
}

std::vector<uint32_t> ConvertHLSLtoSPIRV(const std::string& HLSL)
{
    GlslangInitilizer InitScope{};

    glslang::TShader Shader{EShLangVertex};

    auto* pHLSL = HLSL.c_str();
    Shader.setStrings(&pHLSL, 1);
    Shader.setEntryPoint("main");
    Shader.setEnvInput(glslang::EShSourceHlsl, Shader.getStage(), glslang::EShClientVulkan, 100);
    Shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    Shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    Shader.setEnvTargetHlslFunctionality1();
    Shader.setHlslIoMapping(true);
    Shader.setDxPositionW(true);

    TBuiltInResource Resources;
    Shader.parse(&Resources, 100, false, EShMsgDefault);

    glslang::TProgram Program;
    Program.addShader(&Shader);

    if (!Program.link(EShMsgDefault))
        LOG_ERROR_AND_THROW("Failed to link program: \n", Program.getInfoLog());

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
        auto SPIRV = ConvertHLSLtoSPIRV(HLSL::CubeTextureVS);
        auto WGSL  = ConvertSPIRVtoWGSL(SPIRV);
        std::cout << WGSL << "\n";
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}
