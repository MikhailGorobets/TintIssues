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
#include "src/tint/lang/core/common/multiplanar_options.h"

#include "src/tint/lang/wgsl/reader/parser/parser.h"
#include "src/tint/lang/wgsl/reader/program_to_ir/program_to_ir.h"
#include "src/tint/lang/wgsl/ast/transform/binding_remapper.h"


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

#define LOG_WARNING_MESSAGE(...)                            \
    do                                                      \
    {                                                       \
        std::string Message = ConcatenateArgs(__VA_ARGS__); \
        std::cerr << "Warning: " << Message << std::endl;   \
    } while (false)

#define LOG_INFO_MESSAGE(...)                               \
    do                                                      \
    {                                                       \
        std::string Message = ConcatenateArgs(__VA_ARGS__); \
        std::cerr << "Info: " << Message << std::endl;      \
    } while (false)

namespace HLSL
{

const std::string TestCS = R"(
RWTexture2D<float4> Tex2D_0;
RWTexture2D<float4> Tex2D_1;
Texture2D<float4>   Tex2D;

[numthreads(8, 8, 1)]
void main(uint3 Gid : SV_GroupID,
          uint3 GTid : SV_GroupThreadID)
{
    float4 Color = Tex2D.Load(int3(GTid.xy, 0));
    Tex2D_0[GTid.xy] = float4(0.0, 0.0, 0.0, 1.0);
    Tex2D_1[GTid.xy] = Color; 
}
)";
;
} // namespace HLSL


using BindingRemapingInfo = std::unordered_map<std::string, std::tuple<uint32_t, uint32_t>>;

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

    return std::move(GenerationResult->wgsl);
}

std::string RamapBindingGroupLayoitsWGSL(const std::string& WGSL, const BindingRemapingInfo& RemapIndices)
{
    tint::Source::File srcFile("", WGSL);
    tint::Program      Program = tint::wgsl::reader::Parse(&srcFile, {tint::wgsl::AllowedFeatures::Everything()});

    if (!Program.IsValid())
        LOG_ERROR_AND_THROW("Tint WGSL reader failure:\nParser: ", Program.Diagnostics().Str(), "\n");

    tint::ast::transform::BindingRemapper::BindingPoints BindingPoints;

    tint::inspector::Inspector Inspector{Program};
    for (auto& EntryPoint : Inspector.GetEntryPoints())
    {
        for (auto& Binding : Inspector.GetResourceBindings(EntryPoint.name))
        {

            auto& BindIndices = RemapIndices.find(Binding.variable_name);
            if (BindIndices != RemapIndices.end())
            {
                auto& [Group, Index] = BindIndices->second;
                BindingPoints.emplace(tint::ast::transform::BindingPoint{Binding.bind_group, Binding.binding}, tint::ast::transform::BindingPoint{Group, Index});
            }
            else
                LOG_WARNING_MESSAGE("Binding for variable '", Binding.variable_name, "' not found in the remap indices");
        }
    }

    tint::ast::transform::Manager Manager;
    tint::ast::transform::DataMap Inputs;
    tint::ast::transform::DataMap Outputs;

    Inputs.Add<tint::ast::transform::BindingRemapper::Remappings>(BindingPoints, tint::ast::transform::BindingRemapper::AccessControls{}, true);
    Manager.Add<tint::ast::transform::BindingRemapper>();
    tint::ast::transform::Output TransformResult = Manager.Run(Program, Inputs, Outputs);

    auto GenerationResult = tint::wgsl::writer::Generate(TransformResult.program, {});

    if (GenerationResult != tint::Success)
        LOG_ERROR_AND_THROW("Tint WGSL writer failure:\nGeneate: ", GenerationResult.Failure().reason.Str(), "\n");

    return std::move(GenerationResult->wgsl);
}

int main(int argc, const char* argv[])
{
    try
    {
        BindingRemapingInfo RemapIndices;
        RemapIndices["Tex2D_0"] = {1, 0};
        RemapIndices["Tex2D_1"] = {1, 1};
        RemapIndices["Tex2D"]   = {2, 0};

        auto SPIRV = ConvertHLSLtoSPIRV(HLSL::TestCS);
        auto WGSL  = RamapBindingGroupLayoitsWGSL(ConvertSPIRVtoWGSL(SPIRV), RemapIndices);
        std::cout << WGSL << "\n";
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}
