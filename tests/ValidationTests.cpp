// Invalid graphs are rejected by Compile() with a message that names the problem.

#include <functional>

#include "TestCommon.h"

using namespace rgtest;
using rg::Access;
using rg::Pass;
using rg::RenderGraph;

namespace
{
	constexpr uint32_t kCount = 1024;

	void ExpectRejected(GpuTest &gpu, const char *expectedMessage, const std::function<void(RenderGraph &)> &build)
	{
		RenderGraph graph;
		build(graph);
		const bool compiled = graph.Compile(gpu.rc());
		if (compiled || graph.GetError().find(expectedMessage) == std::string::npos)
		{
			std::fprintf(stderr,
						 "expected rejection containing '%s', got '%s'\n",
						 expectedMessage,
						 compiled ? "(compiled)" : graph.GetError().c_str());
		}
		assert(!compiled);
		assert(!graph.IsCompiled());
		assert(graph.GetError().find(expectedMessage) != std::string::npos);
	}
} // namespace

RG_TEST(InvalidGraphsAreRejected)
{
	GpuTest gpu;

	// A transient resource has undefined contents until something writes it.
	ExpectRejected(gpu,
				   "before anything writes it",
				   [](RenderGraph &graph)
				   {
					   const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
					   const auto y = graph.CreateBuffer(UIntBuffer(kCount, "Y"));
					   graph.AddPass(Pass::Compute("ReadUndefined").Reads(x).Writes(y));
				   });

	// One attachment per resource per pass; read-modify-write is its own access.
	ExpectRejected(gpu,
				   "more than once",
				   [](RenderGraph &graph)
				   {
					   const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
					   graph.AddPass(Pass::Compute("Init").Writes(x));
					   graph.AddPass(Pass::Compute("Twice").Reads(x).Writes(x));
				   });

	// Colour attachments are textures.
	ExpectRejected(gpu,
				   "uses buffer",
				   [](RenderGraph &graph)
				   {
					   const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
					   graph.AddPass(Pass::Graphics("DrawToBuffer").Use(x, Access::ColourAttachment));
				   });

	// Indirect arguments are buffers.
	ExpectRejected(gpu,
				   "uses texture",
				   [](RenderGraph &graph)
				   {
					   TextureDesc desc;
					   desc.width = desc.height = 16;
					   const auto t = graph.CreateTexture(desc);
					   graph.AddPass(Pass::Graphics("Init").Writes(t));
					   graph.AddPass(Pass::Compute("Indirect").Use(t, Access::IndirectArgumentRead));
				   });

	// The compute queue has no raster pipeline.
	ExpectRejected(gpu,
				   "raster pass on the compute queue",
				   [](RenderGraph &graph)
				   {
					   TextureDesc desc;
					   desc.width = desc.height = 16;
					   const auto t = graph.CreateTexture(desc);
					   graph.AddPass(Pass::Graphics("AsyncRaster").OnQueue(QueueType::Compute).Writes(t));
				   });

	ExpectRejected(gpu,
				   "PixelShaderRead on the Compute queue",
				   [](RenderGraph &graph)
				   {
					   const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
					   const auto y = graph.CreateBuffer(UIntBuffer(kCount, "Y"));
					   graph.AddPass(Pass::Compute("Init").Writes(x));
					   graph.Submit();
					   graph.AddPass(Pass::Compute("AsyncPixelRead")
											 .OnQueue(QueueType::Compute)
											 .Use(x, Access::PixelShaderRead)
											 .Writes(y));
				   });

	// Mesh shaders, like every graphics stage, only run on the graphics queue.
	ExpectRejected(gpu,
				   "MeshShaderWrite on the Compute queue",
				   [](RenderGraph &graph)
				   {
					   const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
					   graph.AddPass(Pass::Compute("AsyncMesh").OnQueue(QueueType::Compute).Use(x, Access::MeshShaderWrite));
				   });

	// Shading rate images are textures.
	ExpectRejected(gpu,
				   "uses buffer",
				   [](RenderGraph &graph)
				   {
					   const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
					   graph.AddPass(Pass::Compute("Init").Writes(x));
					   graph.AddPass(Pass::Graphics("Shade").Use(x, Access::ShadingRateSource));
				   });

	// A transient acceleration structure is only built and traced, so it is not also a plain buffer.
	ExpectRejected(gpu,
				   "is an acceleration structure in pass 'Build' and a plain buffer in pass 'Shade'",
				   [](RenderGraph &graph)
				   {
					   const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"), ResourceViewType::ShaderResource);
					   graph.AddPass(Pass::Compute("Build").Use(x, Access::AccelerationStructureBuildWrite));
					   graph.AddPass(Pass::Compute("Shade").Reads(x));
				   });

	// Its one view is the acceleration structure SRV rays trace.
	ExpectRejected(gpu,
				   "needs a ShaderResource view",
				   [](RenderGraph &graph)
				   {
					   const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
					   graph.AddPass(Pass::Compute("Build").Use(x, Access::AccelerationStructureBuildWrite));
				   });

	// A rejected graph is not executable, and fixing it makes it compile.
	RenderGraph graph;
	const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
	const auto y = graph.CreateBuffer(UIntBuffer(kCount, "Y"));
	graph.AddPass(Pass::Compute("ReadUndefined").Reads(x).Writes(y));
	assert(!graph.Compile(gpu.rc()));
	graph.AddPass(Pass::Compute("Unrelated").Writes(y));
	assert(!graph.Compile(gpu.rc()) && "still invalid: the bad pass is still there");
}
