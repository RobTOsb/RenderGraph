// Access types: every barrier state is valid for the debug layer, and mesh shading and ray tracing order real work.

#include <optional>
#include <set>
#include <string>

#include "TestCommon.h"

using namespace rgtest;
using rg::Access;
using rg::Pass;
using rg::PassContext;
using rg::RenderGraph;

namespace
{
	constexpr uint32_t kSection = RenderGraph::kFirstUserSection;
	constexpr QueueType kGraphics = QueueType::Graphics;
	constexpr uint32_t kGrid = 8; // the mesh test's viewport and the ray tracing tests' ray grid
	constexpr uint32_t kElements = kGrid * kGrid;

	std::string ShaderPath(const char *file)
	{
		return std::string(RG_TEST_SHADER_DIR) + "/" + file;
	}

	// The kinds of resource an access can apply to, each with the accesses that fit it.
	enum class Subject
	{
		Buffer,
		ColourTexture,
		DepthTexture,
		ShadingRateTexture,
		AccelerationStructure,
	};

	const char *ToString(Subject subject)
	{
		switch (subject)
		{
			case Subject::Buffer:
				return "buffer";
			case Subject::ColourTexture:
				return "colour texture";
			case Subject::DepthTexture:
				return "depth texture";
			case Subject::ShadingRateTexture:
				return "shading rate texture";
			case Subject::AccelerationStructure:
				return "acceleration structure";
		}
		return "?";
	}

	std::vector<Access> SubjectAccesses(Subject subject)
	{
		switch (subject)
		{
			case Subject::Buffer:
				return { Access::ComputeShaderRead,
						 Access::ComputeShaderWrite,
						 Access::ComputeShaderReadWrite,
						 Access::ClearUnorderedAccess,
						 Access::VertexShaderRead,
						 Access::VertexShaderWrite,
						 Access::MeshShaderRead,
						 Access::MeshShaderWrite,
						 Access::PixelShaderRead,
						 Access::PixelShaderWrite,
						 Access::GraphicsShaderRead,
						 Access::GraphicsShaderWrite,
						 Access::ConstantBufferRead,
						 Access::IndirectArgumentRead,
						 Access::PredicationRead,
						 Access::CopySource,
						 Access::CopyDest,
						 Access::RaytracingShaderRead,
						 Access::RaytracingShaderWrite,
						 Access::AccelerationStructureBuildInput,
						 Access::AccelerationStructureScratch,
						 Access::AccelerationStructurePostbuildInfoWrite };
			case Subject::ColourTexture:
				return { Access::ComputeShaderRead,
						 Access::ComputeShaderWrite,
						 Access::ComputeShaderReadWrite,
						 Access::ClearUnorderedAccess,
						 Access::VertexShaderRead,
						 Access::VertexShaderWrite,
						 Access::MeshShaderRead,
						 Access::MeshShaderWrite,
						 Access::PixelShaderRead,
						 Access::PixelShaderWrite,
						 Access::GraphicsShaderRead,
						 Access::GraphicsShaderWrite,
						 Access::CopySource,
						 Access::CopyDest,
						 Access::ColourAttachment,
						 Access::RaytracingShaderRead,
						 Access::RaytracingShaderWrite };
			case Subject::DepthTexture:
				return { Access::DepthStencilWrite,
						 Access::DepthStencilRead,
						 Access::PixelShaderRead,
						 Access::ComputeShaderRead,
						 Access::CopySource };
			case Subject::ShadingRateTexture:
				return { Access::ShadingRateSource };
			case Subject::AccelerationStructure:
				return { Access::AccelerationStructureRead,
						 Access::AccelerationStructureBuildRead,
						 Access::AccelerationStructureBuildWrite,
						 Access::AccelerationStructureCopySource,
						 Access::AccelerationStructureCopyDest,
						 Access::AccelerationStructurePostbuildInfoRead };
		}
		return {};
	}

	// The write that gives a subject defined contents, and that each chain transitions back to.
	Access SubjectWrite(Subject subject)
	{
		switch (subject)
		{
			case Subject::DepthTexture:
				return Access::DepthStencilWrite;
			case Subject::AccelerationStructure:
				return Access::AccelerationStructureBuildWrite;
			default:
				return Access::ComputeShaderWrite;
		}
	}

	rg::ResourceId AddSubject(RenderGraph &graph, Subject subject, ResourceHandle tlas)
	{
		TextureDesc desc;
		desc.width = kGrid;
		desc.height = kGrid;
		switch (subject)
		{
			case Subject::Buffer:
				return graph.CreateBuffer(UIntBuffer(kElements, "Buffer"));
			case Subject::ColourTexture:
				desc.debugName = "Colour";
				return graph.CreateTexture(desc);
			case Subject::DepthTexture:
				desc.format = Format::R32_TYPELESS;
				desc.debugName = "Depth";
				return graph.CreateTexture(desc);
			case Subject::ShadingRateTexture:
				desc.format = Format::R8_UINT;
				desc.debugName = "ShadingRate";
				return graph.CreateTexture(desc);
			case Subject::AccelerationStructure:
				return graph.ImportBuffer(tlas, "TLAS");
		}
		return {};
	}

	// Runs the subject's write on the graphics queue, then each access in its own empty pass on the queue.
	// Recording the barriers is enough for the debug layer to check their sync, access and layout.
	void RunChain(GpuTest &gpu, Subject subject, QueueType queue, const std::vector<Access> &accesses, ResourceHandle tlas)
	{
		RenderGraph graph;
		const rg::ResourceId resource = AddSubject(graph, subject, tlas);
		const auto noop = [](PassContext &) {};
		graph.AddPass(Pass::Graphics("Write").Use(resource, SubjectWrite(subject)).Executes(noop));
		if (queue == QueueType::Compute)
		{
			graph.Submit();
		}
		for (const Access access: accesses)
		{
			graph.AddPass(Pass::Compute(rg::ToString(access)).OnQueue(queue).Use(resource, access).Executes(noop));
		}

		std::string chain = std::string(ToString(subject)) + " on the " + rg::ToString(queue) + " queue: write";
		for (const Access access: accesses)
		{
			chain += std::string(", ") + rg::ToString(access);
		}
		const bool compiled = graph.Compile(gpu.rc());
		if (!compiled)
		{
			std::fprintf(stderr, "%s did not compile: %s\n", chain.c_str(), graph.GetError().c_str());
		}
		assert(compiled);

		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame();
		if (gpu.HasDebugLayerErrors())
		{
			std::fprintf(stderr, "invalid barriers for %s\n%s\n", chain.c_str(), graph.Dump().c_str());
			gpu.AssertNoDebugLayerErrors();
		}
	}

	// A one triangle BLAS and a TLAS over it, which the graph imports.
	struct Scene
	{
		ResourceHandle vertices = InvalidResourceHandle;
		ResourceHandle indices = InvalidResourceHandle;
		AccelerationStructureHandle blas = InvalidAccelerationStructureHandle;
		AccelerationStructureHandle tlas = InvalidAccelerationStructureHandle;

		explicit Scene(GpuTest &gpu)
		{
			vertices = gpu.CreateBuffer(9, "Vertices");
			indices = gpu.CreateBuffer(3, "Indices");
			BLASGeometry geometry;
			geometry.vertexBuffer = vertices;
			geometry.vertexCount = 3;
			geometry.vertexStride = 3 * sizeof(float);
			geometry.indexBuffer = indices;
			geometry.indexCount = 3;
			blas = gpu.rc().CreateBLAS(geometry);

			TLASInstance instance;
			instance.blas = blas;
			instance.transform[0][0] = instance.transform[1][1] = instance.transform[2][2] = 1.0f;
			tlas = gpu.rc().CreateTLAS(std::span<const TLASInstance>(&instance, 1));
		}

		void Destroy(GpuTest &gpu)
		{
			gpu.rc().WaitForGPUIdle();
			gpu.rc().DestroyAccelerationStructure(tlas);
			gpu.rc().DestroyAccelerationStructure(blas);
			gpu.rc().DestroyResource(vertices);
			gpu.rc().DestroyResource(indices);
		}
	};

	// Rays through the centres of an 8x8 grid over [0,1]^2 hit the triangle (0,0), (1.01,0), (0,1.01) where x + y <= 7.
	bool RayHits(uint32_t index)
	{
		return index % kGrid + index / kGrid <= kGrid - 1;
	}
} // namespace

// Each access is entered from a write and left for one, on every queue it allows, and every pair of reads
// shares a merged state. The debug layer rejects any sync, access and layout that do not go together.
RG_TEST(EveryAccessIsAValidBarrier)
{
	GpuTest gpu;
	const bool raytracing = SupportsRaytracing(gpu.rc());
	const bool shadingRate = SupportsShadingRateImages(gpu.rc());
	std::optional<Scene> scene;
	if (raytracing)
	{
		scene.emplace(gpu);
	}
	const ResourceHandle tlas = scene ? gpu.rc().GetAccelerationStructureResource(scene->tlas) : InvalidResourceHandle;

	std::set<Access> covered;
	for (const Subject subject: { Subject::Buffer,
								  Subject::ColourTexture,
								  Subject::DepthTexture,
								  Subject::ShadingRateTexture,
								  Subject::AccelerationStructure })
	{
		if ((subject == Subject::AccelerationStructure && !raytracing) ||
			(subject == Subject::ShadingRateTexture && !shadingRate))
		{
			std::printf("  (%s not supported: skipped)\n", ToString(subject));
			continue;
		}

		const std::vector<Access> accesses = SubjectAccesses(subject);
		const Access write = SubjectWrite(subject);
		for (const QueueType queue: { QueueType::Graphics, QueueType::Compute })
		{
			for (const Access access: accesses)
			{
				if (!rg::IsAccessAllowedOnQueue(access, queue))
				{
					continue;
				}
				covered.insert(access);
				std::vector<Access> chain = { access };
				if (rg::IsAccessAllowedOnQueue(write, queue))
				{
					chain.push_back(write);
				}
				RunChain(gpu, subject, queue, chain, tlas);
			}

			for (size_t a = 0; a < accesses.size(); ++a)
			{
				for (size_t b = a + 1; b < accesses.size(); ++b)
				{
					const bool reads = !rg::IsWriteAccess(accesses[a]) && !rg::IsWriteAccess(accesses[b]);
					if (reads && rg::IsAccessAllowedOnQueue(accesses[a], queue) &&
						rg::IsAccessAllowedOnQueue(accesses[b], queue))
					{
						RunChain(gpu, subject, queue, { accesses[a], accesses[b] }, tlas);
					}
				}
			}
		}
	}

	for (uint32_t i = 0; i < rg::kAccessCount; ++i)
	{
		const Access access = static_cast<Access>(i);
		const bool optional = (access == Access::ShadingRateSource && !shadingRate) ||
							  (!raytracing && std::string(rg::ToString(access)).starts_with("AccelerationStructure") &&
							   access != Access::AccelerationStructureBuildInput && access != Access::AccelerationStructureScratch &&
							   access != Access::AccelerationStructurePostbuildInfoWrite);
		if (!covered.contains(access) && !optional)
		{
			std::fprintf(stderr, "%s is not covered\n", rg::ToString(access));
			assert(false && "every access needs a subject");
		}
	}

	if (scene)
	{
		scene->Destroy(gpu);
	}
}

// Compute fills Input with 7 + i and Params with 3 + i. Amplification, mesh and pixel shaders read Input, the mesh
// shader reads Params as a constant buffer, and the mesh and pixel shaders each write a buffer.
RG_TEST(MeshShaderPassOrdersReadsAndWrites)
{
	GpuTest gpu;
	if (!SupportsMeshShaders(gpu.rc()))
	{
		std::printf("  (mesh shaders not supported: skipped)\n");
		return;
	}

	PipelineEntryPoint entries[] = { { "asMain", PipelineShaderStage::Amplification },
									 { "msMain", PipelineShaderStage::Mesh },
									 { "psMain", PipelineShaderStage::Pixel } };
	GraphicsPipelineCreateInfo info;
	info.pipelineType = PipelineType::GRAPHICS_MESH;
	info.rasterizerState.cullMode = D3D12_CULL_MODE_NONE;
	info.depthStencilState.depthEnable = FALSE;
	info.depthStencilState.depthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
	const PipelineHandle pipeline = gpu.rc().CreateGraphicsPipeline(ShaderPath("mesh_access.slang"), entries, info);
	assert(pipeline != InvalidPipelineHandle);

	const ResourceHandle meshOutHandle = gpu.CreateBuffer(kElements, "MeshOut");
	const ResourceHandle pixelOutHandle = gpu.CreateBuffer(kElements, "PixelOut");

	RenderGraph graph;
	const auto input = graph.CreateBuffer(UIntBuffer(kElements, "Input"));
	const auto params = graph.CreateBuffer(UIntBuffer(kElements, "Params")); // 256 bytes, a whole constant buffer
	const auto meshOut = graph.ImportBuffer(meshOutHandle, "MeshOut");
	const auto pixelOut = graph.ImportBuffer(pixelOutHandle, "PixelOut");

	graph.AddPass(Pass::Compute("FillInput").Writes(input).Executes([&](PassContext &ctx) { gpu.Fill(ctx, input, 7, kElements); }));
	graph.AddPass(Pass::Compute("FillParams").Writes(params).Executes([&](PassContext &ctx) { gpu.Fill(ctx, params, 3, kElements); }));
	const auto draw = graph.AddPass(Pass::Graphics("Mesh")
											.Use(input, Access::GraphicsShaderRead)
											.Use(params, Access::ConstantBufferRead)
											.Use(meshOut, Access::MeshShaderWrite)
											.Use(pixelOut, Access::PixelShaderWrite)
											.Executes(
													[&](PassContext &ctx)
													{
														ctx.rc.SetRenderTargets(nullptr, 0);
														ctx.rc.SetViewport(Viewport{ 0, 0, kGrid, kGrid, 0.0f, 1.0f });
														ctx.rc.SetScissorRect(ScissorRect{ 0, 0, kGrid, kGrid });
														ctx.rc.SetGraphicsPipeline(pipeline);
														ctx.rc.SetSRVByName("Input", ctx.Get(input));
														ctx.rc.SetConstantBufferByName("g_Params", ctx.Get(params));
														ctx.rc.SetUAVByName("MeshOut", ctx.Get(meshOut));
														ctx.rc.SetUAVByName("PixelOut", ctx.Get(pixelOut));
														ctx.rc.DispatchMesh(1, 1, 1);
													}));
	assert(graph.Compile(gpu.rc()));

	// Input is read in the amplification, mesh and pixel stages; Params as a constant buffer.
	const rg::CompiledBatch &drawBatch = graph.GetBatch(kSection, kGraphics, graph.GetPassBatch(draw));
	const rg::BarrierInfo *inputBarrier = FindBarrier(drawBatch, input);
	assert(inputBarrier && inputBarrier->after.access == ResourceAccess::ShaderResource);
	assert(inputBarrier->after.sync == (ResourceSync::VertexShading | ResourceSync::PixelShading));
	const rg::BarrierInfo *paramsBarrier = FindBarrier(drawBatch, params);
	assert(paramsBarrier && paramsBarrier->after.access == ResourceAccess::ConstantBuffer);

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	AssertBufferEquals(gpu.Read(meshOutHandle, kElements), [](uint32_t i) { return 7u + i + 7u * 3u; }, "MeshOut", &graph);
	AssertBufferEquals(gpu.Read(pixelOutHandle, kElements), [](uint32_t i) { return (7u + i) * 2u + 1u; }, "PixelOut", &graph);
}

// Compute writes a triangle, a BLAS is built from it and a TLAS over the BLAS, then rays are traced against
// the TLAS with DispatchRays and with an inline ray query. Runs once on each queue.
RG_TEST(RaytracingBuildsAndTracesInOrder)
{
	GpuTest gpu;
	RenderContext &rc = gpu.rc();
	if (!SupportsRaytracing(rc))
	{
		std::printf("  (ray tracing not supported: skipped)\n");
		return;
	}

	const PipelineHandle trianglePipeline = rc.CreateComputePipeline(ShaderPath("triangle.slang"), "main");
	const PipelineHandle queryPipeline = rc.CreateComputePipeline(ShaderPath("rt_query.slang"), "main");
	RaytracingPipelineDesc traceDesc;
	traceDesc.exports = { "rayGen", "miss", "closestHit" };
	traceDesc.hitGroups = { { "HitGroup", "closestHit", "", "" } };
	traceDesc.maxPayloadSizeInBytes = sizeof(uint32_t);
	const PipelineHandle tracePipeline = rc.CreateRaytracingPipeline(ShaderPath("rt_trace.slang"), traceDesc);
	assert(trianglePipeline != InvalidPipelineHandle && queryPipeline != InvalidPipelineHandle);
	assert(tracePipeline != InvalidPipelineHandle);
	const ShaderTableHandle table = rc.CreateShaderTable(tracePipeline);
	rc.AddRayGenShaderRecord(table, "rayGen");
	rc.AddMissShaderRecord(table, "miss");
	rc.AddHitGroupShaderRecord(table, "HitGroup");
	rc.BuildShaderTable(table);

	for (const QueueType queue: { QueueType::Graphics, QueueType::Compute })
	{
		Scene scene(gpu);
		const ResourceHandle traceOutHandle = gpu.CreateBuffer(kElements, "TraceOut");
		const ResourceHandle queryOutHandle = gpu.CreateBuffer(kElements, "QueryOut");

		RenderGraph graph;
		const auto vertices = graph.ImportBuffer(scene.vertices, "Vertices");
		const auto indices = graph.ImportBuffer(scene.indices, "Indices");
		const auto blas = graph.ImportBuffer(rc.GetAccelerationStructureResource(scene.blas), "BLAS");
		const auto tlas = graph.ImportBuffer(rc.GetAccelerationStructureResource(scene.tlas), "TLAS");
		const auto params = graph.CreateBuffer(UIntBuffer(kElements, "Params"));
		const auto traceOut = graph.ImportBuffer(traceOutHandle, "TraceOut");
		const auto queryOut = graph.ImportBuffer(queryOutHandle, "QueryOut");

		graph.AddPass(Pass::Compute("WriteTriangle")
							  .OnQueue(queue)
							  .Use(vertices, Access::ComputeShaderWrite)
							  .Use(indices, Access::ComputeShaderWrite)
							  .Executes(
									  [&](PassContext &ctx)
									  {
										  ctx.rc.SetComputePipeline(trianglePipeline);
										  ctx.rc.SetComputeUAVByName("Vertices", ctx.Get(vertices));
										  ctx.rc.SetComputeUAVByName("Indices", ctx.Get(indices));
										  ctx.rc.Dispatch(1, 1, 1);
									  }));
		graph.AddPass(Pass::Compute("FillParams").OnQueue(queue).Writes(params).Executes(
				[&](PassContext &ctx) { gpu.Fill(ctx, params, 5, kElements); }));
		const auto buildBlas = graph.AddPass(Pass::Compute("BuildBLAS")
													 .OnQueue(queue)
													 .Use(vertices, Access::AccelerationStructureBuildInput)
													 .Use(indices, Access::AccelerationStructureBuildInput)
													 .Use(blas, Access::AccelerationStructureBuildWrite)
													 .Executes([&](PassContext &ctx) { ctx.rc.BuildAccelerationStructure(scene.blas); }));
		const auto buildTlas = graph.AddPass(Pass::Compute("BuildTLAS")
													 .OnQueue(queue)
													 .Use(blas, Access::AccelerationStructureBuildRead)
													 .Use(tlas, Access::AccelerationStructureBuildWrite)
													 .Executes([&](PassContext &ctx) { ctx.rc.BuildAccelerationStructure(scene.tlas); }));
		const auto trace = graph.AddPass(Pass::Compute("Trace")
												 .OnQueue(queue)
												 .Use(tlas, Access::AccelerationStructureRead)
												 .Use(params, Access::RaytracingShaderRead)
												 .Use(traceOut, Access::RaytracingShaderWrite)
												 .Executes(
														 [&](PassContext &ctx)
														 {
															 ctx.rc.SetRaytracingPipeline(tracePipeline);
															 ctx.rc.SetRaytracingSRVByName("Scene", ctx.Get(tlas));
															 ctx.rc.SetRaytracingSRVByName("Params", ctx.Get(params));
															 ctx.rc.SetRaytracingUAVByName("Output", ctx.Get(traceOut));
															 ctx.rc.DispatchRays(table, kGrid, kGrid, 1);
														 }));
		const auto query = graph.AddPass(Pass::Compute("Query")
												 .OnQueue(queue)
												 .Use(tlas, Access::AccelerationStructureRead)
												 .Use(queryOut, Access::ComputeShaderWrite)
												 .Executes(
														 [&](PassContext &ctx)
														 {
															 ctx.rc.SetComputePipeline(queryPipeline);
															 ctx.rc.SetComputeSRVByName("Scene", ctx.Get(tlas));
															 ctx.rc.SetComputeUAVByName("Output", ctx.Get(queryOut));
															 ctx.rc.Dispatch(1, 1, 1);
														 }));
		const bool compiled = graph.Compile(rc);
		if (!compiled)
		{
			std::fprintf(stderr, "%s\n", graph.GetError().c_str());
		}
		assert(compiled);

		// Triangle writes, BLAS build, TLAS build and tracing each wait for the one before.
		assert(graph.GetPassBatch(buildBlas) < graph.GetPassBatch(buildTlas));
		assert(graph.GetPassBatch(buildTlas) < graph.GetPassBatch(trace));
		assert(graph.GetPassBatch(trace) == graph.GetPassBatch(query) && "both only read the TLAS");
		const rg::BarrierInfo *toBuildInput = FindBarrier(graph.GetBatch(kSection, queue, graph.GetPassBatch(buildBlas)), vertices);
		assert(toBuildInput && toBuildInput->before.access == ResourceAccess::UnorderedAccess);
		assert(toBuildInput->after.sync == ResourceSync::BuildRaytracingAccelerationStructure);
		const rg::BarrierInfo *blasToRead = FindBarrier(graph.GetBatch(kSection, queue, graph.GetPassBatch(buildTlas)), blas);
		assert(blasToRead && blasToRead->before.access == ResourceAccess::RaytracingAccelerationStructureWrite);
		assert(blasToRead->after.access == ResourceAccess::RaytracingAccelerationStructureRead);
		const rg::BarrierInfo *tlasToRead = FindBarrier(graph.GetBatch(kSection, queue, graph.GetPassBatch(trace)), tlas);
		assert(tlasToRead && tlasToRead->before.sync == ResourceSync::BuildRaytracingAccelerationStructure);
		assert(tlasToRead->after.sync == (ResourceSync::Raytracing | ResourceSync::AllShading));

		gpu.BeginFrame();
		graph.Execute(rc);
		gpu.EndFrame();

		AssertBufferEquals(gpu.Read(traceOutHandle, kElements), [](uint32_t i) { return RayHits(i) ? 5u : 2u; }, "TraceOut", &graph);
		AssertBufferEquals(gpu.Read(queryOutHandle, kElements), [](uint32_t i) { return RayHits(i) ? 1u : 2u; }, "QueryOut", &graph);

		graph.ReleaseResources();
		scene.Destroy(gpu);
		rc.DestroyResource(traceOutHandle);
		rc.DestroyResource(queryOutHandle);
	}
}
