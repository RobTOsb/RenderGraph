// Mips: graph.Mip(texture, i) is a resource of its own, so bloom style chains get a barrier per mip.

#include <string>
#include <vector>

#include "TestCommon.h"

using namespace rgtest;
using rg::Access;
using rg::BarrierKind;
using rg::Pass;
using rg::PassContext;
using rg::RenderGraph;
using rg::ResourceId;

namespace
{
	constexpr uint32_t kSize = 16;
	constexpr uint32_t kMips = 5; // 16, 8, 4, 2, 1
	constexpr uint32_t kSeed = 1;
	constexpr QueueType kGraphics = QueueType::Graphics;
	constexpr QueueType kCompute = QueueType::Compute;

	uint32_t MipSize(uint32_t mip)
	{
		return kSize >> mip;
	}

	// Where each mip starts in the readback buffer, mips laid end to end.
	uint32_t MipOffset(uint32_t mip)
	{
		uint32_t offset = 0;
		for (uint32_t m = 0; m < mip; ++m)
		{
			offset += MipSize(m) * MipSize(m);
		}
		return offset;
	}

	const uint32_t kTexels = MipOffset(kMips);

	uint32_t MipOfTexel(uint32_t index)
	{
		uint32_t mip = 0;
		while (index >= MipOffset(mip + 1))
		{
			++mip;
		}
		return mip;
	}

	// Down sums 2x2 texels, so mip m of a uniform seed holds 4^m times it.
	uint32_t Downsampled(uint32_t mip)
	{
		return kSeed << (2 * mip);
	}

	// Up then adds the coarser mip to each finer one, so every mip holds the sum of itself and all coarser mips.
	uint32_t Upsampled(uint32_t mip)
	{
		uint32_t sum = 0;
		for (uint32_t m = mip; m < kMips; ++m)
		{
			sum += Downsampled(m);
		}
		return sum;
	}

	struct Constants
	{
		uint32_t value;
		uint32_t width;
		uint32_t height;
		uint32_t offset;
	};

	class MipShaders
	{
	public:
		explicit MipShaders(RenderContext &rc) :
			seed_(Load(rc, "seed")), down_(Load(rc, "down")), up_(Load(rc, "up")), readback_(Load(rc, "readback"))
		{
		}

		void Seed(PassContext &ctx, ResourceId mip, uint32_t size) const
		{
			Dispatch(ctx, seed_, { kSeed, size, size, 0 }, {}, mip);
		}
		void Down(PassContext &ctx, ResourceId finer, ResourceId mip, uint32_t size) const
		{
			Dispatch(ctx, down_, { 0, size, size, 0 }, finer, mip);
		}
		void Up(PassContext &ctx, ResourceId coarser, ResourceId mip, uint32_t size) const
		{
			Dispatch(ctx, up_, { 0, size, size, 0 }, coarser, mip);
		}
		// Every mip of the texture into the buffer, at MipOffset.
		void Readback(PassContext &ctx, const std::vector<ResourceId> &mips, ResourceId buffer) const
		{
			ctx.rc.SetComputePipeline(readback_);
			ctx.rc.SetComputeUAVByName("Buffer", ctx.Get(buffer));
			for (uint32_t m = 0; m < mips.size(); ++m)
			{
				const Constants constants{ 0, MipSize(m), MipSize(m), MipOffset(m) };
				ctx.rc.SetComputeSRVByName("Input", ctx.Get(mips[m]));
				ctx.rc.SetCompute32BitConstants(&constants, 4);
				ctx.rc.Dispatch((MipSize(m) + 7) / 8, (MipSize(m) + 7) / 8, 1);
			}
		}

	private:
		static PipelineHandle Load(RenderContext &rc, const char *entry)
		{
			const PipelineHandle handle = rc.CreateComputePipeline(std::string(RG_TEST_SHADER_DIR) + "/mips.slang", entry);
			assert(handle != InvalidPipelineHandle);
			return handle;
		}

		static void Dispatch(PassContext &ctx, PipelineHandle pipeline, Constants constants, ResourceId input, ResourceId output)
		{
			ctx.rc.SetComputePipeline(pipeline);
			if (input.IsValid())
			{
				ctx.rc.SetComputeSRVByName("Input", ctx.Get(input));
			}
			ctx.rc.SetComputeUAVByName("Output", ctx.Get(output));
			ctx.rc.SetCompute32BitConstants(&constants, 4);
			ctx.rc.Dispatch((constants.width + 7) / 8, (constants.height + 7) / 8, 1);
		}

		PipelineHandle seed_;
		PipelineHandle down_;
		PipelineHandle up_;
		PipelineHandle readback_;
	};

	TextureDesc MipTextureDesc(const char *name)
	{
		TextureDesc desc;
		desc.width = desc.height = kSize;
		desc.mipLevels = kMips;
		desc.format = Format::R32_UINT;
		desc.debugName = name;
		return desc;
	}

	std::vector<ResourceId> AllMips(RenderGraph &graph, ResourceId texture, uint32_t count = kMips)
	{
		std::vector<ResourceId> mips;
		for (uint32_t m = 0; m < count; ++m)
		{
			mips.push_back(graph.Mip(texture, m));
		}
		return mips;
	}

	void AssertTexels(GpuTest &gpu, ResourceHandle buffer, uint32_t (*expected)(uint32_t), const RenderGraph &graph)
	{
		AssertBufferEquals(gpu.Read(buffer, kTexels), [&](uint32_t i) { return expected(MipOfTexel(i)); }, "Texels", &graph);
	}
} // namespace

// Downsample then upsample through every mip. Each pass reads one mip and writes the next, so each
// barrier moves just those two mips; the texture as a whole never gets one.
RG_TEST(BloomChainTransitionsEachMip)
{
	GpuTest gpu;
	const MipShaders shaders(gpu.rc());

	// Synchronous, then with the downsample chain on the async compute queue.
	for (const bool async: { false, true })
	{
		const ResourceHandle outHandle = gpu.CreateBuffer(kTexels, "Texels");
		RenderGraph graph;
		const auto bloom = graph.CreateTexture(MipTextureDesc("Bloom"));
		const auto out = graph.ImportBuffer(outHandle, "Texels");
		const std::vector<ResourceId> mip = AllMips(graph, bloom);

		graph.AddPass(Pass::Compute("Seed").Writes(mip[0]).Executes([&](PassContext &ctx) { shaders.Seed(ctx, mip[0], kSize); }));
		if (async)
		{
			graph.Submit();
		}
		std::vector<rg::PassId> down(kMips);
		for (uint32_t m = 1; m < kMips; ++m)
		{
			down[m] = graph.AddPass(Pass::Compute("Down" + std::to_string(m))
											.OnQueue(async ? kCompute : kGraphics)
											.Reads(mip[m - 1])
											.Writes(mip[m])
											.Executes([&, m](PassContext &ctx) { shaders.Down(ctx, mip[m - 1], mip[m], MipSize(m)); }));
		}
		if (async)
		{
			graph.Submit();
		}
		for (uint32_t m = kMips - 1; m-- > 0;)
		{
			graph.AddPass(Pass::Compute("Up" + std::to_string(m))
								  .Reads(mip[m + 1])
								  .Use(mip[m], Access::ComputeShaderReadWrite)
								  .Executes([&, m](PassContext &ctx) { shaders.Up(ctx, mip[m + 1], mip[m], MipSize(m)); }));
		}
		// Reading the whole texture reads every mip.
		const auto readback = graph.AddPass(
				Pass::Compute("Readback").Reads(bloom).Writes(out).Executes([&](PassContext &ctx) { shaders.Readback(ctx, mip, out); }));

		const bool compiled = graph.Compile(gpu.rc());
		if (!compiled)
		{
			std::fprintf(stderr, "%s\n", graph.GetError().c_str());
		}
		assert(compiled);

		const uint32_t downSection = RenderGraph::kFirstUserSection + (async ? 1 : 0);
		const QueueType downQueue = async ? kCompute : kGraphics;
		for (uint32_t m = 1; m < kMips; ++m)
		{
			// Each down pass waits for the one before it, and moves only its two mips.
			const rg::CompiledBatch &batch = graph.GetBatch(downSection, downQueue, graph.GetPassBatch(down[m]));
			assert(batch.passes.size() == 1);
			const rg::BarrierInfo *target = FindBarrier(batch, mip[m]);
			assert(target && target->kind == BarrierKind::Initial);
			assert(target->after.layout == ResourceLayout::UnorderedAccess);
			const rg::BarrierInfo *source = FindBarrier(batch, mip[m - 1]);
			assert(source && source->after.layout == ResourceLayout::ShaderResource);
			if (async && m == 1)
			{
				assert(source->kind == BarrierKind::QueueAcquire);
			}
			else
			{
				assert(source->kind == BarrierKind::Transition && source->before.layout == ResourceLayout::UnorderedAccess);
			}
			assert(!FindBarrier(batch, bloom));
		}

		// Only mip 0 moves for the readback: each coarser mip is already a shader resource for the Up pass above it.
		const uint32_t lastSection = RenderGraph::kFirstUserSection + (async ? 2 : 0);
		const rg::CompiledBatch &readbackBatch = graph.GetBatch(lastSection, kGraphics, graph.GetPassBatch(readback));
		const rg::BarrierInfo *mip0 = FindBarrier(readbackBatch, mip[0]);
		assert(mip0 && mip0->before.layout == ResourceLayout::UnorderedAccess);
		assert(mip0->after.layout == ResourceLayout::ShaderResource);
		for (uint32_t m = 1; m < kMips; ++m)
		{
			assert(!FindBarrier(readbackBatch, mip[m]));
		}

		if (async)
		{
			// Every mip crosses queues on its own: mip 0 to compute and back, the others back to graphics.
			assert(FindRelease(graph, RenderGraph::kFirstUserSection, kGraphics, mip[0]));
			for (uint32_t m = 0; m < kMips; ++m)
			{
				assert(FindRelease(graph, downSection, kCompute, mip[m]));
			}
		}

		// Mips share their texture's memory.
		const auto texture = graph.GetTransientAllocation(bloom);
		assert(texture.has_value());
		for (const ResourceId m: mip)
		{
			const auto allocation = graph.GetTransientAllocation(m);
			assert(allocation && allocation->offset == texture->offset && allocation->size == texture->size);
		}

		// Twice: the second run starts from where the first left the memory.
		for (int frame = 0; frame < 2; ++frame)
		{
			gpu.BeginFrame();
			graph.Execute(gpu.rc());
			gpu.EndFrame();
			AssertTexels(gpu, outHandle, Upsampled, graph);
		}
		gpu.rc().WaitForGPUIdle();
		graph.ReleaseResources();
		gpu.rc().DestroyResource(outHandle);
	}
	gpu.AssertNoDebugLayerErrors();
}

// The same downsample with raster passes: each draws into one mip through a render target view of that mip.
RG_TEST(RasterPassesDrawIntoMips)
{
	GpuTest gpu;
	const MipShaders shaders(gpu.rc());

	PipelineEntryPoint entries[] = { { "vsFullscreen", PipelineShaderStage::Vertex }, { "psDown", PipelineShaderStage::Pixel } };
	GraphicsPipelineCreateInfo info;
	info.rasterizerState.cullMode = CullMode::None;
	info.depthStencilState.depthEnable = false;
	info.depthStencilState.depthWrite = false;
	info.renderTargetFormats = { Format::R32_UINT };
	const PipelineHandle pipeline =
			gpu.rc().CreateGraphicsPipeline(std::string(RG_TEST_SHADER_DIR) + "/mips.slang", entries, info);
	assert(pipeline != InvalidPipelineHandle);

	const ResourceHandle outHandle = gpu.CreateBuffer(kTexels, "Texels");
	RenderGraph graph;
	const auto bloom = graph.CreateTexture(MipTextureDesc("Bloom"));
	const auto out = graph.ImportBuffer(outHandle, "Texels");
	const std::vector<ResourceId> mip = AllMips(graph, bloom);

	graph.AddPass(Pass::Compute("Seed").Writes(mip[0]).Executes([&](PassContext &ctx) { shaders.Seed(ctx, mip[0], kSize); }));
	for (uint32_t m = 1; m < kMips; ++m)
	{
		graph.AddPass(Pass::Graphics("Down" + std::to_string(m))
							  .Reads(mip[m - 1])
							  .Writes(mip[m])
							  .Executes(
									  [&, m](PassContext &ctx)
									  {
										  const Resource target = ctx.GetResource(mip[m]);
										  assert(target.mip == m);
										  float clear[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; // the texture's clear value
										  ctx.rc.ClearRenderTarget(target, clear);
										  ctx.rc.SetRenderTargets(&target, 1);
										  ctx.rc.SetViewport(Viewport{ 0, 0, MipSize(m), MipSize(m), 0.0f, 1.0f });
										  ctx.rc.SetScissorRect(ScissorRect{ 0, 0, MipSize(m), MipSize(m) });
										  ctx.rc.SetGraphicsPipeline(pipeline);
										  ctx.rc.SetPrimitiveTopology();
										  ctx.rc.SetSRVByName("Input", ctx.Get(mip[m - 1]));
										  ctx.rc.Draw(3);
									  }));
	}
	graph.AddPass(Pass::Compute("Readback").Reads(bloom).Writes(out).Executes([&](PassContext &ctx) { shaders.Readback(ctx, mip, out); }));
	assert(graph.Compile(gpu.rc()));

	// Render target mips start as a discard from undefined, then are sampled by the next draw.
	for (uint32_t m = 1; m < kMips; ++m)
	{
		const rg::CompiledBatch &batch = graph.GetBatch(RenderGraph::kFirstUserSection, kGraphics, m);
		const rg::BarrierInfo *target = FindBarrier(batch, mip[m]);
		assert(target && target->kind == BarrierKind::Initial && target->after.layout == ResourceLayout::RenderTarget);
		const rg::BarrierInfo *source = FindBarrier(batch, mip[m - 1]);
		assert(source && source->after.layout == ResourceLayout::ShaderResource);
		assert(source->after.access == ResourceAccess::ShaderResource);
	}

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();
	AssertTexels(gpu, outHandle, Downsampled, graph);
}

// An imported texture used by mip ends with every mip in one state, the one state the context tracks,
// so the next graph (or frame) starts from the right place.
RG_TEST(ImportedTextureMipsEndInOneState)
{
	GpuTest gpu;
	const MipShaders shaders(gpu.rc());

	TextureDesc desc = MipTextureDesc("Persistent");
	desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	const ResourceHandle textureHandle = gpu.rc().CreateTexture(desc, ResourceViewType::ShaderResource);
	const ResourceHandle outHandle = gpu.CreateBuffer(kTexels, "Texels");

	// The first graph builds only the first three mips, so the last two keep a state known only at Execute.
	RenderGraph build;
	std::vector<ResourceId> buildMips;
	{
		const auto texture = build.ImportTexture(textureHandle, "Persistent");
		buildMips = AllMips(build, texture, 3);
		const std::vector<ResourceId> &mip = buildMips;
		build.AddPass(Pass::Compute("Seed").Writes(mip[0]).Executes([&](PassContext &ctx) { shaders.Seed(ctx, mip[0], kSize); }));
		for (uint32_t m = 1; m < 3; ++m)
		{
			build.AddPass(Pass::Compute("Down" + std::to_string(m))
								  .Reads(mip[m - 1])
								  .Writes(mip[m])
								  .Executes([&, m](PassContext &ctx) { shaders.Down(ctx, mip[m - 1], mip[m], MipSize(m)); }));
		}
		assert(build.Compile(gpu.rc()));

		// Mixed and unknown end states all go to COMMON at the end, the unused mips from their tracked state.
		const uint32_t epilogue = build.GetEpilogueSection();
		const std::vector<ResourceId> all = AllMips(build, texture);
		for (uint32_t m = 0; m < kMips; ++m)
		{
			const rg::BarrierInfo *unify = FindRelease(build, epilogue, kGraphics, all[m]);
			assert(unify && unify->after.layout == ResourceLayout::Common);
			assert(unify->beforeFromTrackedState == (m >= 3));
		}
	}

	// The second graph finishes the chain and reads everything back, leaving every mip a shader resource.
	RenderGraph finish;
	std::vector<ResourceId> finishMips;
	{
		const auto texture = finish.ImportTexture(textureHandle, "Persistent");
		const auto out = finish.ImportBuffer(outHandle, "Texels");
		finishMips = AllMips(finish, texture);
		const std::vector<ResourceId> &mip = finishMips;
		for (uint32_t m = 3; m < kMips; ++m)
		{
			finish.AddPass(Pass::Compute("Down" + std::to_string(m))
								   .Reads(mip[m - 1])
								   .Writes(mip[m])
								   .Executes([&, m](PassContext &ctx) { shaders.Down(ctx, mip[m - 1], mip[m], MipSize(m)); }));
		}
		finish.AddPass(Pass::Compute("Readback").Reads(texture).Writes(out).Executes([&, out](PassContext &ctx) { shaders.Readback(ctx, mip, out); }));
		assert(finish.Compile(gpu.rc()));
		// Every mip ends in the same state, so nothing needs unifying.
		assert(finish.GetQueueReleases(finish.GetEpilogueSection(), kGraphics).empty());
	}

	// The third samples mip 0 in a pixel shader and the rest in compute: one layout, different sync scopes.
	RenderGraph sample;
	{
		const auto texture = sample.ImportTexture(textureHandle, "Persistent");
		const std::vector<ResourceId> mip = AllMips(sample, texture);
		sample.AddPass(Pass::Graphics("SampleMip0").Reads(mip[0]));
		sample.AddPass(Pass::Compute("SampleRest").Reads(mip[1]).Reads(mip[2]).Reads(mip[3]).Reads(mip[4]));
		assert(sample.Compile(gpu.rc()));
		for (uint32_t m = 0; m < kMips; ++m)
		{
			const rg::BarrierInfo *unify = FindRelease(sample, sample.GetEpilogueSection(), kGraphics, mip[m]);
			assert(unify && unify->after.layout == ResourceLayout::ShaderResource);
			assert(unify->after.sync == (ResourceSync::PixelShading | ResourceSync::ComputeShading));
		}
	}

	for (int frame = 0; frame < 2; ++frame)
	{
		gpu.BeginFrame();
		build.Execute(gpu.rc());
		assert(gpu.rc().GetResourceState(textureHandle).layout == ResourceLayout::Common);
		finish.Execute(gpu.rc());
		assert(gpu.rc().GetResourceState(textureHandle).layout == ResourceLayout::ShaderResource);
		sample.Execute(gpu.rc());
		assert(gpu.rc().GetResourceState(textureHandle).layout == ResourceLayout::ShaderResource);
		gpu.EndFrame();
		AssertTexels(gpu, outHandle, Downsampled, finish);
	}
}

RG_TEST(InvalidMipUseIsRejected)
{
	GpuTest gpu;

	{
		RenderGraph graph;
		const auto bloom = graph.CreateTexture(MipTextureDesc("Bloom"));
		graph.AddPass(Pass::Compute("TooFar").Writes(graph.Mip(bloom, kMips)));
		assert(!graph.Compile(gpu.rc()));
		assert(graph.GetError().find("has 5 mips, so it has no mip 5") != std::string::npos);
	}
	{
		// The whole texture already covers the mip.
		RenderGraph graph;
		const auto bloom = graph.CreateTexture(MipTextureDesc("Bloom"));
		graph.AddPass(Pass::Compute("Seed").Writes(bloom));
		graph.AddPass(Pass::Compute("Both").Reads(bloom).Writes(graph.Mip(bloom, 1)));
		assert(!graph.Compile(gpu.rc()));
		assert(graph.GetError().find("not both") != std::string::npos);
	}
	{
		// Mips are separate resources, but the texture is still on one queue at a time.
		RenderGraph graph;
		const auto bloom = graph.CreateTexture(MipTextureDesc("Bloom"));
		graph.AddPass(Pass::Compute("Mip0").Writes(graph.Mip(bloom, 0)));
		graph.AddPass(Pass::Compute("Mip1").OnQueue(kCompute).Writes(graph.Mip(bloom, 1)));
		assert(!graph.Compile(gpu.rc()));
		assert(graph.GetError().find("both the graphics and compute queues") != std::string::npos);
	}
	{
		// Each mip is written before it is read, even when the read comes through the whole texture.
		RenderGraph graph;
		const auto bloom = graph.CreateTexture(MipTextureDesc("Bloom"));
		const auto out = graph.CreateBuffer(UIntBuffer(kTexels, "Out"));
		graph.AddPass(Pass::Compute("Mip0").Writes(graph.Mip(bloom, 0)));
		graph.AddPass(Pass::Compute("ReadAll").Reads(bloom).Writes(out));
		assert(!graph.Compile(gpu.rc()));
		assert(graph.GetError().find("'Bloom[mip 1]' before anything writes it") != std::string::npos);
	}
}
