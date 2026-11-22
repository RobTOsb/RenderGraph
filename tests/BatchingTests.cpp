// Batching, dependency tracking and barrier generation on a single (graphics) queue.

#include "TestCommon.h"

#include <algorithm>

using namespace rgtest;
using rg::Access;
using rg::BarrierKind;
using rg::ExecutionEventType;
using rg::Pass;
using rg::PassContext;
using rg::RenderGraph;

namespace
{
	constexpr uint32_t kCount = 4096;
	constexpr uint32_t kSection = RenderGraph::kFirstUserSection;
	constexpr QueueType kGraphics = QueueType::Graphics;

	uint32_t CountBarriersOn(const rg::CompiledBatch &batch, rg::ResourceId resource)
	{
		uint32_t count = 0;
		for (const rg::BarrierInfo &barrier: batch.barriers)
		{
			count += barrier.resource == resource ? 1 : 0;
		}
		return count;
	}
} // namespace

// fill -> add -> add: each pass depends on the previous one, so three batches with a barrier before each.
RG_TEST(LinearChain)
{
	GpuTest gpu;
	const ResourceHandle outHandle = gpu.CreateBuffer(kCount, "Out");

	RenderGraph graph;
	const auto a = graph.CreateBuffer(UIntBuffer(kCount, "A"));
	const auto b = graph.CreateBuffer(UIntBuffer(kCount, "B"));
	const auto out = graph.ImportBuffer(outHandle, "Out");

	const auto fill = graph.AddPass(
			Pass::Compute("FillA").Writes(a).Executes([&](PassContext &ctx) { gpu.Fill(ctx, a, 100, kCount); }));
	const auto addB = graph.AddPass(
			Pass::Compute("AddB").Reads(a).Writes(b).Executes([&](PassContext &ctx) { gpu.Add(ctx, a, b, 10, kCount); }));
	const auto addOut = graph.AddPass(Pass::Compute("AddOut").Reads(b).Writes(out).Executes(
			[&](PassContext &ctx) { gpu.Add(ctx, b, out, 1, kCount); }));

	const bool compiled = graph.Compile(gpu.rc());
	assert(compiled);

	assert(graph.GetBatchCount(kSection, kGraphics) == 3);
	assert(graph.GetPassBatch(fill) == 0);
	assert(graph.GetPassBatch(addB) == 1);
	assert(graph.GetPassBatch(addOut) == 2);

	// A is born in batch 0 (contents undefined), then turns from UAV write into SRV read before batch 1.
	const rg::BarrierInfo *aInitial = FindBarrier(graph.GetBatch(kSection, kGraphics, 0), a);
	assert(aInitial && aInitial->kind == BarrierKind::Initial);
	assert(aInitial->before.access == ResourceAccess::NoAccess);
	assert(aInitial->after.access == ResourceAccess::UnorderedAccess);

	const rg::BarrierInfo *aToRead = FindBarrier(graph.GetBatch(kSection, kGraphics, 1), a);
	assert(aToRead && aToRead->kind == BarrierKind::Transition);
	assert(aToRead->before.sync == ResourceSync::ComputeShading && aToRead->before.access == ResourceAccess::UnorderedAccess);
	assert(aToRead->after.sync == ResourceSync::ComputeShading && aToRead->after.access == ResourceAccess::ShaderResource);
	assert(aToRead->after.layout == ResourceLayout::Undefined && "buffers have no layout");

	// The imported output transitions from whatever state the context tracks when the graph runs.
	const rg::BarrierInfo *outFirst = FindBarrier(graph.GetBatch(kSection, kGraphics, 2), out);
	assert(outFirst && outFirst->beforeFromTrackedState);

	// One queue: no submits or waits are needed.
	assert(graph.GetQueueWaits().empty());

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	// One barrier call per batch boundary, then the batch's pass.
	const auto &trace = graph.GetExecutionTrace();
	assert(trace.size() == 6);
	assert(trace[0].type == ExecutionEventType::Barriers && trace[1].pass == fill);
	assert(trace[2].type == ExecutionEventType::Barriers && trace[3].pass == addB);
	assert(trace[4].type == ExecutionEventType::Barriers && trace[5].pass == addOut);

	// The context's tracking now holds the graph's last access to the imported buffer.
	const ResourceState outState = gpu.rc().GetResourceState(outHandle);
	assert(outState == rg::GetAccessState(Access::ComputeShaderWrite, false));

	AssertBufferEquals(gpu.Read(outHandle, kCount), [](uint32_t i) { return 100 + i + 10 + 1; }, "Out", &graph);
}

// Two independent producers share batch 0. Their consumer's barriers are merged into one call.
RG_TEST(IndependentPassesShareBatch)
{
	GpuTest gpu;
	const ResourceHandle outHandle = gpu.CreateBuffer(kCount, "Out");

	RenderGraph graph;
	const auto a = graph.CreateBuffer(UIntBuffer(kCount, "A"));
	const auto b = graph.CreateBuffer(UIntBuffer(kCount, "B"));
	const auto out = graph.ImportBuffer(outHandle, "Out");

	const auto fillA = graph.AddPass(
			Pass::Compute("FillA").Writes(a).Executes([&](PassContext &ctx) { gpu.Fill(ctx, a, 100, kCount); }));
	const auto fillB = graph.AddPass(
			Pass::Compute("FillB").Writes(b).Executes([&](PassContext &ctx) { gpu.Fill(ctx, b, 5000, kCount); }));
	const auto combine = graph.AddPass(Pass::Compute("Combine").Reads(a).Reads(b).Writes(out).Executes(
			[&](PassContext &ctx) { gpu.Combine(ctx, a, b, out, kCount); }));

	assert(graph.Compile(gpu.rc()));
	assert(graph.GetBatchCount(kSection, kGraphics) == 2);
	assert(graph.GetPassBatch(fillA) == 0 && graph.GetPassBatch(fillB) == 0);
	assert(graph.GetPassBatch(combine) == 1);
	assert(graph.GetBatch(kSection, kGraphics, 0).passes.size() == 2);
	// A, B and Out all change state before batch 1.
	assert(graph.GetBatch(kSection, kGraphics, 1).barriers.size() == 3);

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	// Exactly two barrier calls: one per batch boundary, however many resources are involved.
	uint32_t barrierCalls = 0;
	for (const rg::ExecutionEvent &event: graph.GetExecutionTrace())
	{
		barrierCalls += event.type == ExecutionEventType::Barriers ? 1 : 0;
	}
	assert(barrierCalls == 2);
	assert(graph.GetExecutionTrace()[0].barrierCount == 2);
	assert(FindEvent(graph,
					 [](const rg::ExecutionEvent &e)
					 { return e.type == ExecutionEventType::Barriers && e.barrierCount == 3; }) >= 0);

	AssertBufferEquals(gpu.Read(outHandle, kCount), [](uint32_t i) { return (100 + i) + (5000 + i); }, "Out", &graph);
}

// Consecutive reads form one access group: all readers share a batch, behind a single barrier whose
// sync and access bits cover every reader.
RG_TEST(ReadGroupMerging)
{
	GpuTest gpu;
	const ResourceHandle aHandle = gpu.CreateBuffer(kCount, "A");
	const ResourceHandle bHandle = gpu.CreateBuffer(kCount, "B");
	const ResourceHandle cHandle = gpu.CreateBuffer(kCount, "C");

	RenderGraph graph;
	const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
	const auto a = graph.ImportBuffer(aHandle, "A");
	const auto b = graph.ImportBuffer(bHandle, "B");
	const auto c = graph.ImportBuffer(cHandle, "C");

	const auto write = graph.AddPass(
			Pass::Compute("WriteX").Writes(x).Executes([&](PassContext &ctx) { gpu.Fill(ctx, x, 100, kCount); }));
	const auto read1 = graph.AddPass(
			Pass::Compute("ReadX1").Reads(x).Writes(a).Executes([&](PassContext &ctx) { gpu.Add(ctx, x, a, 1, kCount); }));
	const auto read2 = graph.AddPass(
			Pass::Compute("ReadX2").Reads(x).Writes(b).Executes([&](PassContext &ctx) { gpu.Add(ctx, x, b, 2, kCount); }));
	const auto read3 = graph.AddPass(Pass::Copy("CopyX").Reads(x).Writes(c).Executes(
			[&](PassContext &ctx) { ctx.rc.CopyBufferRegion(ctx.Get(c), 0, ctx.Get(x), 0, kCount * sizeof(uint32_t)); }));

	assert(graph.Compile(gpu.rc()));
	assert(graph.GetPassBatch(write) == 0);
	assert(graph.GetPassBatch(read1) == 1 && graph.GetPassBatch(read2) == 1 && graph.GetPassBatch(read3) == 1);
	assert(graph.GetBatchCount(kSection, kGraphics) == 2);

	// One barrier for X serves all three readers.
	const rg::CompiledBatch &readers = graph.GetBatch(kSection, kGraphics, 1);
	assert(CountBarriersOn(readers, x) == 1);
	const rg::BarrierInfo *xRead = FindBarrier(readers, x);
	assert(xRead->after.sync == (ResourceSync::ComputeShading | ResourceSync::Copy));
	assert(xRead->after.access == (ResourceAccess::ShaderResource | ResourceAccess::CopySource));

	// Inside the batch, the copy pass is sorted ahead of the compute passes.
	assert(readers.passes.front() == read3);

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	AssertBufferEquals(gpu.Read(aHandle, kCount), [](uint32_t i) { return 100 + i + 1; }, "A", &graph);
	AssertBufferEquals(gpu.Read(bHandle, kCount), [](uint32_t i) { return 100 + i + 2; }, "B", &graph);
	AssertBufferEquals(gpu.Read(cHandle, kCount), [](uint32_t i) { return 100 + i; }, "C", &graph);
}

// A write after a read must wait for the read (write-after-read): overwriting X must not race with the
// pass still reading the old contents.
RG_TEST(WriteAfterReadWaitsForReaders)
{
	GpuTest gpu;
	const ResourceHandle yHandle = gpu.CreateBuffer(kCount, "Y");
	const ResourceHandle zHandle = gpu.CreateBuffer(kCount, "Z");

	RenderGraph graph;
	const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
	const auto y = graph.ImportBuffer(yHandle, "Y");
	const auto z = graph.ImportBuffer(zHandle, "Z");

	const auto write1 = graph.AddPass(
			Pass::Compute("WriteX1").Writes(x).Executes([&](PassContext &ctx) { gpu.Fill(ctx, x, 100, kCount); }));
	const auto read1 = graph.AddPass(Pass::Compute("ReadX1").Reads(x).Writes(y).Executes(
			[&](PassContext &ctx) { gpu.Add(ctx, x, y, 1, kCount, 200); }));
	const auto write2 = graph.AddPass(
			Pass::Compute("WriteX2").Writes(x).Executes([&](PassContext &ctx) { gpu.Fill(ctx, x, 7000, kCount); }));
	const auto read2 = graph.AddPass(
			Pass::Compute("ReadX2").Reads(x).Writes(z).Executes([&](PassContext &ctx) { gpu.Add(ctx, x, z, 2, kCount); }));

	assert(graph.Compile(gpu.rc()));
	assert(graph.GetPassBatch(write1) == 0);
	assert(graph.GetPassBatch(read1) == 1);
	assert(graph.GetPassBatch(write2) == 2);
	assert(graph.GetPassBatch(read2) == 3);

	const rg::BarrierInfo *beforeOverwrite = FindBarrier(graph.GetBatch(kSection, kGraphics, 2), x);
	assert(beforeOverwrite && beforeOverwrite->kind == BarrierKind::Transition);
	assert(beforeOverwrite->before.access == ResourceAccess::ShaderResource);
	assert(beforeOverwrite->after.access == ResourceAccess::UnorderedAccess);

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	AssertBufferEquals(gpu.Read(yHandle, kCount),
					   [](uint32_t i) { return 100 + i + 1 + LcgLowBits(i, 200); },
					   "Y (old contents of X)",
					   &graph);
	AssertBufferEquals(gpu.Read(zHandle, kCount), [](uint32_t i) { return 7000 + i + 2; }, "Z (new contents of X)", &graph);
}

// Back-to-back UAV read-modify-writes are separate groups with a UAV -> UAV barrier between them.
RG_TEST(UavWriteAfterWriteNeedsBarrier)
{
	GpuTest gpu;
	const ResourceHandle outHandle = gpu.CreateBuffer(kCount, "Out");

	RenderGraph graph;
	const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
	const auto out = graph.ImportBuffer(outHandle, "Out");

	graph.AddPass(Pass::Compute("FillX").Writes(x).Executes([&](PassContext &ctx) { gpu.Fill(ctx, x, 10, kCount); }));
	const auto inc1 = graph.AddPass(Pass::Compute("Increment5").Use(x, Access::ComputeShaderReadWrite).Executes(
			[&](PassContext &ctx) { gpu.Increment(ctx, x, 5, kCount); }));
	const auto inc2 = graph.AddPass(Pass::Compute("Increment7").Use(x, Access::ComputeShaderReadWrite).Executes(
			[&](PassContext &ctx) { gpu.Increment(ctx, x, 7, kCount); }));
	graph.AddPass(Pass::Compute("CopyOut").Reads(x).Writes(out).Executes(
			[&](PassContext &ctx) { gpu.Add(ctx, x, out, 0, kCount); }));

	assert(graph.Compile(gpu.rc()));
	assert(graph.GetBatchCount(kSection, kGraphics) == 4);
	assert(graph.GetPassBatch(inc1) == 1 && graph.GetPassBatch(inc2) == 2);

	for (uint32_t batch: { 1u, 2u })
	{
		const rg::BarrierInfo *uav = FindBarrier(graph.GetBatch(kSection, kGraphics, batch), x);
		assert(uav && uav->kind == BarrierKind::Transition);
		assert(uav->before.access == ResourceAccess::UnorderedAccess);
		assert(uav->after.access == ResourceAccess::UnorderedAccess);
		assert(uav->before.sync == ResourceSync::ComputeShading && uav->after.sync == ResourceSync::ComputeShading);
	}

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	AssertBufferEquals(gpu.Read(outHandle, kCount), [](uint32_t i) { return 10 + i + 5 + 7; }, "Out", &graph);
}

// Writes declared concurrent (disjoint halves of one buffer) share a group and a batch, with no barrier
// between them. Without the declaration they serialise.
RG_TEST(ConcurrentWritesShareBatch)
{
	GpuTest gpu;
	const ResourceHandle outHandle = gpu.CreateBuffer(kCount, "Out");
	constexpr uint32_t kHalf = kCount / 2;

	RenderGraph graph;
	const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
	const auto out = graph.ImportBuffer(outHandle, "Out");

	const auto low = graph.AddPass(Pass::Compute("FillLow").UseConcurrent(x, Access::ComputeShaderWrite).Executes(
			[&](PassContext &ctx) { gpu.Fill(ctx, x, 1000, kHalf, 0); }));
	const auto high = graph.AddPass(Pass::Compute("FillHigh").UseConcurrent(x, Access::ComputeShaderWrite).Executes(
			[&](PassContext &ctx) { gpu.Fill(ctx, x, 2000, kHalf, kHalf); }));
	const auto read = graph.AddPass(
			Pass::Compute("Read").Reads(x).Writes(out).Executes([&](PassContext &ctx) { gpu.Add(ctx, x, out, 0, kCount); }));

	assert(graph.Compile(gpu.rc()));
	assert(graph.GetPassBatch(low) == 0 && graph.GetPassBatch(high) == 0);
	assert(graph.GetPassBatch(read) == 1);
	assert(CountBarriersOn(graph.GetBatch(kSection, kGraphics, 0), x) == 1); // only the initial barrier

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	AssertBufferEquals(gpu.Read(outHandle, kCount),
					   [](uint32_t i) { return i < kHalf ? 1000 + i : 2000 + (i - kHalf); },
					   "Out",
					   &graph);

	// The same passes without the concurrent declaration: the second write waits for the first.
	RenderGraph serial;
	const auto sx = serial.CreateBuffer(UIntBuffer(kCount, "X"));
	const auto sLow = serial.AddPass(Pass::Compute("FillLow").Writes(sx));
	const auto sHigh = serial.AddPass(Pass::Compute("FillHigh").Writes(sx));
	assert(serial.Compile(gpu.rc()));
	assert(serial.GetPassBatch(sLow) == 0 && serial.GetPassBatch(sHigh) == 1);
	assert(FindBarrier(serial.GetBatch(kSection, kGraphics, 1), sx)->after.access == ResourceAccess::UnorderedAccess);
}

// Two independent chains, recorded one after the other. Linear batching needs three batches.
// Scheduling each pass as early as possible needs two, the critical path length.
RG_TEST(ReorderingShortensCriticalPath)
{
	GpuTest gpu;
	const ResourceHandle yHandle = gpu.CreateBuffer(kCount, "Y");
	const ResourceHandle wHandle = gpu.CreateBuffer(kCount, "W");

	for (const bool reorder: { false, true })
	{
		RenderGraph graph;
		const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
		const auto z = graph.CreateBuffer(UIntBuffer(kCount, "Z"));
		const auto y = graph.ImportBuffer(yHandle, "Y");
		const auto w = graph.ImportBuffer(wHandle, "W");

		const auto passA = graph.AddPass(
				Pass::Compute("A").Writes(x).Executes([&](PassContext &ctx) { gpu.Fill(ctx, x, 10, kCount); }));
		const auto passB = graph.AddPass(
				Pass::Compute("B").Reads(x).Writes(y).Executes([&](PassContext &ctx) { gpu.Add(ctx, x, y, 1, kCount); }));
		const auto passC = graph.AddPass(
				Pass::Compute("C").Writes(z).Executes([&](PassContext &ctx) { gpu.Fill(ctx, z, 20, kCount); }));
		const auto passD = graph.AddPass(
				Pass::Compute("D").Reads(z).Writes(w).Executes([&](PassContext &ctx) { gpu.Add(ctx, z, w, 2, kCount); }));

		rg::CompileOptions options;
		options.reorderPasses = reorder;
		assert(graph.Compile(gpu.rc(), options));

		assert(graph.GetPassBatch(passA) == 0 && graph.GetPassBatch(passB) == 1);
		if (reorder)
		{
			assert(graph.GetBatchCount(kSection, kGraphics) == 2);
			assert(graph.GetPassBatch(passC) == 0 && graph.GetPassBatch(passD) == 1);
		}
		else
		{
			// C depends on nothing, but linear batching never goes back to an earlier batch.
			assert(graph.GetBatchCount(kSection, kGraphics) == 3);
			assert(graph.GetPassBatch(passC) == 1 && graph.GetPassBatch(passD) == 2);
		}

		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame();

		AssertBufferEquals(gpu.Read(yHandle, kCount), [](uint32_t i) { return 10 + i + 1; }, "Y", &graph);
		AssertBufferEquals(gpu.Read(wHandle, kCount), [](uint32_t i) { return 20 + i + 2; }, "W", &graph);
	}
}

// X is written in batch 0 but first read in batch 2. Its barrier goes right before batch 2, not
// right after the write, so batch 1 runs without waiting on it.
RG_TEST(BarriersPlacedAsLateAsPossible)
{
	GpuTest gpu;
	const ResourceHandle outHandle = gpu.CreateBuffer(kCount, "Out");

	RenderGraph graph;
	const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
	const auto y = graph.CreateBuffer(UIntBuffer(kCount, "Y"));
	const auto z = graph.CreateBuffer(UIntBuffer(kCount, "Z"));
	const auto out = graph.ImportBuffer(outHandle, "Out");

	const auto writeX = graph.AddPass(
			Pass::Compute("WriteX").Writes(x).Executes([&](PassContext &ctx) { gpu.Fill(ctx, x, 100, kCount); }));
	graph.AddPass(Pass::Compute("WriteY").Writes(y).Executes([&](PassContext &ctx) { gpu.Fill(ctx, y, 200, kCount); }));
	graph.AddPass(
			Pass::Compute("YtoZ").Reads(y).Writes(z).Executes([&](PassContext &ctx) { gpu.Add(ctx, y, z, 3, kCount); }));
	const auto combine = graph.AddPass(Pass::Compute("Combine").Reads(x).Reads(z).Writes(out).Executes(
			[&](PassContext &ctx) { gpu.Combine(ctx, x, z, out, kCount); }));

	rg::CompileOptions options;
	options.optimizeLifetimes = false; // keep WriteX in batch 0 (wiggling would move it to batch 1)
	assert(graph.Compile(gpu.rc(), options));
	assert(graph.GetPassBatch(writeX) == 0 && graph.GetPassBatch(combine) == 2);

	assert(FindBarrier(graph.GetBatch(kSection, kGraphics, 1), x) == nullptr);
	const rg::BarrierInfo *xRead = FindBarrier(graph.GetBatch(kSection, kGraphics, 2), x);
	assert(xRead && xRead->kind == BarrierKind::Transition);
	assert(xRead->before.access == ResourceAccess::UnorderedAccess && xRead->after.access == ResourceAccess::ShaderResource);

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	AssertBufferEquals(gpu.Read(outHandle, kCount), [](uint32_t i) { return (100 + i) + (200 + i + 3); }, "Out", &graph);
}

// No layout allows both depth test reads and shader reads, so a depth buffer read both ways gets two
// read groups: DepthStencilRead first, then a transition to ShaderResource before the sampling pass.
RG_TEST(DepthTestReadsAndSamplingUseSeparateLayouts)
{
	GpuTest gpu;
	RenderGraph graph;
	TextureDesc desc;
	desc.width = 64;
	desc.height = 64;
	desc.format = Format::R32_TYPELESS;
	desc.debugName = "Depth";
	const auto depth = graph.CreateTexture(desc);

	graph.AddPass(Pass::Graphics("Prepass").Use(depth, Access::DepthStencilWrite).Executes(
			[&](PassContext &ctx) { ctx.rc.ClearDepthStencil(ctx.GetResource(depth), 0.0f); }));
	const auto depthTest =
			graph.AddPass(Pass::Graphics("DepthTest").Use(depth, Access::DepthStencilRead).Executes([](PassContext &) {}));
	const auto sample = graph.AddPass(Pass::Graphics("Sample").Reads(depth).Executes([](PassContext &) {}));
	assert(graph.Compile(gpu.rc()));

	assert(graph.GetPassBatch(depthTest) < graph.GetPassBatch(sample));
	const rg::BarrierInfo *toDepthRead = FindBarrier(graph.GetBatch(kSection, kGraphics, graph.GetPassBatch(depthTest)), depth);
	assert(toDepthRead && toDepthRead->after.layout == ResourceLayout::DepthStencilRead);
	assert(toDepthRead->after.access == ResourceAccess::DepthStencilRead);
	const rg::BarrierInfo *toSample = FindBarrier(graph.GetBatch(kSection, kGraphics, graph.GetPassBatch(sample)), depth);
	assert(toSample && toSample->kind == BarrierKind::Transition);
	assert(toSample->before.layout == ResourceLayout::DepthStencilRead);
	assert(toSample->after.layout == ResourceLayout::ShaderResource);

	// The debug layer validates every barrier's layout and access when the graph runs.
	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();
}

// Any order is legal inside a batch, so passes are grouped by kind (copy, compute, raster) to avoid
// subchannel switches. With sorting disabled they keep recording order.
RG_TEST(SubchannelSortingInsideBatch)
{
	GpuTest gpu;
	const ResourceHandle srcHandle = gpu.CreateBuffer(kCount, "Src");
	const ResourceHandle dstHandle = gpu.CreateBuffer(kCount, "Dst");

	for (const bool sort: { true, false })
	{
		RenderGraph graph;
		TextureDesc textureDesc;
		textureDesc.width = 64;
		textureDesc.height = 64;
		textureDesc.format = Format::R8G8B8A8_UNORM;
		textureDesc.debugName = "Target";
		textureDesc.clearValue = ClearValue{ { 0.0f, 0.5f, 1.0f, 1.0f } };
		const auto target = graph.CreateTexture(textureDesc);
		const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
		const auto src = graph.ImportBuffer(srcHandle, "Src");
		const auto dst = graph.ImportBuffer(dstHandle, "Dst");

		const auto raster = graph.AddPass(Pass::Graphics("Clear").Writes(target).Executes(
				[&](PassContext &ctx)
				{
					float colour[4] = { 0.0f, 0.5f, 1.0f, 1.0f };
					ctx.rc.ClearRenderTarget(ctx.GetResource(target), colour);
				}));
		const auto compute = graph.AddPass(
				Pass::Compute("Fill").Writes(x).Executes([&](PassContext &ctx) { gpu.Fill(ctx, x, 1, kCount); }));
		const auto copy = graph.AddPass(Pass::Copy("Copy").Reads(src).Writes(dst).Executes(
				[&](PassContext &ctx)
				{ ctx.rc.CopyBufferRegion(ctx.Get(dst), 0, ctx.Get(src), 0, kCount * sizeof(uint32_t)); }));

		rg::CompileOptions options;
		options.sortBatchesByPassKind = sort;
		assert(graph.Compile(gpu.rc(), options));
		assert(graph.GetBatchCount(kSection, kGraphics) == 1);

		const std::vector<rg::PassId> &order = graph.GetBatch(kSection, kGraphics, 0).passes;
		assert(order.size() == 3);
		if (sort)
		{
			assert(order[0] == copy && order[1] == compute && order[2] == raster);
		}
		else
		{
			assert(order[0] == raster && order[1] == compute && order[2] == copy);
		}

		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame();

		// Execution follows the batch order.
		assert(FindPassEvent(graph, order[0]) < FindPassEvent(graph, order[1]));
		assert(FindPassEvent(graph, order[1]) < FindPassEvent(graph, order[2]));
		graph.ReleaseResources();
	}
}

// Compile once, execute every frame. An imported counter is incremented each frame. The graph reads its
// state from the context's tracking every time, so frames chain correctly even with frames in flight.
RG_TEST(StaticGraphExecutesManyFrames)
{
	GpuTest gpu;
	const ResourceHandle counterHandle = gpu.CreateBuffer(kCount, "Counter");

	{
		RenderGraph zero;
		const auto counter = zero.ImportBuffer(counterHandle, "Counter");
		zero.AddPass(Pass::Compute("Zero").Use(counter, Access::ClearUnorderedAccess).Executes(
				[&](PassContext &ctx) { ctx.rc.ClearUAVBufferUint(ctx.Get(counter), 0); }));
		assert(zero.Compile(gpu.rc()));
		gpu.BeginFrame();
		zero.Execute(gpu.rc());
		gpu.EndFrame();
	}

	RenderGraph graph;
	const auto counter = graph.ImportBuffer(counterHandle, "Counter");
	graph.AddPass(Pass::Compute("Increment").Use(counter, Access::ComputeShaderReadWrite).Executes(
			[&](PassContext &ctx) { gpu.Increment(ctx, counter, 3, kCount); }));
	assert(graph.Compile(gpu.rc()));

	constexpr uint32_t kFrames = 10;
	for (uint32_t frame = 0; frame < kFrames; ++frame)
	{
		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame(false); // let frames overlap
	}
	gpu.rc().WaitForGPUIdle();

	assert(gpu.rc().GetResourceState(counterHandle) == rg::GetAccessState(Access::ComputeShaderReadWrite, false));
	AssertBufferEquals(gpu.Read(counterHandle, kCount), [](uint32_t) { return 3 * kFrames; }, "Counter", &graph);
}

// Recorded every frame, but only compiled when the recording changes: a new pass, other compile options, or a
// failed compile. Reused frames still run this frame's callbacks and write this frame's import.
RG_TEST(UnchangedRecordingReusesTheCompile)
{
	GpuTest gpu;
	const ResourceHandle outHandles[2] = { gpu.CreateBuffer(kCount, "OutA"), gpu.CreateBuffer(kCount, "OutB") };
	RenderGraph graph;
	rg::CompileOptions noAlias;
	noAlias.aliasTransientMemory = false;

	struct Frame
	{
		bool extraPass = false; // AddFour between AddOne and AddTwo
		bool noAliasing = false;
		bool valid = true; // without Fill, AddOne reads A before anything writes it
		bool reused = false;
	};
	const Frame frames[] = {
		{ false, false, true, false }, { false, false, true, true }, { false, false, true, true },
		{ true, false, true, false },  { true, false, true, true },	 { false, false, true, false },
		{ false, true, true, false },  { false, true, true, true },	 { false, true, false, false },
		{ false, true, true, false },  { false, true, true, true },
	};

	for (uint32_t frame = 0; frame < std::size(frames); ++frame)
	{
		const Frame &f = frames[frame];
		const ResourceHandle outHandle = outHandles[frame % 2];
		graph.Reset();
		const auto a = graph.CreateBuffer(UIntBuffer(kCount, "A"));
		const auto b = graph.CreateBuffer(UIntBuffer(kCount, "B"));
		const auto c = graph.CreateBuffer(UIntBuffer(kCount, "C"));
		const auto out = graph.ImportBuffer(outHandle, "Out");
		if (f.valid)
		{
			graph.AddPass(Pass::Compute("Fill").Writes(a).Executes([&, frame](PassContext &ctx) { gpu.Fill(ctx, a, frame, kCount); }));
		}
		graph.AddPass(Pass::Compute("AddOne").Reads(a).Writes(b).Executes([&](PassContext &ctx) { gpu.Add(ctx, a, b, 1, kCount); }));
		const auto last = f.extraPass ? c : b;
		if (f.extraPass)
		{
			graph.AddPass(Pass::Compute("AddFour").Reads(b).Writes(c).Executes([&](PassContext &ctx) { gpu.Add(ctx, b, c, 4, kCount); }));
		}
		graph.AddPass(Pass::Compute("AddTwo").Reads(last).Writes(out).Executes(
				[&, last](PassContext &ctx) { gpu.Add(ctx, last, out, 2, kCount); }));

		const bool compiled = graph.Compile(gpu.rc(), f.noAliasing ? noAlias : rg::CompileOptions{});
		assert(compiled == f.valid);
		assert(graph.WasLastCompileReused() == f.reused);
		if (!compiled)
		{
			continue;
		}

		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame();
		const uint32_t added = f.extraPass ? 7 : 3;
		AssertBufferEquals(gpu.Read(outHandle, kCount), [&](uint32_t i) { return frame + i + added; }, "Out", &graph);
	}

	// Without a Reset, a recompile always compiles: the passes may have changed in place.
	assert(graph.Compile(gpu.rc(), noAlias) && !graph.WasLastCompileReused());
	graph.ReleaseResources();
	gpu.rc().DestroyResource(outHandles[0]);
	gpu.rc().DestroyResource(outHandles[1]);
}

// Reset, recorded and compiled every frame, switching between two sizes (Fill writes frame + i, then +1, +2): each size gets its transients
// back, the first big frame grows the heap, and every frame's output is right, overlapping frames included.
RG_TEST(GraphRecompilesEveryFrame)
{
	GpuTest gpu;
	const ResourceHandle outHandle = gpu.CreateBuffer(kCount * 64, "Out");
	RenderGraph graph;
	std::vector<ResourceHandle> smallHandles[2];
	std::vector<ResourceHandle> largeHandles[2];

	constexpr uint32_t kFrames = 16;
	for (uint32_t frame = 0; frame < kFrames; ++frame)
	{
		const bool isLarge = frame % 3 == 2;
		const uint32_t count = isLarge ? kCount * 64 : kCount; // 64 KiB placements either way below that
		graph.Reset();
		const auto a = graph.CreateBuffer(UIntBuffer(count, "A"));
		const auto b = graph.CreateBuffer(UIntBuffer(count, "B"));
		const auto out = graph.ImportBuffer(outHandle, "Out");
		graph.AddPass(Pass::Compute("Fill").Writes(a).Executes([&, frame](PassContext &ctx) { gpu.Fill(ctx, a, frame, count); }));
		graph.AddPass(Pass::Compute("AddOne").Reads(a).Writes(b).Executes([&](PassContext &ctx) { gpu.Add(ctx, a, b, 1, count); }));
		graph.AddPass(Pass::Compute("AddTwo").Reads(b).Writes(out).Executes([&](PassContext &ctx) { gpu.Add(ctx, b, out, 2, count); }));
		assert(graph.Compile(gpu.rc()));

		const bool wait = frame % 4 == 3;
		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame(wait); // the other frames overlap the next one
		(isLarge ? largeHandles : smallHandles)[0].push_back(graph.GetHandle(a));
		(isLarge ? largeHandles : smallHandles)[1].push_back(graph.GetHandle(b));
		if (wait)
		{
			AssertBufferEquals(gpu.Read(outHandle, count), [frame](uint32_t i) { return frame + i + 3; }, "Out", &graph);
		}
	}
	gpu.rc().WaitForGPUIdle();
	AssertBufferEquals(gpu.Read(outHandle, kCount), [](uint32_t i) { return kFrames - 1 + i + 3; }, "Out", &graph);

	// Small frames 0 and 1 come before the heap grows, then every later smallHandles frame shares one set.
	for (uint32_t r = 0; r < 2; ++r)
	{
		assert(smallHandles[r][0] == smallHandles[r][1]);
		assert(smallHandles[r][2] != smallHandles[r][0] && "the heap grew, so new resources");
		assert(std::all_of(smallHandles[r].begin() + 2, smallHandles[r].end(), [&](ResourceHandle h) { return h == smallHandles[r][2]; }));
		assert(std::all_of(largeHandles[r].begin(), largeHandles[r].end(), [&](ResourceHandle h) { return h == largeHandles[r][0]; }));
	}
}
