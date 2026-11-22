// Async compute: passes on the compute queue, submit sections, queue transfers and fence waits.

#include <cstdlib>

#include "TestCommon.h"

using namespace rgtest;
using rg::Access;
using rg::BarrierKind;
using rg::ExecutionEvent;
using rg::ExecutionEventType;
using rg::Pass;
using rg::PassContext;
using rg::RenderGraph;

namespace
{
	constexpr uint32_t kCount = 16384;
	constexpr QueueType kGraphics = QueueType::Graphics;
	constexpr QueueType kCompute = QueueType::Compute;

	bool IsWait(const ExecutionEvent &e, QueueType queue, QueueType on, uint32_t onSection)
	{
		return e.type == ExecutionEventType::Wait && e.queue == queue && e.otherQueue == on && e.otherSection == onSection;
	}

	bool IsSubmit(const ExecutionEvent &e, QueueType queue, uint32_t section)
	{
		return e.type == ExecutionEventType::Submit && e.queue == queue && e.section == section;
	}

	// Busy loop iterations that keep a pass running far longer than the next submit takes.
	// WARP gets fewer; RG_TEST_SLOW_ITERATIONS overrides both.
	uint32_t SlowIterations()
	{
		const std::string iterations = ReadEnv("RG_TEST_SLOW_ITERATIONS");
		if (!iterations.empty())
		{
			return static_cast<uint32_t>(std::strtoul(iterations.c_str(), nullptr, 10));
		}
		return UseWarp() ? 200u : 2000u;
	}

	void ZeroBuffer(GpuTest &gpu, ResourceHandle handle)
	{
		RenderGraph zero;
		const auto buffer = zero.ImportBuffer(handle, "Zero");
		zero.AddPass(Pass::Compute("Zero").Use(buffer, Access::ClearUnorderedAccess).Executes(
				[&](PassContext &ctx) { ctx.rc.ClearUAVBufferUint(ctx.Get(buffer), 0); }));
		assert(zero.Compile(gpu.rc()));
		gpu.BeginFrame();
		zero.Execute(gpu.rc());
		gpu.EndFrame();
	}
} // namespace

// Graphics SlowA makes A, compute AsyncProcess turns A into B while graphics runs FillC, then Combine.
// Exactly two waits are needed, and the inputs change every frame so an early read gives wrong data.
RG_TEST(AsyncComputeOverlapsGraphics)
{
	GpuTest gpu;
	const ResourceHandle dHandle = gpu.CreateBuffer(kCount, "D");
	const uint32_t slow = SlowIterations();

	RenderGraph graph;
	const auto a0 = graph.CreateBuffer(UIntBuffer(kCount, "A0"));
	const auto a = graph.CreateBuffer(UIntBuffer(kCount, "A"));
	const auto b = graph.CreateBuffer(UIntBuffer(kCount, "B"));
	const auto c = graph.CreateBuffer(UIntBuffer(kCount, "C"));
	const auto d = graph.ImportBuffer(dHandle, "D");

	uint32_t seed = 0; // read when each pass records, so one compiled graph produces different data per frame
	graph.AddPass(Pass::Compute("FillA0").Writes(a0).Executes(
			[&](PassContext &ctx) { gpu.Fill(ctx, a0, 100 + seed, kCount); }));
	const auto slowA = graph.AddPass(Pass::Compute("SlowA").Reads(a0).Writes(a).Executes(
			[&](PassContext &ctx) { gpu.Add(ctx, a0, a, 1, kCount, slow); }));
	graph.Submit();
	const auto async = graph.AddPass(Pass::Compute("AsyncProcess").OnQueue(kCompute).Reads(a).Writes(b).Executes(
			[&](PassContext &ctx)
			{
				assert(ctx.GetQueue() == kCompute && ctx.rc.GetRecordingQueue() == kCompute);
				gpu.Add(ctx, a, b, 10, kCount, slow);
			}));
	const auto fillC = graph.AddPass(
			Pass::Compute("FillC").Writes(c).Executes([&](PassContext &ctx) { gpu.Fill(ctx, c, 5000 + 3 * seed, kCount); }));
	graph.Submit();
	const auto combine = graph.AddPass(Pass::Compute("Combine").Reads(b).Reads(c).Writes(d).Executes(
			[&](PassContext &ctx)
			{
				assert(ctx.rc.GetRecordingQueue() == kGraphics);
				gpu.Combine(ctx, b, c, d, kCount);
			}));

	assert(graph.Compile(gpu.rc()));

	const uint32_t s1 = RenderGraph::kFirstUserSection;
	const uint32_t s2 = s1 + 1;
	const uint32_t s3 = s1 + 2;
	assert(graph.GetPassSection(slowA) == s1 && graph.GetPassSection(async) == s2);
	assert(graph.GetPassSection(fillC) == s2 && graph.GetPassSection(combine) == s3);
	assert(graph.GetBatchCount(s1, kGraphics) == 2);
	assert(graph.GetBatchCount(s2, kCompute) == 1 && graph.GetBatchCount(s2, kGraphics) == 1);

	// Exactly the two waits the data flow needs.
	const auto &waits = graph.GetQueueWaits();
	assert(waits.size() == 2);
	assert(waits[0].waitingQueue == kCompute && waits[0].waitingSection == s2);
	assert(waits[0].signalQueue == kGraphics && waits[0].signalSection == s1);
	assert(waits[1].waitingQueue == kGraphics && waits[1].waitingSection == s3);
	assert(waits[1].signalQueue == kCompute && waits[1].signalSection == s2);
	assert(graph.IsSubmitSignalled(s1, kGraphics));
	assert(graph.IsSubmitSignalled(s2, kCompute));
	assert(!graph.IsSubmitSignalled(s2, kGraphics)); // nobody waits for FillC's section

	// Buffers cross queues with no release, since the submit boundary syncs, and an acquire from nothing.
	assert(FindRelease(graph, s1, kGraphics, a) == nullptr);
	const rg::BarrierInfo *acquireA = FindBarrier(graph.GetBatch(s2, kCompute, 0), a);
	assert(acquireA && acquireA->kind == BarrierKind::QueueAcquire);
	assert(acquireA->before.sync == ResourceSync::None && acquireA->before.access == ResourceAccess::NoAccess);
	assert(acquireA->after.access == ResourceAccess::ShaderResource);

	// B is acquired by graphics. C stays on graphics and gets an ordinary transition.
	assert(FindRelease(graph, s2, kCompute, b) == nullptr);
	assert(FindBarrier(graph.GetBatch(s3, kGraphics, 0), b)->kind == BarrierKind::QueueAcquire);
	assert(FindRelease(graph, s2, kGraphics, c) == nullptr);
	assert(FindBarrier(graph.GetBatch(s3, kGraphics, 0), c)->kind == BarrierKind::Transition);

	// B and C are live in the same section on different queues, which do not order against each other,
	// so they must not share memory.
	const auto bAlloc = graph.GetTransientAllocation(b);
	const auto cAlloc = graph.GetTransientAllocation(c);
	assert(bAlloc->offset + bAlloc->size <= cAlloc->offset || cAlloc->offset + cAlloc->size <= bAlloc->offset);

	auto expected = [slow, &seed](uint32_t i)
	{ return (100 + seed + i + 1 + LcgLowBits(i, slow)) + 10 + LcgLowBits(i, slow) + (5000 + 3 * seed + i); };

	// Several frames, reading back after each one.
	for (uint32_t frame = 0; frame < 4; ++frame)
	{
		seed = 1000 * (frame + 1);
		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame();

		if (frame == 0)
		{
			// Recorded order: graphics submits section 1, then compute waits for it and runs AsyncProcess,
			// then compute submits, then graphics waits for it before Combine.
			const int submitGraphics1 = FindEvent(graph, [&](const ExecutionEvent &e) { return IsSubmit(e, kGraphics, s1); });
			const int computeWait = FindEvent(graph, [&](const ExecutionEvent &e) { return IsWait(e, kCompute, kGraphics, s1); });
			const int asyncPass = FindPassEvent(graph, async);
			const int submitCompute2 = FindEvent(graph, [&](const ExecutionEvent &e) { return IsSubmit(e, kCompute, s2); });
			const int graphicsWait = FindEvent(graph, [&](const ExecutionEvent &e) { return IsWait(e, kGraphics, kCompute, s2); });
			const int combinePass = FindPassEvent(graph, combine);
			assert(submitGraphics1 >= 0);
			assert(submitGraphics1 < computeWait && computeWait < asyncPass);
			assert(asyncPass < submitCompute2 && submitCompute2 < graphicsWait && graphicsWait < combinePass);

			// FillC is recorded with no wait on the graphics queue in its section.
			assert(FindEvent(graph,
							 [&](const ExecutionEvent &e)
							 { return e.type == ExecutionEventType::Wait && e.queue == kGraphics && e.section == s2; }) == -1);
			assert(FindPassEvent(graph, fillC) >= 0);
		}

		AssertBufferEquals(gpu.Read(dHandle, kCount), expected, "D", &graph);
	}

	// Then a burst of overlapping frames: the frame-start waits keep transient memory safe.
	for (uint32_t frame = 0; frame < 6; ++frame)
	{
		seed = 100000 * (frame + 1);
		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame(false);
	}
	gpu.rc().WaitForGPUIdle();
	AssertBufferEquals(gpu.Read(dHandle, kCount), expected, "D after burst", &graph);
}

// Inside one section a resource lives on one queue. Using it from both is rejected; a Submit() between
// the two passes fixes it.
RG_TEST(AsyncComputeQueueConflictRejected)
{
	GpuTest gpu;

	{
		RenderGraph graph;
		const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
		const auto y = graph.CreateBuffer(UIntBuffer(kCount, "Y"));
		graph.AddPass(Pass::Compute("WriteX").Writes(x));
		graph.AddPass(Pass::Compute("AsyncRead").OnQueue(kCompute).Reads(x).Writes(y));
		assert(!graph.Compile(gpu.rc()));
		assert(!graph.IsCompiled());
		assert(graph.GetError().find("both the graphics and compute queues") != std::string::npos);
		assert(graph.GetError().find("Submit()") != std::string::npos);
	}
	{
		RenderGraph graph;
		const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
		const auto y = graph.CreateBuffer(UIntBuffer(kCount, "Y"));
		graph.AddPass(Pass::Compute("WriteX").Writes(x));
		graph.Submit();
		graph.AddPass(Pass::Compute("AsyncRead").OnQueue(kCompute).Reads(x).Writes(y));
		assert(graph.Compile(gpu.rc()));
		assert(graph.GetQueueWaits().size() == 1);
	}
}

// An imported counter used only on compute moves there in the prologue and back in the epilogue.
// Many overlapping frames must still count every increment.
RG_TEST(AsyncComputeImportedResourceRoundTrip)
{
	GpuTest gpu;
	const ResourceHandle counterHandle = gpu.CreateBuffer(kCount, "Counter");
	ZeroBuffer(gpu, counterHandle);

	RenderGraph graph;
	const auto counter = graph.ImportBuffer(counterHandle, "Counter");
	graph.AddPass(Pass::Compute("AsyncIncrement").OnQueue(kCompute).Use(counter, Access::ComputeShaderReadWrite).Executes(
			[&](PassContext &ctx) { gpu.Increment(ctx, counter, 2, kCount); }));
	assert(graph.Compile(gpu.rc()));

	const uint32_t s1 = RenderGraph::kFirstUserSection;
	const uint32_t prologue = RenderGraph::kPrologueSection;
	const uint32_t epilogue = graph.GetEpilogueSection();

	// Graphics submits the prologue for compute to wait on, then waits for compute in the epilogue.
	// A buffer needs no release barriers on either side.
	assert(FindRelease(graph, prologue, kGraphics, counter) == nullptr);
	assert(graph.IsSubmitSignalled(prologue, kGraphics));
	assert(FindRelease(graph, s1, kCompute, counter) == nullptr);
	assert(graph.IsSubmitSignalled(s1, kCompute));

	const auto &waits = graph.GetQueueWaits();
	assert(waits.size() == 2);
	assert(waits[0].waitingQueue == kCompute && waits[0].waitingSection == s1 && waits[0].signalSection == prologue);
	assert(waits[1].waitingQueue == kGraphics && waits[1].waitingSection == epilogue && waits[1].signalSection == s1);

	constexpr uint32_t kFrames = 8;
	for (uint32_t frame = 0; frame < kFrames; ++frame)
	{
		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame(false);
	}
	gpu.rc().WaitForGPUIdle();

	// The context tracks the state of the counter's last use.
	const ResourceState state = gpu.rc().GetResourceState(counterHandle);
	assert(state.access == ResourceAccess::UnorderedAccess);

	AssertBufferEquals(gpu.Read(counterHandle, kCount), [](uint32_t) { return 2 * kFrames; }, "Counter", &graph);
}

// A texture moving between queues changes layout: graphics renders it, releases it to COMMON (usable
// on every queue type), then compute acquires it as a shader resource.
RG_TEST(AsyncComputeTextureQueueTransfer)
{
	GpuTest gpu;
	constexpr uint32_t kSize = 64;
	const ResourceHandle outHandle = gpu.CreateBuffer(kSize * kSize, "Pixels");

	RenderGraph graph;
	TextureDesc textureDesc;
	textureDesc.width = kSize;
	textureDesc.height = kSize;
	textureDesc.format = Format::R8G8B8A8_UNORM;
	textureDesc.debugName = "Target";
	textureDesc.clearValue = ClearValue{ { 0.2f, 0.4f, 0.6f, 1.0f } };
	const auto target = graph.CreateTexture(textureDesc, ResourceViewType::ShaderResource);
	const auto out = graph.ImportBuffer(outHandle, "Pixels");

	graph.AddPass(Pass::Graphics("Clear").Writes(target).Executes(
			[&](PassContext &ctx)
			{
				float colour[4] = { 0.2f, 0.4f, 0.6f, 1.0f };
				ctx.rc.ClearRenderTarget(ctx.GetResource(target), colour);
			}));
	graph.Submit();
	graph.AddPass(Pass::Compute("AsyncReadback").OnQueue(kCompute).Reads(target).Writes(out).Executes(
			[&](PassContext &ctx) { gpu.TextureToBuffer(ctx, target, out, kSize, kSize); }));

	assert(graph.Compile(gpu.rc()));
	const uint32_t s1 = RenderGraph::kFirstUserSection;
	const uint32_t s2 = s1 + 1;

	const rg::BarrierInfo *release = FindRelease(graph, s1, kGraphics, target);
	assert(release && release->before.layout == ResourceLayout::RenderTarget);
	assert(release->after.layout == ResourceLayout::Common);
	const rg::BarrierInfo *acquire = FindBarrier(graph.GetBatch(s2, kCompute, 0), target);
	assert(acquire && acquire->kind == BarrierKind::QueueAcquire);
	assert(acquire->before.layout == ResourceLayout::Common && acquire->after.layout == ResourceLayout::ShaderResource);

	gpu.BeginFrame();
	graph.Execute(gpu.rc());
	gpu.EndFrame();

	const uint32_t r = 51; // round(0.2 * 255)
	const uint32_t g = 102; // round(0.4 * 255)
	const uint32_t b = 153; // round(0.6 * 255)
	const uint32_t packed = r | (g << 8) | (b << 16) | (255u << 24);
	AssertBufferEquals(gpu.Read(outHandle, kSize * kSize), [=](uint32_t) { return packed; }, "Pixels", &graph);
}

// The HTML printer draws every pass, one accent arrow per cross-queue wait, and escapes names.
RG_TEST(HtmlDumpDrawsScheduleAndWaits)
{
	GpuTest gpu;
	RenderGraph graph;
	const auto a = graph.CreateBuffer(UIntBuffer(kCount, "A"));
	const auto b = graph.CreateBuffer(UIntBuffer(kCount, "B<&>"));
	const auto c = graph.CreateBuffer(UIntBuffer(kCount, "C"));
	graph.AddPass(Pass::Compute("Produce").Writes(a).Executes([](PassContext &) {}));
	graph.Submit();
	graph.AddPass(Pass::Compute("Async & Co").OnQueue(kCompute).Reads(a).Writes(b).Executes([](PassContext &) {}));
	graph.AddPass(Pass::Compute("Overlap").Writes(c).Executes([](PassContext &) {}));
	graph.Submit();
	graph.AddPass(Pass::Compute("Consume").Reads(b).Reads(c).Writes(a).Executes([](PassContext &) {}));
	assert(graph.Compile(gpu.rc()));

	const std::string page = graph.DumpHtml("Test <graph>");
	auto count = [&page](const std::string &text)
	{
		size_t n = 0;
		for (size_t at = page.find(text); at != std::string::npos; at = page.find(text, at + 1))
		{
			++n;
		}
		return n;
	};
	assert(page.rfind("<!DOCTYPE html>", 0) == 0);
	assert(count("Test &lt;graph&gt;") >= 1 && count("Test <graph>") == 0);
	assert(count(">Produce</text>") == 1 && count(">Overlap</text>") == 1 && count(">Consume</text>") == 1);
	assert(count(">Async &amp; Co</text>") == 1 && count("B&lt;&amp;&gt;") >= 1);
	assert(count("url(#rg-schedule-arrow-accent)") == graph.GetQueueWaits().size() + 1); // plus the legend's
	assert(count("aria-labelledby=\"rg-schedule-title rg-schedule-desc\"") == 1);
	assert(count("<svg") == 2 && count("</svg>") == 2);
}
