// Transient memory: lifetimes, aliasing placement, aliasing barriers and lifetime "wiggling".

#include <algorithm>
#include <random>

#include "TestCommon.h"

using namespace rgtest;
using rg::Access;
using rg::BarrierKind;
using rg::Pass;
using rg::PassContext;
using rg::RenderGraph;

namespace
{
	constexpr uint32_t kCount = 4096;
	constexpr uint32_t kSection = RenderGraph::kFirstUserSection;
	constexpr QueueType kGraphics = QueueType::Graphics;
	constexpr QueueType kCompute = QueueType::Compute;

	bool MemoryOverlaps(const rg::TransientAllocation &a, const rg::TransientAllocation &b)
	{
		return a.offset < b.offset + b.size && b.offset < a.offset + a.size;
	}

	bool LifetimesOverlap(const rg::TransientAllocation &a, const rg::TransientAllocation &b)
	{
		return a.firstBatch <= b.lastBatch && b.firstBatch <= a.lastBatch;
	}
} // namespace

// T0 -> T1 -> T2 -> Out. T0 dies (batch 1) before T2 is born (batch 2), so T2 reuses T0's memory and
// the heap holds two buffers instead of three. T2's first barrier waits for T0's last access.
RG_TEST(TransientAliasingReusesMemory)
{
	GpuTest gpu;
	const ResourceHandle outHandle = gpu.CreateBuffer(kCount, "Out");

	for (const bool alias: { true, false })
	{
		RenderGraph graph;
		const auto t0 = graph.CreateBuffer(UIntBuffer(kCount, "T0"));
		const auto t1 = graph.CreateBuffer(UIntBuffer(kCount, "T1"));
		const auto t2 = graph.CreateBuffer(UIntBuffer(kCount, "T2"));
		const auto out = graph.ImportBuffer(outHandle, "Out");

		graph.AddPass(Pass::Compute("FillT0").Writes(t0).Executes([&](PassContext &ctx) { gpu.Fill(ctx, t0, 100, kCount); }));
		graph.AddPass(Pass::Compute("T0toT1").Reads(t0).Writes(t1).Executes(
				[&](PassContext &ctx) { gpu.Add(ctx, t0, t1, 1, kCount); }));
		graph.AddPass(Pass::Compute("T1toT2").Reads(t1).Writes(t2).Executes(
				[&](PassContext &ctx) { gpu.Add(ctx, t1, t2, 2, kCount); }));
		graph.AddPass(Pass::Compute("T2toOut").Reads(t2).Writes(out).Executes(
				[&](PassContext &ctx) { gpu.Add(ctx, t2, out, 3, kCount); }));

		rg::CompileOptions options;
		options.aliasTransientMemory = alias;
		assert(graph.Compile(gpu.rc(), options));

		const auto a0 = *graph.GetTransientAllocation(t0);
		const auto a1 = *graph.GetTransientAllocation(t1);
		const auto a2 = *graph.GetTransientAllocation(t2);
		assert(a0.firstBatch == 0 && a0.lastBatch == 1);
		assert(a1.firstBatch == 1 && a1.lastBatch == 2);
		assert(a2.firstBatch == 2 && a2.lastBatch == 3);
		assert(!graph.GetTransientAllocation(out).has_value() && "imported resources are not placed");

		if (alias)
		{
			assert(a2.offset == a0.offset && "T2 reuses T0's memory");
			assert(!MemoryOverlaps(a1, a0) && !MemoryOverlaps(a1, a2));
			assert(graph.GetTransientMemorySize() == a0.size + a1.size);

			// The first occupant waits for anything (the previous execution may still use the memory). T2
			// waits for T0's last access (a compute-shader read) before taking over its bytes.
			const rg::BarrierInfo *t0Initial = FindBarrier(graph.GetBatch(kSection, kGraphics, 0), t0);
			assert(t0Initial->kind == BarrierKind::Initial && t0Initial->before.sync == ResourceSync::All);
			const rg::BarrierInfo *t2Initial = FindBarrier(graph.GetBatch(kSection, kGraphics, 2), t2);
			assert(t2Initial->kind == BarrierKind::Initial);
			assert(t2Initial->before.sync == ResourceSync::ComputeShading);
			assert(t2Initial->before.access == ResourceAccess::NoAccess && t2Initial->before.layout == ResourceLayout::Undefined);
		}
		else
		{
			assert(!MemoryOverlaps(a0, a1) && !MemoryOverlaps(a0, a2) && !MemoryOverlaps(a1, a2));
			assert(graph.GetTransientMemorySize() == a0.size + a1.size + a2.size);
		}

		// Run it a few times: data passing through reused memory must stay intact.
		for (uint32_t frame = 0; frame < 3; ++frame)
		{
			gpu.BeginFrame();
			graph.Execute(gpu.rc());
			gpu.EndFrame();
			AssertBufferEquals(gpu.Read(outHandle, kCount), [](uint32_t i) { return 100 + i + 1 + 2 + 3; }, "Out", &graph);
		}
		graph.ReleaseResources();
	}
}

// Random graphs in every scheduling mode: overlapping lifetimes never share bytes and every
// dependency between passes sharing a resource is scheduled in order.
RG_TEST(AliasingNeverOverlapsLiveMemoryRandomized)
{
	GpuTest gpu;

	struct Use
	{
		uint32_t resource;
		bool write;
	};
	struct GeneratedPass
	{
		rg::PassId id;
		uint32_t section;
		QueueType queue;
		std::vector<Use> uses;
	};

	for (uint32_t seed = 1; seed <= 40; ++seed)
	{
		std::mt19937 rng(seed);
		auto random = [&rng](uint32_t n) { return static_cast<uint32_t>(rng() % n); };

		RenderGraph graph;
		const uint32_t resourceCount = 6 + random(12);
		std::vector<rg::ResourceId> resources;
		for (uint32_t r = 0; r < resourceCount; ++r)
		{
			const uint32_t elements = (1 + random(64)) * 1024; // 4 KiB .. 256 KiB
			resources.push_back(graph.CreateBuffer(UIntBuffer(elements, "R")));
		}

		std::vector<bool> written(resourceCount, false);
		std::vector<int> ownerQueue(resourceCount, -1); // queue that uses each resource in the current section
		uint32_t section = RenderGraph::kFirstUserSection;
		std::vector<GeneratedPass> generated;

		const uint32_t passCount = 15 + random(30);
		for (uint32_t p = 0; p < passCount; ++p)
		{
			if (p > 0 && random(6) == 0)
			{
				graph.Submit();
				++section;
				std::fill(ownerQueue.begin(), ownerQueue.end(), -1);
			}

			const QueueType queue = random(3) == 0 ? kCompute : kGraphics;
			Pass pass = Pass::Compute("P" + std::to_string(p)).OnQueue(queue);
			GeneratedPass record{ {}, section, queue, {} };

			const uint32_t wanted = 1 + random(3);
			for (uint32_t k = 0; k < wanted; ++k)
			{
				const uint32_t r = random(resourceCount);
				const bool duplicate = std::any_of(
						record.uses.begin(), record.uses.end(), [r](const Use &u) { return u.resource == r; });
				const bool otherQueue = ownerQueue[r] != -1 && ownerQueue[r] != static_cast<int>(queue);
				if (duplicate || otherQueue)
				{
					continue;
				}

				Access access = Access::ComputeShaderWrite;
				if (written[r])
				{
					const uint32_t pick = random(4);
					access = pick < 2 ? Access::ComputeShaderRead
									  : (pick == 2 ? Access::ComputeShaderWrite : Access::ComputeShaderReadWrite);
				}
				pass.Use(resources[r], access);
				record.uses.push_back({ r, rg::IsWriteAccess(access) });
				written[r] = true;
				ownerQueue[r] = static_cast<int>(queue);
			}
			if (record.uses.empty())
			{
				continue;
			}
			record.id = graph.AddPass(std::move(pass));
			generated.push_back(std::move(record));
		}

		struct Mode
		{
			bool reorder;
			bool wiggle;
		};
		uint32_t linearBatches = 0;
		uint64_t aliasedMemory[3] = {};
		uint32_t modeIndex = 0;
		for (const Mode mode: { Mode{ false, false }, Mode{ true, false }, Mode{ true, true } })
		{
			rg::CompileOptions options;
			options.reorderPasses = mode.reorder;
			options.optimizeLifetimes = mode.wiggle;
			const bool compiled = graph.Compile(gpu.rc(), options);
			if (!compiled)
			{
				std::fprintf(stderr, "seed %u: %s\n", seed, graph.GetError().c_str());
			}
			assert(compiled);

			// Dependencies are scheduled in order.
			for (size_t i = 0; i < generated.size(); ++i)
			{
				for (size_t j = i + 1; j < generated.size(); ++j)
				{
					const GeneratedPass &earlier = generated[i];
					const GeneratedPass &later = generated[j];
					if (earlier.section != later.section || earlier.queue != later.queue)
					{
						continue; // ordered by section, or on a queue it may not share resources with
					}
					for (const Use &u: earlier.uses)
					{
						for (const Use &v: later.uses)
						{
							if (u.resource == v.resource && (u.write || v.write) &&
								graph.GetPassBatch(earlier.id) >= graph.GetPassBatch(later.id))
							{
								std::fprintf(stderr, "seed %u: dependency out of order\n%s", seed, graph.Dump().c_str());
								assert(false);
							}
						}
					}
				}
			}

			// Live resources never share memory; placements are aligned and inside the heap.
			uint64_t alignedSum = 0;
			for (uint32_t a = 0; a < resourceCount; ++a)
			{
				const auto allocA = graph.GetTransientAllocation(resources[a]);
				if (!allocA)
				{
					continue; // never used
				}
				assert(allocA->offset % allocA->alignment == 0);
				assert(allocA->offset + allocA->size <= graph.GetTransientMemorySize());
				alignedSum += (allocA->size + allocA->alignment - 1) / allocA->alignment * allocA->alignment;
				for (uint32_t b = a + 1; b < resourceCount; ++b)
				{
					const auto allocB = graph.GetTransientAllocation(resources[b]);
					if (allocB && LifetimesOverlap(*allocA, *allocB) && MemoryOverlaps(*allocA, *allocB))
					{
						std::fprintf(stderr, "seed %u: live resources %u and %u overlap\n%s", seed, a, b, graph.Dump().c_str());
						assert(false);
					}
				}
			}
			assert(graph.GetTransientMemorySize() <= alignedSum);
			aliasedMemory[modeIndex] = graph.GetTransientMemorySize();

			if (!mode.reorder)
			{
				linearBatches = graph.GetTotalBatchCount();
			}
			else
			{
				assert(graph.GetTotalBatchCount() <= linearBatches && "reordering never adds batches");
			}
			assert(!graph.Dump().empty());
			++modeIndex;
		}

		// Without aliasing every resource has its own range: never smaller than with aliasing.
		rg::CompileOptions noAlias;
		noAlias.aliasTransientMemory = false;
		assert(graph.Compile(gpu.rc(), noAlias));
		assert(graph.GetTransientMemorySize() >= aliasedMemory[2]);
	}
}

// Transients in a section using both queues live for the whole section, so nothing there aliases.
// The same passes on one queue let G0 and G2 share memory.
RG_TEST(AsyncSectionsDoNotAliasAcrossQueues)
{
	GpuTest gpu;
	const ResourceHandle outGHandle = gpu.CreateBuffer(kCount, "OutG");
	const ResourceHandle outCHandle = gpu.CreateBuffer(kCount, "OutC");

	for (const bool async: { true, false })
	{
		RenderGraph graph;
		const auto g0 = graph.CreateBuffer(UIntBuffer(kCount, "G0"));
		const auto g1 = graph.CreateBuffer(UIntBuffer(kCount, "G1"));
		const auto g2 = graph.CreateBuffer(UIntBuffer(kCount, "G2"));
		const auto c0 = graph.CreateBuffer(UIntBuffer(kCount, "C0"));
		const auto outG = graph.ImportBuffer(outGHandle, "OutG");
		const auto outC = graph.ImportBuffer(outCHandle, "OutC");
		const QueueType sideQueue = async ? kCompute : kGraphics;

		graph.AddPass(Pass::Compute("FillG0").Writes(g0).Executes([&](PassContext &ctx) { gpu.Fill(ctx, g0, 100, kCount); }));
		graph.AddPass(Pass::Compute("G0toG1").Reads(g0).Writes(g1).Executes([&](PassContext &ctx) { gpu.Add(ctx, g0, g1, 1, kCount); }));
		graph.AddPass(Pass::Compute("G1toG2").Reads(g1).Writes(g2).Executes([&](PassContext &ctx) { gpu.Add(ctx, g1, g2, 2, kCount); }));
		graph.AddPass(Pass::Compute("G2toOut").Reads(g2).Writes(outG).Executes(
				[&](PassContext &ctx) { gpu.Add(ctx, g2, outG, 3, kCount); }));
		graph.AddPass(Pass::Compute("FillC0").OnQueue(sideQueue).Writes(c0).Executes(
				[&](PassContext &ctx) { gpu.Fill(ctx, c0, 300, kCount); }));
		graph.AddPass(Pass::Compute("C0toOut").OnQueue(sideQueue).Reads(c0).Writes(outC).Executes(
				[&](PassContext &ctx) { gpu.Add(ctx, c0, outC, 5, kCount); }));

		assert(graph.Compile(gpu.rc()));

		const rg::ResourceId transients[] = { g0, g1, g2, c0 };
		if (async)
		{
			for (const rg::ResourceId a: transients)
			{
				const auto allocA = *graph.GetTransientAllocation(a);
				assert(allocA.firstBatch == 0 && allocA.lastBatch == 3 && "live for the whole section");
				for (const rg::ResourceId b: transients)
				{
					assert(a == b || !MemoryOverlaps(allocA, *graph.GetTransientAllocation(b)));
				}
			}
			assert(graph.GetTransientMemorySize() == 4 * graph.GetTransientAllocation(g0)->size);
		}
		else
		{
			assert(MemoryOverlaps(*graph.GetTransientAllocation(g0), *graph.GetTransientAllocation(g2)));
			assert(graph.GetTransientMemorySize() == 3 * graph.GetTransientAllocation(g0)->size);
		}

		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame();
		AssertBufferEquals(gpu.Read(outGHandle, kCount), [](uint32_t i) { return 100 + i + 1 + 2 + 3; }, "OutG", &graph);
		AssertBufferEquals(gpu.Read(outCHandle, kCount), [](uint32_t i) { return 300 + i + 5; }, "OutC", &graph);
		graph.ReleaseResources();
	}
}

// X is made first but only used at the end of a chain, so the wiggle moves its producer from batch 0
// to batch 2: same batch count, smaller virtual area.
RG_TEST(WigglingShrinksTransientLifetimes)
{
	GpuTest gpu;
	const ResourceHandle outHandle = gpu.CreateBuffer(kCount, "Out");

	uint64_t areaWithout = 0;
	uint64_t memoryWithout = 0;
	for (const bool wiggle: { false, true })
	{
		RenderGraph graph;
		const auto x = graph.CreateBuffer(UIntBuffer(kCount, "X"));
		const auto y0 = graph.CreateBuffer(UIntBuffer(kCount, "Y0"));
		const auto y1 = graph.CreateBuffer(UIntBuffer(kCount, "Y1"));
		const auto y2 = graph.CreateBuffer(UIntBuffer(kCount, "Y2"));
		const auto out = graph.ImportBuffer(outHandle, "Out");

		const auto fillX = graph.AddPass(
				Pass::Compute("FillX").Writes(x).Executes([&](PassContext &ctx) { gpu.Fill(ctx, x, 100, kCount); }));
		graph.AddPass(Pass::Compute("FillY0").Writes(y0).Executes([&](PassContext &ctx) { gpu.Fill(ctx, y0, 200, kCount); }));
		graph.AddPass(Pass::Compute("Y0toY1").Reads(y0).Writes(y1).Executes([&](PassContext &ctx) { gpu.Add(ctx, y0, y1, 1, kCount); }));
		graph.AddPass(Pass::Compute("Y1toY2").Reads(y1).Writes(y2).Executes([&](PassContext &ctx) { gpu.Add(ctx, y1, y2, 2, kCount); }));
		const auto combine = graph.AddPass(Pass::Compute("Combine").Reads(x).Reads(y2).Writes(out).Executes(
				[&](PassContext &ctx) { gpu.Combine(ctx, x, y2, out, kCount); }));

		rg::CompileOptions options;
		options.optimizeLifetimes = wiggle;
		assert(graph.Compile(gpu.rc(), options));
		assert(graph.GetBatchCount(kSection, kGraphics) == 4 && "wiggling never lengthens the critical path");
		assert(graph.GetPassBatch(combine) == 3);

		const auto xAlloc = *graph.GetTransientAllocation(x);
		if (!wiggle)
		{
			assert(graph.GetPassBatch(fillX) == 0);
			assert(xAlloc.firstBatch == 0 && xAlloc.lastBatch == 3);
			areaWithout = graph.GetVirtualArea();
			memoryWithout = graph.GetTransientMemorySize();
		}
		else
		{
			assert(graph.GetPassBatch(fillX) == 2 && "moved as late as its consumer allows");
			assert(xAlloc.firstBatch == 2 && xAlloc.lastBatch == 3);
			assert(graph.GetVirtualArea() < areaWithout);
			assert(graph.GetTransientMemorySize() <= memoryWithout);
		}

		gpu.BeginFrame();
		graph.Execute(gpu.rc());
		gpu.EndFrame();
		AssertBufferEquals(gpu.Read(outHandle, kCount), [](uint32_t i) { return (100 + i) + (200 + i + 1 + 2); }, "Out", &graph);
		graph.ReleaseResources();
	}
}
