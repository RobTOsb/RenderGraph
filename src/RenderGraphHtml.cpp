// HTML printer for a compiled graph: a queue swimlane, transient memory lifetimes and a barrier table.
// Styled after the diagram-design skill (github.com/cathrynlavery/diagram-design): one self-contained page.
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <initializer_list>
#include <sstream>

#include "RenderGraph.h"

namespace rg
{
	namespace
	{
		// Skin tokens, dark theme: light ink on jet-black paper, nodes on the raised paper-2.
		constexpr const char *kPaper = "#2d3142";
		constexpr const char *kPaper2 = "#393e53";
		constexpr const char *kInk = "#f5f5f5";
		constexpr const char *kMuted = "#bfc0c0";
		constexpr const char *kSoft = "#949eb2";
		constexpr const char *kAccent = "#f08a59";
		constexpr const char *kRule = "rgba(245,245,245,0.12)";
		constexpr const char *kRuleStrong = "rgba(245,245,245,0.22)";
		constexpr const char *kStoreFill = "rgba(245,245,245,0.05)";
		constexpr const char *kSans = "'Geist', system-ui, sans-serif";
		constexpr const char *kMono = "'Geist Mono', ui-monospace, monospace";

		// Geometry, all on the 4px grid.
		constexpr int kMargin = 40;
		constexpr int kLabelColumn = 160; // lane labels sit left of this line
		constexpr int kLaneTop = 72;
		constexpr int kLaneHeight = 104;
		constexpr int kBoxInset = 24; // box top inside its lane
		constexpr int kBoxHeight = 56;
		constexpr int kMinBoxWidth = 128;
		constexpr int kPassGap = 16; // passes in one batch
		constexpr int kBatchGap = 40;
		constexpr int kWaitSpacing = 12;
		constexpr int kPortNear = 14; // wait arrows leave and enter a quarter of the box from an edge
		constexpr int kPortFar = 42;

		size_t CharCount(const std::string &text)
		{
			return static_cast<size_t>(std::count_if(text.begin(),
													 text.end(),
													 [](char c)
													 { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }));
		}

		int RoundUp4(double value)
		{
			const int v = static_cast<int>(value + 0.999);
			return (v + 3) & ~3;
		}

		// Width budget from the style guide: 0.60em per sans character, 0.62em per mono one, plus tracking.
		int SansWidth(const std::string &text, double size)
		{
			return RoundUp4(static_cast<double>(CharCount(text)) * 0.60 * size);
		}

		int MonoWidth(const std::string &text, double size, double tracking = 0.0)
		{
			return RoundUp4(static_cast<double>(CharCount(text)) * (0.62 + tracking) * size);
		}

		std::string Escape(const std::string &text)
		{
			std::string out;
			out.reserve(text.size());
			for (char c: text)
			{
				switch (c)
				{
					case '&':
						out += "&amp;";
						break;
					case '<':
						out += "&lt;";
						break;
					case '>':
						out += "&gt;";
						break;
					case '"':
						out += "&quot;";
						break;
					default:
						out += c;
				}
			}
			return out;
		}

		std::string Upper(std::string text)
		{
			std::transform(text.begin(),
						   text.end(),
						   text.begin(),
						   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
			return text;
		}

		std::string Truncate(const std::string &text, size_t maxChars)
		{
			if (CharCount(text) <= maxChars)
			{
				return text;
			}
			std::string out;
			size_t chars = 0;
			for (char c: text)
			{
				if ((static_cast<unsigned char>(c) & 0xC0) != 0x80 && ++chars > maxChars - 1)
				{
					break;
				}
				out += c;
			}
			return out + "\xE2\x80\xA6"; // ellipsis
		}

		std::string FormatBytes(uint64_t bytes)
		{
			char text[32];
			if (bytes >= (1ull << 20))
			{
				std::snprintf(text,
							  sizeof(text),
							  "%.1f MiB",
							  static_cast<double>(bytes) / static_cast<double>(1ull << 20));
			}
			else if (bytes >= (1ull << 10))
			{
				std::snprintf(text, sizeof(text), "%.1f KiB", static_cast<double>(bytes) / 1024.0);
			}
			else
			{
				std::snprintf(text, sizeof(text), "%llu B", static_cast<unsigned long long>(bytes));
			}
			return text;
		}

		const char *KindName(PassKind kind)
		{
			switch (kind)
			{
				case PassKind::Copy:
					return "COPY";
				case PassKind::Compute:
					return "COMPUTE";
				case PassKind::Graphics:
					return "GRAPHICS";
			}
			return "?";
		}

		const char *BarrierKindName(BarrierKind kind)
		{
			switch (kind)
			{
				case BarrierKind::Transition:
					return "transition";
				case BarrierKind::Initial:
					return "initial";
				case BarrierKind::QueueRelease:
					return "release";
				case BarrierKind::QueueAcquire:
					return "acquire";
			}
			return "?";
		}

		std::string LayoutName(ResourceLayout layout)
		{
			switch (layout)
			{
				case ResourceLayout::Undefined:
					return "Undefined";
				case ResourceLayout::Common:
					return "Common";
				case ResourceLayout::GenericRead:
					return "GenericRead";
				case ResourceLayout::RenderTarget:
					return "RenderTarget";
				case ResourceLayout::UnorderedAccess:
					return "UnorderedAccess";
				case ResourceLayout::DepthStencilWrite:
					return "DepthStencilWrite";
				case ResourceLayout::DepthStencilRead:
					return "DepthStencilRead";
				case ResourceLayout::ShaderResource:
					return "ShaderResource";
				case ResourceLayout::CopySource:
					return "CopySource";
				case ResourceLayout::CopyDest:
					return "CopyDest";
				case ResourceLayout::ShadingRateSource:
					return "ShadingRateSource";
				default:
					return "Layout " + std::to_string(static_cast<uint32_t>(layout));
			}
		}

		struct FlagName
		{
			uint32_t bit;
			const char *name;
		};

		std::string FlagNames(uint32_t value, std::initializer_list<FlagName> names, const char *zero)
		{
			if (value == 0)
			{
				return zero;
			}
			std::string out;
			for (const FlagName &flag: names)
			{
				if ((value & flag.bit) != 0)
				{
					out += (out.empty() ? "" : "|") + std::string(flag.name);
					value &= ~flag.bit;
				}
			}
			if (value != 0)
			{
				char hex[16];
				std::snprintf(hex, sizeof(hex), "0x%x", value);
				out += (out.empty() ? "" : "|") + std::string(hex);
			}
			return out;
		}

		std::string SyncName(ResourceSync sync)
		{
			return FlagNames(static_cast<uint32_t>(sync),
							 { { 0x1, "All" },
							   { 0x2, "Draw" },
							   { 0x4, "IndexInput" },
							   { 0x8, "VertexShading" },
							   { 0x10, "PixelShading" },
							   { 0x20, "DepthStencil" },
							   { 0x40, "RenderTarget" },
							   { 0x80, "ComputeShading" },
							   { 0x100, "Raytracing" },
							   { 0x200, "Copy" },
							   { 0x400, "Resolve" },
							   { 0x800, "ExecuteIndirect" },
							   { 0x1000, "AllShading" },
							   { 0x2000, "NonPixelShading" },
							   { 0x4000, "EmitPostbuildInfo" },
							   { 0x8000, "ClearUAV" },
							   { 0x800000, "BuildAS" },
							   { 0x1000000, "CopyAS" } },
							 "None");
		}

		std::string AccessName(ResourceAccess access)
		{
			return FlagNames(static_cast<uint32_t>(access),
							 { { 0x80000000u, "NoAccess" },
							   { 0x1, "VertexBuffer" },
							   { 0x2, "ConstantBuffer" },
							   { 0x4, "IndexBuffer" },
							   { 0x8, "RenderTarget" },
							   { 0x10, "UnorderedAccess" },
							   { 0x20, "DepthStencilWrite" },
							   { 0x40, "DepthStencilRead" },
							   { 0x80, "ShaderResource" },
							   { 0x200, "IndirectArgument" },
							   { 0x400, "CopyDest" },
							   { 0x800, "CopySource" },
							   { 0x4000, "ASRead" },
							   { 0x8000, "ASWrite" },
							   { 0x10000, "ShadingRateSource" } },
							 "Common");
		}

		std::string StateText(const ResourceState &state)
		{
			return LayoutName(state.layout) + " \xC2\xB7 " + AccessName(state.access) + " (" + SyncName(state.sync) +
				   ")";
		}

		// Small SVG writer: every call is one element, attributes in the skin's vocabulary.
		struct Svg
		{
			std::ostringstream out;

			void Rect(int x,
					  int y,
					  int w,
					  int h,
					  int rx,
					  const char *fill,
					  const char *stroke = nullptr,
					  double strokeWidth = 1.0,
					  const char *dash = nullptr)
			{
				out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w << "\" height=\"" << h << "\" rx=\""
					<< rx << "\" fill=\"" << fill << "\"";
				if (stroke)
				{
					out << " stroke=\"" << stroke << "\" stroke-width=\"" << strokeWidth << "\"";
				}
				if (dash)
				{
					out << " stroke-dasharray=\"" << dash << "\"";
				}
				out << "/>\n";
			}

			void Text(double x,
					  double y,
					  const std::string &text,
					  double size,
					  const char *fill,
					  const char *family,
					  const char *anchor = "start",
					  int weight = 400,
					  double tracking = 0.0)
			{
				out << "<text x=\"" << x << "\" y=\"" << y << "\" fill=\"" << fill << "\" font-size=\"" << size
					<< "\" font-family=\"" << family << "\" text-anchor=\"" << anchor << "\"";
				if (weight != 400)
				{
					out << " font-weight=\"" << weight << "\"";
				}
				if (tracking != 0.0)
				{
					out << " letter-spacing=\"" << tracking << "em\"";
				}
				out << ">" << Escape(text) << "</text>\n";
			}

			void Line(int x1,
					  int y1,
					  int x2,
					  int y2,
					  const char *stroke,
					  double width,
					  const char *dash = nullptr,
					  const char *marker = nullptr)
			{
				out << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\" stroke=\""
					<< stroke << "\" stroke-width=\"" << width << "\"";
				if (dash)
				{
					out << " stroke-dasharray=\"" << dash << "\"";
				}
				if (marker)
				{
					out << " marker-end=\"url(#" << marker << ")\"";
				}
				out << "/>\n";
			}

			void Path(const std::string &d, const char *stroke, double width, const char *marker)
			{
				out << "<path d=\"" << d << "\" fill=\"none\" stroke=\"" << stroke << "\" stroke-width=\"" << width
					<< "\" marker-end=\"url(#" << marker << ")\"/>\n";
			}

			// The rectangular type tag of a node box: eyebrow text in a hairline box.
			void Tag(int x, int y, const std::string &text, const char *colour)
			{
				const int w = MonoWidth(text, 7, 0.08) + 8;
				out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w
					<< "\" height=\"12\" rx=\"2\" fill=\"none\" stroke=\"" << colour
					<< "\" stroke-opacity=\"0.4\" stroke-width=\"0.8\"/>\n";
				Text(x + w / 2.0, y + 9, text, 7, colour, kMono, "middle", 400, 0.08);
			}
		};

		void OpenSvg(std::ostringstream &out,
					 const std::string &slug,
					 int width,
					 int height,
					 const std::string &title,
					 const std::string &desc)
		{
			// Fills the frame, but never shrinks below its own size: a narrow window scrolls the figure instead.
			out << "<svg style=\"min-width:" << width << "px\" viewBox=\"0 0 " << width << " " << height
				<< "\" xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-labelledby=\"" << slug << "-title "
				<< slug << "-desc\">\n<title id=\"" << slug << "-title\">" << Escape(title) << "</title>\n<desc id=\""
				<< slug << "-desc\">" << Escape(desc) << "</desc>\n<defs>\n"
				<< "<marker id=\"" << slug
				<< "-arrow\" markerWidth=\"8\" markerHeight=\"6\" refX=\"7\" refY=\"3\" orient=\"auto\">"
				<< "<polygon points=\"0 0, 8 3, 0 6\" fill=\"" << kMuted << "\"/></marker>\n"
				<< "<marker id=\"" << slug
				<< "-arrow-accent\" markerWidth=\"8\" markerHeight=\"6\" refX=\"7\" refY=\"3\" orient=\"auto\">"
				<< "<polygon points=\"0 0, 8 3, 0 6\" fill=\"" << kAccent << "\"/></marker>\n</defs>\n"
				<< "<rect width=\"100%\" height=\"100%\" fill=\"" << kPaper << "\"/>\n";
		}

		std::string Plural(size_t count, const char *word)
		{
			return std::to_string(count) + " " + word + (count == 1 ? "" : "S");
		}
	} // namespace

	std::string RenderGraph::DumpHtml(const std::string &title) const
	{
		if (!compiled_)
		{
			return "<!DOCTYPE html>\n<html><body><p>Render graph not compiled: " + Escape(error_) +
				   "</p></body></html>\n";
		}

		// One box per pass, or a stub where a queue only waits or releases in a section.
		struct Box
		{
			int x = 0;
			int y = 0;
			int w = 0;
			std::string name;
			std::string sublabel;
			std::string kindTag;
			std::string barrierTag;
			bool stub = false;
		};
		struct Batch
		{
			std::vector<Box> boxes;
			int width = 0;
		};
		struct Cell // one queue in one section
		{
			std::vector<Batch> batches;
		};

		std::vector<uint32_t> lanes; // queue indices that do any work
		for (uint32_t q = 0; q < kQueueTypeCount; ++q)
		{
			if (std::any_of(sections_.begin(), sections_.end(), [q](const Section &s) { return s.queues[q].active; }))
			{
				lanes.push_back(q);
			}
		}
		std::vector<uint32_t> visible; // sections where some queue is active
		for (uint32_t s = 0; s < sections_.size(); ++s)
		{
			if (std::any_of(std::begin(sections_[s].queues),
							std::end(sections_[s].queues),
							[](const QueueSection &q) { return q.active; }))
			{
				visible.push_back(s);
			}
		}

		auto writesOf = [&](const PassNode &pass)
		{
			std::vector<std::string> names;
			for (const Attachment &attachment: pass.desc.attachments)
			{
				const std::string &name = resources_[attachment.resource.index].name;
				if (IsWriteAccess(attachment.access) && std::find(names.begin(), names.end(), name) == names.end())
				{
					names.push_back(name);
				}
			}
			std::string text;
			for (const std::string &name: names)
			{
				text += (text.empty() ? "\xE2\x86\x92 " : ", ") + name; // arrow, then the written resources
			}
			return text.empty() ? std::string("reads only") : Truncate(text, 30);
		};
		auto finishBox = [](Box &box)
		{
			const int tags = (box.kindTag.empty() ? 0 : MonoWidth(box.kindTag, 7, 0.08) + 8) +
							 (box.barrierTag.empty() ? 0 : MonoWidth(box.barrierTag, 7, 0.08) + 16);
			box.w = std::max(
					{ kMinBoxWidth, SansWidth(box.name, 12) + 32, MonoWidth(box.sublabel, 9) + 32, tags + 16 });
		};

		// cells[section][lane]
		std::vector<std::vector<Cell>> cells(sections_.size(), std::vector<Cell>(lanes.size()));
		size_t passCount = 0;
		size_t barrierCount = 0;
		for (uint32_t s: visible)
		{
			for (size_t l = 0; l < lanes.size(); ++l)
			{
				const QueueSection &queueSection = sections_[s].queues[lanes[l]];
				Cell &cell = cells[s][l];
				barrierCount += queueSection.releases.size();
				for (const CompiledBatch &compiled: queueSection.batches)
				{
					Batch batch;
					barrierCount += compiled.barriers.size();
					for (size_t p = 0; p < compiled.passes.size(); ++p)
					{
						const PassNode &pass = passes_[compiled.passes[p].index];
						Box box;
						box.name = pass.desc.name;
						box.sublabel = writesOf(pass);
						box.kindTag = KindName(pass.desc.kind);
						if (p == 0 && !compiled.barriers.empty())
						{
							box.barrierTag = Plural(compiled.barriers.size(), "BARRIER");
						}
						finishBox(box);
						batch.width += box.w + (batch.boxes.empty() ? 0 : kPassGap);
						batch.boxes.push_back(std::move(box));
						++passCount;
					}
					cell.batches.push_back(std::move(batch));
				}
				if (queueSection.active && cell.batches.empty())
				{
					Box box;
					box.stub = true;
					box.name = queueSection.releases.empty() ? "Wait" : "Release";
					for (const BarrierInfo &release: queueSection.releases)
					{
						box.sublabel += (box.sublabel.empty() ? "" : ", ") + resources_[release.resource.index].name;
					}
					box.sublabel = Truncate(box.sublabel, 30);
					finishBox(box);
					cell.batches.push_back({ { box }, box.w });
				}
			}
		}

		// The gap before a section holds the submit line and the vertical runs of the waits into it.
		auto waitsInto = [&](uint32_t section)
		{
			std::vector<const QueueWaitInfo *> waits;
			for (const QueueWaitInfo &wait: queueWaits_)
			{
				if (wait.waitingSection == section && wait.waitingQueue != wait.signalQueue)
				{
					waits.push_back(&wait);
				}
			}
			return waits;
		};

		// Horizontal layout: batches in one section line up in slots across queues.
		std::vector<int> sectionX(sections_.size(), 0);
		std::vector<int> sectionW(sections_.size(), 0);
		std::vector<int> submitX(sections_.size(), 0); // the dashed submit line before a section
		int x = kLabelColumn + 32;
		for (size_t v = 0; v < visible.size(); ++v)
		{
			const uint32_t s = visible[v];
			if (v > 0)
			{
				const size_t waits = waitsInto(s).size();
				submitX[s] = x + 24;
				x += waits == 0 ? 48 : 60 + kWaitSpacing * static_cast<int>(waits - 1);
			}
			sectionX[s] = x;
			size_t slots = 0;
			for (const Cell &cell: cells[s])
			{
				slots = std::max(slots, cell.batches.size());
			}
			for (size_t b = 0; b < slots; ++b)
			{
				int slotW = 0;
				for (const Cell &cell: cells[s])
				{
					if (b < cell.batches.size())
					{
						slotW = std::max(slotW, cell.batches[b].width);
					}
				}
				for (size_t l = 0; l < lanes.size(); ++l)
				{
					if (b >= cells[s][l].batches.size())
					{
						continue;
					}
					Batch &batch = cells[s][l].batches[b];
					int bx = x + RoundUp4((slotW - batch.width) / 2.0);
					for (Box &box: batch.boxes)
					{
						box.x = bx;
						box.y = kLaneTop + static_cast<int>(l) * kLaneHeight + kBoxInset;
						bx += box.w + kPassGap;
					}
				}
				x += slotW + (b + 1 < slots ? kBatchGap : 0);
			}
			sectionW[s] = x - sectionX[s];
		}
		const int lanesBottom = kLaneTop + static_cast<int>(lanes.size()) * kLaneHeight;
		const int legendY = lanesBottom + 24;
		const int width = std::max(x + kMargin, 880);
		const int height = legendY + 60;

		// Every pass and wait in a section, then each queue's boxes in order for the lane arrows.
		auto laneOf = [&](QueueType queue)
		{ return static_cast<size_t>(std::find(lanes.begin(), lanes.end(), QueueIndex(queue)) - lanes.begin()); };
		auto firstBox = [&](uint32_t s, size_t l) { return cells[s][l].batches.front().boxes.front(); };
		auto lastBox = [&](uint32_t s, size_t l) { return cells[s][l].batches.back().boxes.back(); };

		std::ostringstream html;
		Svg svg;
		svg.out << "<!-- lanes -->\n";
		for (size_t l = 0; l <= lanes.size(); ++l)
		{
			const int y = kLaneTop + static_cast<int>(l) * kLaneHeight;
			svg.Line(kMargin, y, width - kMargin, y, (l == 0 || l == lanes.size()) ? kRuleStrong : kRule, 1);
		}
		svg.Line(kLabelColumn, kLaneTop, kLabelColumn, lanesBottom, kRuleStrong, 1);
		for (size_t l = 0; l < lanes.size(); ++l)
		{
			const int y = kLaneTop + static_cast<int>(l) * kLaneHeight;
			svg.Text(kMargin + 20,
					 y + kLaneHeight / 2 + 3,
					 Upper(ToString(static_cast<QueueType>(lanes[l]))),
					 9,
					 kMuted,
					 kMono,
					 "start",
					 500,
					 0.18);
		}
		for (size_t v = 0; v < visible.size(); ++v)
		{
			const uint32_t s = visible[v];
			const std::string label = s == kPrologueSection		  ? "PROLOGUE"
									  : s == GetEpilogueSection() ? "EPILOGUE"
																  : "SECTION " + std::to_string(s);
			svg.Text(sectionX[s] + sectionW[s] / 2.0, kLaneTop - 16, label, 8, kMuted, kMono, "middle", 500, 0.18);
			if (v > 0)
			{
				svg.Line(submitX[s], kLaneTop - 8, submitX[s], lanesBottom, kRuleStrong, 1, "4,3");
			}
		}

		svg.out << "<!-- arrows: queue order, then cross-queue waits -->\n";
		for (size_t l = 0; l < lanes.size(); ++l)
		{
			const Box *previous = nullptr;
			for (uint32_t s: visible)
			{
				for (const Batch &batch: cells[s][l].batches)
				{
					const Box &first = batch.boxes.front();
					if (previous)
					{
						const int y = first.y + kBoxHeight / 2;
						svg.Line(previous->x + previous->w, y, first.x, y, kMuted, 1.2, nullptr, "rg-schedule-arrow");
					}
					previous = &batch.boxes.back();
				}
			}
		}
		for (uint32_t s: visible)
		{
			const auto waits = waitsInto(s);
			for (size_t w = 0; w < waits.size(); ++w)
			{
				const QueueWaitInfo &wait = *waits[w];
				const size_t from = laneOf(wait.signalQueue);
				const size_t to = laneOf(wait.waitingQueue);
				if (from >= lanes.size() || to >= lanes.size() || cells[wait.signalSection][from].batches.empty() ||
					cells[s][to].batches.empty())
				{
					continue;
				}
				const Box &source = lastBox(wait.signalSection, from);
				const Box &target = firstBox(s, to);
				const bool down = to > from;
				const int vx = submitX[s] + 16 + kWaitSpacing * static_cast<int>(w);
				const int y1 = source.y + (down ? kPortFar : kPortNear);
				const int y2 = target.y + (down ? kPortNear : kPortFar);
				const int turn = down ? 8 : -8;
				std::ostringstream d;
				d << "M " << source.x + source.w << "," << y1 << " H " << vx - 8 << " Q " << vx << "," << y1 << " "
				  << vx << "," << y1 + turn << " V " << y2 - turn << " Q " << vx << "," << y2 << " " << vx + 8 << ","
				  << y2 << " H " << target.x;
				svg.Path(d.str(), kAccent, 1.4, "rg-schedule-arrow-accent");
			}
		}

		svg.out << "<!-- batches and passes -->\n";
		for (uint32_t s: visible)
		{
			for (size_t l = 0; l < lanes.size(); ++l)
			{
				for (const Batch &batch: cells[s][l].batches)
				{
					if (batch.boxes.size() > 1)
					{
						const Box &first = batch.boxes.front();
						svg.Rect(first.x - 8,
								 first.y - 8,
								 batch.width + 16,
								 kBoxHeight + 16,
								 8,
								 "none",
								 kRuleStrong,
								 1,
								 "4,3");
					}
					for (const Box &box: batch.boxes)
					{
						svg.Rect(box.x, box.y, box.w, kBoxHeight, 6, kPaper);
						if (box.stub)
						{
							svg.Rect(box.x,
									 box.y,
									 box.w,
									 kBoxHeight,
									 6,
									 "rgba(245,245,245,0.02)",
									 "rgba(245,245,245,0.30)",
									 1,
									 "4,3");
						}
						else
						{
							svg.Rect(box.x, box.y, box.w, kBoxHeight, 6, kPaper2, kInk, 1);
						}
						if (!box.kindTag.empty())
						{
							svg.Tag(box.x + 8, box.y + 6, box.kindTag, kMuted);
						}
						if (!box.barrierTag.empty())
						{
							svg.Tag(box.x + box.w - 8 - (MonoWidth(box.barrierTag, 7, 0.08) + 8),
									box.y + 6,
									box.barrierTag,
									kSoft);
						}
						const double cx = box.x + box.w / 2.0;
						svg.Text(cx, box.y + 34, box.name, 12, kInk, kSans, "middle", 600);
						svg.Text(cx, box.y + 48, box.sublabel, 9, kMuted, kMono, "middle");
					}
				}
			}
		}

		svg.out << "<!-- legend -->\n";
		svg.Line(kMargin, legendY, width - kMargin, legendY, kRule, 0.8);
		svg.Text(kMargin, legendY + 16, "LEGEND", 8, kMuted, kMono, "start", 500, 0.18);
		const int ly = legendY + 32;
		svg.Rect(kMargin, ly, 16, 12, 2, kPaper2, kInk, 1);
		svg.Text(kMargin + 24, ly + 9, "Pass", 9, kMuted, kSans);
		svg.Rect(kMargin + 96, ly, 16, 12, 2, "none", kRuleStrong, 1, "4,3");
		svg.Text(kMargin + 120, ly + 9, "Passes sharing a batch", 9, kMuted, kSans);
		svg.Line(kMargin + 264, ly + 6, kMargin + 292, ly + 6, kMuted, 1.2, nullptr, "rg-schedule-arrow");
		svg.Text(kMargin + 300, ly + 9, "Queue order", 9, kMuted, kSans);
		svg.Line(kMargin + 400, ly + 6, kMargin + 428, ly + 6, kAccent, 1.4, nullptr, "rg-schedule-arrow-accent");
		svg.Text(kMargin + 436, ly + 9, "Waits on the other queue's submit", 9, kMuted, kSans);
		svg.Line(kMargin + 640, ly - 2, kMargin + 640, ly + 14, kRuleStrong, 1, "4,3");
		svg.Text(kMargin + 652, ly + 9, "Submit", 9, kMuted, kSans);

		// Transient memory: lifetime over global batches, and the heap range each resource occupies.
		std::vector<const ResourceNode *> transients;
		uint64_t unaliasedBytes = 0;
		for (const ResourceNode &node: resources_)
		{
			if (OwnsTransientMemory(node))
			{
				transients.push_back(&node);
				unaliasedBytes += node.size;
			}
		}
		std::sort(transients.begin(),
				  transients.end(),
				  [](const ResourceNode *a, const ResourceNode *b)
				  { return a->offset != b->offset ? a->offset < b->offset : a->firstBatch < b->firstBatch; });

		Svg memory;
		int memoryWidth = 0;
		int memoryHeight = 0;
		if (!transients.empty() && transientMemorySize_ > 0)
		{
			const uint32_t batchCount = sections_.back().globalStart + sections_.back().length;
			int nameColumn = 160;
			for (const ResourceNode *node: transients)
			{
				nameColumn = std::max(nameColumn, SansWidth(node->name, 12) + 16);
			}
			constexpr int kColumn = 32;
			constexpr int kRow = 40;
			constexpr int kRowsTop = 72;
			constexpr int kHeapWidth = 240;
			const int batchX = kMargin + nameColumn + 24;
			const int heapX = batchX + static_cast<int>(batchCount) * kColumn + 48;
			memoryWidth = std::max(heapX + kHeapWidth + kMargin, 880);
			memoryHeight = kRowsTop + static_cast<int>(transients.size()) * kRow + 24;
			const int rowsBottom = kRowsTop + static_cast<int>(transients.size()) * kRow;

			for (uint32_t s: visible)
			{
				const int sx = batchX + static_cast<int>(sections_[s].globalStart) * kColumn;
				const int sw = static_cast<int>(sections_[s].length) * kColumn;
				if (sw == 0)
				{
					continue;
				}
				memory.Text(sx + sw / 2.0, 36, "S" + std::to_string(s), 8, kMuted, kMono, "middle", 500, 0.18);
				memory.Line(sx, 44, sx, rowsBottom, kRuleStrong, 1, "4,3");
				for (uint32_t b = 0; b < sections_[s].length; ++b)
				{
					memory.Text(sx + b * kColumn + kColumn / 2.0,
								60,
								std::to_string(sections_[s].globalStart + b),
								8,
								kSoft,
								kMono,
								"middle");
				}
			}
			memory.Text(kMargin, 60, "BATCH", 8, kMuted, kMono, "start", 500, 0.18);
			memory.Text(heapX, 36, "HEAP RANGE", 8, kMuted, kMono, "start", 500, 0.18);
			memory.Text(heapX, 60, "0", 8, kSoft, kMono, "start");
			memory.Text(heapX + kHeapWidth, 60, FormatBytes(transientMemorySize_), 8, kSoft, kMono, "end");
			memory.Line(kMargin, kRowsTop, memoryWidth - kMargin, kRowsTop, kRuleStrong, 1);

			const double scale = static_cast<double>(kHeapWidth) / static_cast<double>(transientMemorySize_);
			for (size_t r = 0; r < transients.size(); ++r)
			{
				const ResourceNode &node = *transients[r];
				const int top = kRowsTop + static_cast<int>(r) * kRow;
				memory.Line(kMargin, top + kRow, memoryWidth - kMargin, top + kRow, kRule, 1);
				memory.Text(kMargin, top + 18, node.name, 12, kInk, kSans, "start", 600);
				memory.Text(kMargin,
							top + 31,
							FormatBytes(node.size) + " @ " + FormatBytes(node.offset),
							9,
							kMuted,
							kMono);
				// Opaque paper first, so the section lines stop at the bar.
				const int bx = batchX + static_cast<int>(node.firstBatch) * kColumn + 4;
				const int bw = static_cast<int>(node.lastBatch - node.firstBatch + 1) * kColumn - 8;
				memory.Rect(bx, top + 8, bw, 24, 4, kPaper);
				memory.Rect(bx, top + 8, bw, 24, 4, kStoreFill, kMuted, 1);
				memory.Rect(heapX, top + 19, kHeapWidth, 2, 1, kRule);
				const int hx = heapX + static_cast<int>(static_cast<double>(node.offset) * scale);
				const int hw = std::max(4, static_cast<int>(static_cast<double>(node.size) * scale + 0.5));
				memory.Rect(hx, top + 12, hw, 16, 2, kStoreFill, kMuted, 1);
			}
		}

		// Page.
		size_t queueSyncs = 0;
		for (const QueueWaitInfo &wait: queueWaits_)
		{
			queueSyncs += wait.waitingQueue != wait.signalQueue ? 1 : 0;
		}
		const uint32_t batchTotal = GetTotalBatchCount();
		const std::string escapedTitle = Escape(title);
		html << R"(<!DOCTYPE html>
<html lang="en" style="color-scheme: dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>)" << escapedTitle
			 << R"( · Render graph</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --paper: #2d3142; --paper-2: #393e53; --ink: #f5f5f5; --muted: #bfc0c0; --soft: #949eb2; --accent: #f08a59;
  --rule: rgba(245,245,245,0.12); --rule-strong: rgba(245,245,245,0.22);
  --serif: 'Instrument Serif', serif; --sans: 'Geist', system-ui, sans-serif; --mono: 'Geist Mono', ui-monospace, monospace;
}
body { font-family: var(--sans); background: var(--paper); color: var(--ink); padding: 3rem 2rem; }
.frame { max-width: 1200px; margin: 0 auto; }
.eyebrow { font-family: var(--mono); font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem; }
h1 { font-family: var(--serif); font-size: 2rem; font-weight: 400; letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 1.5rem; }
section { margin-top: 2.5rem; }
.figure { overflow-x: auto; }
.figure svg { display: block; width: 100%; height: auto; }
.cards { display: grid; grid-template-columns: 1.1fr 1fr 0.9fr 1.2fr; gap: 12px; }
.card { background: var(--paper-2); border: 1px solid var(--rule); border-radius: 6px; padding: 16px 20px; }
.card .value { font-family: var(--serif); font-size: 2rem; line-height: 1.1; }
.card .value.focal { color: var(--accent); }
.card .note { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
th { font-family: var(--mono); font-size: 0.62rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); text-align: left; padding: 8px 12px 8px 0; border-bottom: 1px solid var(--rule-strong); }
td { padding: 6px 12px 6px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
td.mono { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); }
td.name { font-weight: 600; }
p.empty { font-family: var(--mono); font-size: 0.75rem; color: var(--muted); }
</style>
</head>
<body>
<div class="frame">
<p class="eyebrow">Render graph · Compiled schedule</p>
<h1>)" << escapedTitle
			 << "</h1>\n<div class=\"cards\">\n";
		auto card = [&](const std::string &label, const std::string &value, const std::string &note, bool focal = false)
		{
			html << "<div class=\"card\"><p class=\"eyebrow\">" << Escape(label) << "</p><p class=\"value"
				 << (focal ? " focal" : "") << "\">" << Escape(value) << "</p><p class=\"note\">" << Escape(note)
				 << "</p></div>\n";
		};
		card("Passes",
			 std::to_string(passCount),
			 std::to_string(lanes.size()) + (lanes.size() == 1 ? " queue" : " queues"));
		card("Batches", std::to_string(batchTotal), std::to_string(visible.size()) + " sections");
		card("Barriers", std::to_string(barrierCount), std::to_string(queueSyncs) + " cross-queue waits");
		if (transientMemorySize_ > 0 && unaliasedBytes > 0)
		{
			const int saved = static_cast<int>(
					100.0 * (1.0 - static_cast<double>(transientMemorySize_) / static_cast<double>(unaliasedBytes)) +
					0.5);
			card("Transient memory",
				 FormatBytes(transientMemorySize_),
				 std::to_string(saved) + "% saved by aliasing (" + FormatBytes(unaliasedBytes) + " unaliased)",
				 true);
		}
		else
		{
			card("Transient memory", "0 B", "no transient resources");
		}
		html << "</div>\n";

		html << "<section>\n<p class=\"eyebrow\">Schedule</p>\n<div class=\"figure\">\n";
		OpenSvg(html,
				"rg-schedule",
				width,
				height,
				title + " schedule",
				"Swimlane of the compiled render graph: each queue is a lane, each section a column, passes run left "
				"to right, "
				"and accent arrows mark where one queue waits for the other.");
		html << svg.out.str() << "</svg>\n</div>\n</section>\n";

		html << "<section>\n<p class=\"eyebrow\">Transient memory</p>\n";
		if (memoryWidth > 0)
		{
			html << "<div class=\"figure\">\n";
			OpenSvg(html,
					"rg-memory",
					memoryWidth,
					memoryHeight,
					title + " transient memory",
					"Each transient resource's lifetime in batches, and the range of the shared heap it is placed in; "
					"resources whose lifetimes do not overlap share memory.");
			html << memory.out.str() << "</svg>\n</div>\n";
		}
		else
		{
			html << "<p class=\"empty\">No transient resources.</p>\n";
		}
		html << "</section>\n";

		html << "<section>\n<p "
				"class=\"eyebrow\">Barriers</p>\n<table>\n<tr><th>Section</th><th>Queue</th><th>Before</th>"
				"<th>Resource</th><th>Kind</th><th>From</th><th>To</th></tr>\n";
		auto row = [&](uint32_t s, uint32_t q, const std::string &before, const BarrierInfo &barrier)
		{
			html << "<tr><td class=\"mono\">" << s << "</td><td class=\"mono\">" << ToString(static_cast<QueueType>(q))
				 << "</td><td>" << Escape(before) << "</td><td class=\"name\">"
				 << Escape(resources_[barrier.resource.index].name) << "</td><td class=\"mono\">"
				 << BarrierKindName(barrier.kind) << "</td><td class=\"mono\">"
				 << Escape(barrier.beforeFromTrackedState ? std::string("tracked by the context")
														  : StateText(barrier.before))
				 << "</td><td class=\"mono\">" << Escape(StateText(barrier.after)) << "</td></tr>\n";
		};
		for (uint32_t s: visible)
		{
			for (uint32_t q: lanes)
			{
				const QueueSection &queueSection = sections_[s].queues[q];
				for (const CompiledBatch &batch: queueSection.batches)
				{
					const std::string before =
							batch.passes.empty() ? std::string() : passes_[batch.passes.front().index].desc.name;
					for (const BarrierInfo &barrier: batch.barriers)
					{
						row(s, q, before, barrier);
					}
				}
				for (const BarrierInfo &barrier: queueSection.releases)
				{
					row(s, q, "end of section", barrier);
				}
			}
		}
		html << "</table>\n</section>\n</div>\n</body>\n</html>\n";
		return html.str();
	}
} // namespace rg
