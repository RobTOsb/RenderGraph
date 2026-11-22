// Output[i] = Input[i] + value, plus (Lcg(i, iterations) & 0xF) when iterations > 0
// The LCG loop makes the pass slow on purpose, so a missing barrier or queue wait shows up as wrong data.
struct PushConstants
{
	uint value;
	uint count;
	uint iterations;
	uint pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> g_PushConstants;

StructuredBuffer<uint> Input : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
	if (id.x >= g_PushConstants.count)
	{
		return;
	}
	uint h = id.x;
	for (uint k = 0; k < g_PushConstants.iterations; ++k)
	{
		h = h * 1664525u + 1013904223u;
	}
	Output[id.x] = Input[id.x] + g_PushConstants.value + (g_PushConstants.iterations > 0 ? (h & 0xFu) : 0u);
}
