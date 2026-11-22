// Output[offset + i] = value + i
struct PushConstants
{
	uint value;
	uint count;
	uint offset;
	uint pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> g_PushConstants;

RWStructuredBuffer<uint> Output : register(u0);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
	if (id.x < g_PushConstants.count)
	{
		Output[g_PushConstants.offset + id.x] = g_PushConstants.value + id.x;
	}
}
