// Data[i] += value (read-modify-write through a UAV)
struct PushConstants
{
	uint value;
	uint count;
	uint2 pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> g_PushConstants;

RWStructuredBuffer<uint> Data : register(u0);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
	if (id.x < g_PushConstants.count)
	{
		Data[id.x] += g_PushConstants.value;
	}
}
