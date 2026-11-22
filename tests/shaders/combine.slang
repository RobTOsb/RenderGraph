// Output[i] = InputA[i] + InputB[i]
struct PushConstants
{
	uint count;
	uint3 pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> g_PushConstants;

StructuredBuffer<uint> InputA : register(t0);
StructuredBuffer<uint> InputB : register(t1);
RWStructuredBuffer<uint> Output : register(u0);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
	if (id.x < g_PushConstants.count)
	{
		Output[id.x] = InputA[id.x] + InputB[id.x];
	}
}
