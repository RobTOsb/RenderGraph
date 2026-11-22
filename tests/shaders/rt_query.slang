// The same 8x8 grid of rays as rt_trace.slang, traced with an inline ray query: a hit writes 1, a miss 2.

RaytracingAccelerationStructure Scene;
RWStructuredBuffer<uint> Output;

[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
	RayDesc ray;
	ray.Origin = float3((float2(id.xy) + 0.5f) / 8.0f, -1.0f);
	ray.Direction = float3(0.0f, 0.0f, 1.0f);
	ray.TMin = 0.0f;
	ray.TMax = 10.0f;

	RayQuery<RAY_FLAG_NONE> query;
	query.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
	query.Proceed();
	Output[id.y * 8 + id.x] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1 : 2;
}
