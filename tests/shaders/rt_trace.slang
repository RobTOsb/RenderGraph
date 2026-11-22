// One ray per cell of an 8x8 grid over [0,1]^2, straight along +z through the triangle at z = 0.
// A hit writes Params[0], a miss writes 2.

RaytracingAccelerationStructure Scene;
StructuredBuffer<uint> Params;
RWStructuredBuffer<uint> Output;

struct Payload
{
	uint hit;
};

[shader("raygeneration")]
void rayGen()
{
	const uint2 cell = DispatchRaysIndex().xy;
	RayDesc ray;
	ray.Origin = float3((float2(cell) + 0.5f) / 8.0f, -1.0f);
	ray.Direction = float3(0.0f, 0.0f, 1.0f);
	ray.TMin = 0.0f;
	ray.TMax = 10.0f;

	Payload payload;
	payload.hit = 0;
	TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);
	Output[cell.y * 8 + cell.x] = payload.hit != 0 ? Params[0] : 2;
}

[shader("miss")]
void miss(inout Payload payload)
{
	payload.hit = 0;
}

[shader("closesthit")]
void closestHit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
	payload.hit = 1;
}
