// Ray traced AO: cosine weighted hemisphere rays against the scene TLAS with inline ray queries.
// Same inputs, output and 4x4 interleaved rotation as SSAO, so csBlur in ssao.slang denoises it too.
#include "common.slang"

Texture2D<float4> g_Input; // xyz world space normal, w reversed Z depth (0 where nothing was drawn)
RaytracingAccelerationStructure g_Scene;
RWTexture2D<float> g_Out;

static const uint kRayCount = 8; // per pixel; the blur averages 16 pixels, so 128 directions in all
static const float kPi = 3.14159265f;

float ComputeAO(int2 pixel)
{
	const float4 input = g_Input.Load(int3(pixel, 0));
	const float depth = input.w;
	if (!(depth > 0.0f))
	{
		return 1.0f; // background
	}
	const float normalLengthSq = dot(input.xyz, input.xyz);
	if (!(normalLengthSq > 1e-8f))
	{
		return 1.0f;
	}
	const float3 n = input.xyz * rsqrt(normalLengthSq);
	// Pushed off the surface so rays never hit the triangle they start on.
	const float3 origin = ReconstructWorld(float2(pixel) + 0.5f, depth) + n * 2e-3f;
	const float3 helper = abs(n.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
	const float3 t = normalize(cross(helper, n));
	const float3 b = cross(n, t);

	const uint tile = (uint(pixel.x) & 3u) + (uint(pixel.y) & 3u) * 4u;
	const float rotation = (float(tile) + 0.5f) / 16.0f * 2.0f * kPi;
	const float cosR = cos(rotation);
	const float sinR = sin(rotation);
	const float radius = g_Frame.rtaoRadius;

	float occlusion = 0.0f;
	for (uint i = 0; i < kRayCount; ++i)
	{
		const float u1 = (float(i) + 0.5f) / float(kRayCount);
		const float u2 = float(reversebits(i)) * 2.3283064365386963e-10f;
		const float r = sqrt(u1);
		const float phi = 2.0f * kPi * u2;
		const float2 disk = float2(r * cos(phi), r * sin(phi));
		const float2 rotated = float2(disk.x * cosR - disk.y * sinR, disk.x * sinR + disk.y * cosR);

		RayDesc ray;
		ray.Origin = origin;
		ray.Direction = t * rotated.x + b * rotated.y + n * sqrt(max(0.0f, 1.0f - u1));
		ray.TMin = 0.0f;
		ray.TMax = radius;

		RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
		query.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
		query.Proceed();
		if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
		{
			// Near hits occlude fully, fading out towards the radius so the AO has no hard edge.
			const float distance = query.CommittedRayT() / radius;
			occlusion += 1.0f - distance * distance;
		}
	}
	const float ao = saturate(1.0f - occlusion / float(kRayCount));
	return isnan(ao) ? 1.0f : ao;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csRTAO(uint3 id: SV_DispatchThreadID)
{
	if (all(id.xy < uint2(g_Frame.size)))
		g_Out[id.xy] = ComputeAO(int2(id.xy));
}
