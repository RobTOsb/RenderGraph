// Ray traced AO: a primary ray per pixel finds the surface, then cosine weighted hemisphere rays from it test
// occlusion, all with inline ray queries against the scene TLAS. It needs nothing from the raster passes, so
// it runs on async compute beside them. Same output and 4x4 interleaved rotation as SSAO, so csBlur denoises it.
#include "common.slang"

// Matches Vertex in scene.slang and main.cpp.
struct SceneVertex
{
	float3 position;
	float3 normal;
	float3 albedo;
	float emissive;
};

StructuredBuffer<SceneVertex> g_Vertices;
StructuredBuffer<uint> g_Indices; // the BLAS triangles, three per primitive
RaytracingAccelerationStructure g_Scene;
RWTexture2D<float4> g_Surface; // what the blur reads: xyz world space normal, w reversed Z depth (0 for background)
RWTexture2D<float> g_Out;

static const uint kRayCount = 8; // per pixel; the blur averages 16 pixels, so 128 directions in all
static const float kPi = 3.14159265f;

typedef RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> SceneQuery;

// The surface the rasterizer shows at the pixel centre: the camera ray from the near plane, as raster clipping has it.
bool TracePrimary(int2 pixel, out float3 position, out float3 normal, out float depth)
{
	position = 0.0f;
	normal = 0.0f;
	depth = 0.0f;
	const float3 nearPoint = ReconstructWorld(float2(pixel) + 0.5f, 1.0f); // reversed Z: 1 is the near plane
	RayDesc ray;
	ray.Origin = nearPoint;
	ray.Direction = normalize(nearPoint - g_Frame.cameraPos);
	ray.TMin = 0.0f;
	ray.TMax = g_Frame.farZ;

	SceneQuery query;
	query.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
	query.Proceed();
	if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
	{
		return false;
	}

	// Interpolated vertex normals, like the G-buffer's.
	const uint first = query.CommittedPrimitiveIndex() * 3;
	const float2 bary = query.CommittedTriangleBarycentrics();
	normal = normalize(g_Vertices[g_Indices[first]].normal * (1.0f - bary.x - bary.y) +
					   g_Vertices[g_Indices[first + 1]].normal * bary.x + g_Vertices[g_Indices[first + 2]].normal * bary.y);
	position = ray.Origin + ray.Direction * query.CommittedRayT();
	const float4 clip = mul(float4(position, 1.0f), g_Frame.viewProj);
	depth = clip.z / clip.w;
	return true;
}

float ComputeAO(int2 pixel)
{
	float3 position;
	float3 n;
	float depth;
	if (!TracePrimary(pixel, position, n, depth))
	{
		g_Surface[pixel] = 0.0f;
		return 1.0f; // background
	}
	g_Surface[pixel] = float4(n, depth);

	// Pushed off the surface so rays never hit the triangle they start on.
	const float3 origin = position + n * 2e-3f;
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

		SceneQuery query;
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
