set(RENDER_GRAPH_TEST_SOURCES
    TestCommon.cpp
    BatchingTests.cpp
    AsyncComputeTests.cpp
    AliasingTests.cpp
    ValidationTests.cpp
    AccessTests.cpp
    MipTests.cpp
)

add_executable(RenderGraphTests ${RENDER_GRAPH_TEST_SOURCES})
target_link_libraries(RenderGraphTests PRIVATE RenderGraph)
target_compile_definitions(RenderGraphTests PRIVATE RG_TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
# slang.dll, dxcompiler.dll and the Agility SDK runtime must sit next to the executable.
d3d12_deploy_runtime_dlls(RenderGraphTests)

# One CTest entry per test, each in its own process, so a failed assert only fails that test.
# Keep this list in sync with the RG_TEST(...) names (RenderGraphTests --list prints them).
set(RENDER_GRAPH_TESTS
    LinearChain
    IndependentPassesShareBatch
    ReadGroupMerging
    WriteAfterReadWaitsForReaders
    UavWriteAfterWriteNeedsBarrier
    ConcurrentWritesShareBatch
    ReorderingShortensCriticalPath
    BarriersPlacedAsLateAsPossible
    DepthTestReadsAndSamplingUseSeparateLayouts
    SubchannelSortingInsideBatch
    StaticGraphExecutesManyFrames
    AsyncComputeOverlapsGraphics
    AsyncComputeQueueConflictRejected
    AsyncComputeImportedResourceRoundTrip
    AsyncComputeTextureQueueTransfer
    HtmlDumpDrawsScheduleAndWaits
    TransientAliasingReusesMemory
    AliasingNeverOverlapsLiveMemoryRandomized
    AsyncSectionsDoNotAliasAcrossQueues
    WigglingShrinksTransientLifetimes
    InvalidGraphsAreRejected
    EveryAccessIsAValidBarrier
    MeshShaderPassOrdersReadsAndWrites
    RaytracingBuildsAndTracesInOrder
    BloomChainTransitionsEachMip
    RasterPassesDrawIntoMips
    ImportedTextureMipsEndInOneState
    InvalidMipUseIsRejected
)

foreach(test_name IN LISTS RENDER_GRAPH_TESTS)
    add_test(NAME ${test_name} COMMAND RenderGraphTests ${test_name})
    # Tests share one GPU and open a window each; running them one at a time keeps timings honest.
    set_tests_properties(${test_name} PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)
endforeach()
