// TerrainMeshBuilder.h - heights (metres) -> walkable ProceduralMeshComponent
#pragma once

#include "CoreMinimal.h"

class UProceduralMeshComponent;
class UMaterialInterface;

struct GAMEDIT_API FTerrainMeshBuilder
{
    // Bilinear resize of an N x N grid (row-major), same corner alignment as align_corners=True
    static TArray<float> Resample(const TArray<float>& Src, int32 SrcN, int32 DstN);

    // Separable Gaussian blur; returns Src unchanged if Sigma <= 0
    static TArray<float> Blur(const TArray<float>& Src, int32 N, float Sigma);

    // Heights in metres (N x N, row-major) -> mesh sections on Mesh
    static void Build(UProceduralMeshComponent* Mesh, const TArray<float>& HeightsM, int32 N,
                      float CellSizeCm, int32 Chunks, bool bCollision, UMaterialInterface* Material);
};
