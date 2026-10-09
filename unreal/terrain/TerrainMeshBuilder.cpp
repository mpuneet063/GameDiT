// TerrainMeshBuilder.cpp - heights (metres) -> walkable ProceduralMeshComponent
// Grids are N x N, stored row-major like a flattened numpy array: index = Row * N + Col

#include "TerrainMeshBuilder.h"
#include "ProceduralMeshComponent.h"
#include "Materials/MaterialInterface.h"

// ---------------------------------------------------------------------------
// Bilinear resize, corner-aligned (same as F.interpolate(..., align_corners=True))
// ---------------------------------------------------------------------------
TArray<float> FTerrainMeshBuilder::Resample(const TArray<float>& Src, int32 SrcN, int32 DstN)
{
    if (SrcN < 2 || DstN < 2 || Src.Num() != SrcN * SrcN)
    {
        UE_LOG(LogTemp, Warning, TEXT("Resample: bad input (SrcN=%d, DstN=%d, Num=%d)"), SrcN, DstN, Src.Num());
        return {};
    }

    TArray<float> Out;
    Out.SetNumUninitialized(DstN * DstN);              // like np.empty(DstN * DstN)

    const float Scale = float(SrcN - 1) / float(DstN - 1);

    for (int32 R = 0; R < DstN; ++R)
    {
        const float SY = R * Scale;                     // where this row falls in the source grid
        const int32 Y0 = FMath::Min(FMath::FloorToInt(SY), SrcN - 1);
        const int32 Y1 = FMath::Min(Y0 + 1, SrcN - 1);
        const float FY = SY - Y0;                       // fraction between rows Y0 and Y1

        for (int32 C = 0; C < DstN; ++C)
        {
            const float SX = C * Scale;
            const int32 X0 = FMath::Min(FMath::FloorToInt(SX), SrcN - 1);
            const int32 X1 = FMath::Min(X0 + 1, SrcN - 1);
            const float FX = SX - X0;

            const float Top = FMath::Lerp(Src[Y0 * SrcN + X0], Src[Y0 * SrcN + X1], FX);
            const float Bot = FMath::Lerp(Src[Y1 * SrcN + X0], Src[Y1 * SrcN + X1], FX);
            Out[R * DstN + C] = FMath::Lerp(Top, Bot, FY);
        }
    }
    return Out;
}

// ---------------------------------------------------------------------------
// Separable Gaussian blur (your gaussian_blur, but edges are clamped, not reflected)
// ---------------------------------------------------------------------------
TArray<float> FTerrainMeshBuilder::Blur(const TArray<float>& Src, int32 N, float Sigma)
{
    if (Sigma <= 0.f || N < 2 || Src.Num() != N * N)
    {
        return Src;                                     // returns a copy, so the caller's array is untouched
    }

    // 1D kernel, normalised to sum to 1
    const int32 Radius = FMath::CeilToInt(3.f * Sigma);
    TArray<float> Kernel;
    Kernel.SetNumUninitialized(2 * Radius + 1);
    float Sum = 0.f;
    for (int32 K = -Radius; K <= Radius; ++K)
    {
        const float W = FMath::Exp(-float(K * K) / (2.f * Sigma * Sigma));
        Kernel[K + Radius] = W;
        Sum += W;
    }
    for (float& W : Kernel)                             // '&' = edit the element in place
    {
        W /= Sum;
    }

    // Horizontal pass: Src -> Tmp
    TArray<float> Tmp;
    Tmp.SetNumUninitialized(N * N);
    for (int32 R = 0; R < N; ++R)
    {
        for (int32 C = 0; C < N; ++C)
        {
            float Acc = 0.f;
            for (int32 K = -Radius; K <= Radius; ++K)
            {
                const int32 CC = FMath::Clamp(C + K, 0, N - 1);
                Acc += Kernel[K + Radius] * Src[R * N + CC];
            }
            Tmp[R * N + C] = Acc;
        }
    }

    // Vertical pass: Tmp -> Out
    TArray<float> Out;
    Out.SetNumUninitialized(N * N);
    for (int32 R = 0; R < N; ++R)
    {
        for (int32 C = 0; C < N; ++C)
        {
            float Acc = 0.f;
            for (int32 K = -Radius; K <= Radius; ++K)
            {
                const int32 RR = FMath::Clamp(R + K, 0, N - 1);
                Acc += Kernel[K + Radius] * Tmp[RR * N + C];
            }
            Out[R * N + C] = Acc;
        }
    }
    return Out;
}

// ---------------------------------------------------------------------------
// Heights (metres) -> Chunks x Chunks mesh sections on Mesh
// ---------------------------------------------------------------------------
void FTerrainMeshBuilder::Build(UProceduralMeshComponent* Mesh, const TArray<float>& HeightsM, int32 N,
                                float CellSizeCm, int32 Chunks, bool bCollision, UMaterialInterface* Material)
{
    if (!Mesh || N < 2 || HeightsM.Num() != N * N || Chunks < 1 || CellSizeCm <= 0.f)
    {
        UE_LOG(LogTemp, Warning, TEXT("Build: bad input (N=%d, Num=%d, Chunks=%d)"), N, HeightsM.Num(), Chunks);
        return;
    }
    Chunks = FMath::Min(Chunks, N - 1);                 // can't have more chunks than quads

    Mesh->ClearAllMeshSections();

    // Lowest point sits at Z = 0
    float MinH = HeightsM[0];
    for (const float H : HeightsM)
    {
        MinH = FMath::Min(MinH, H);
    }

    // Centre the terrain on the actor's origin
    const float Half = (N - 1) * CellSizeCm * 0.5f;

    // 1. Positions for the whole grid (UE units are cm, so metres * 100)
    TArray<FVector> Pos;
    Pos.SetNumUninitialized(N * N);
    for (int32 R = 0; R < N; ++R)
    {
        for (int32 C = 0; C < N; ++C)
        {
            Pos[R * N + C] = FVector(C * CellSizeCm - Half,
                                     R * CellSizeCm - Half,
                                     (HeightsM[R * N + C] - MinH) * 100.f);
        }
    }

    // 2. Normals for the whole grid, from height differences (central where possible, one-sided at edges).
    //    Computing them globally, not per chunk, means chunk borders get identical normals: no lighting seams.
    auto Z = [&](int32 R, int32 C) { return Pos[R * N + C].Z; };   // small helper, like a Python lambda

    TArray<FVector> Nrm;
    Nrm.SetNumUninitialized(N * N);
    for (int32 R = 0; R < N; ++R)
    {
        for (int32 C = 0; C < N; ++C)
        {
            const int32 CL = FMath::Max(C - 1, 0), CR = FMath::Min(C + 1, N - 1);
            const int32 RD = FMath::Max(R - 1, 0), RU = FMath::Min(R + 1, N - 1);
            const float DzDx = (Z(R, CR) - Z(R, CL)) / ((CR - CL) * CellSizeCm);
            const float DzDy = (Z(RU, C) - Z(RD, C)) / ((RU - RD) * CellSizeCm);
            Nrm[R * N + C] = FVector(-DzDx, -DzDy, 1.f).GetSafeNormal();
        }
    }

    // CreateMeshSection wants arrays for these even when unused
    const TArray<FLinearColor> NoColours;
    const TArray<FProcMeshTangent> NoTangents;

    // 3. One mesh section per chunk. Neighbouring chunks share their border row/column of vertices.
    for (int32 CY = 0; CY < Chunks; ++CY)
    {
        for (int32 CX = 0; CX < Chunks; ++CX)
        {
            const int32 R0 = CY * (N - 1) / Chunks, R1 = (CY + 1) * (N - 1) / Chunks;   // integer division
            const int32 C0 = CX * (N - 1) / Chunks, C1 = (CX + 1) * (N - 1) / Chunks;
            const int32 W = C1 - C0 + 1;                // vertices per chunk row
            const int32 H = R1 - R0 + 1;                // vertices per chunk column

            TArray<FVector> Verts;  Verts.Reserve(W * H);   // Reserve = allocate capacity, size stays 0
            TArray<FVector> Norms;  Norms.Reserve(W * H);
            TArray<FVector2D> UV0;  UV0.Reserve(W * H);
            for (int32 R = R0; R <= R1; ++R)
            {
                for (int32 C = C0; C <= C1; ++C)
                {
                    Verts.Add(Pos[R * N + C]);
                    Norms.Add(Nrm[R * N + C]);
                    UV0.Add(FVector2D(C / float(N - 1), R / float(N - 1)));   // 0..1 across the whole terrain
                }
            }

            // Two triangles per quad. A = this vertex, B = +X neighbour, D = +Y neighbour, E = diagonal.
            TArray<int32> Tris;
            Tris.Reserve((W - 1) * (H - 1) * 6);
            for (int32 I = 0; I < H - 1; ++I)
            {
                for (int32 J = 0; J < W - 1; ++J)
                {
                    const int32 A = I * W + J;
                    const int32 B = A + 1;
                    const int32 D = A + W;
                    const int32 E = D + 1;
                    Tris.Append({A, D, B, B, D, E});    // if terrain is only visible from below: swap B and D
                }
            }

            const int32 Section = CY * Chunks + CX;
            Mesh->CreateMeshSection_LinearColor(Section, Verts, Tris, Norms, UV0, NoColours, NoTangents, bCollision);
            if (Material)
            {
                Mesh->SetMaterial(Section, Material);
            }
        }
    }
}
