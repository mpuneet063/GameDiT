// GeneratedTerrain.h - the actor you drop in the level: owns the terrain mesh, the NNE generator,
// an overview camera for watching generation, the exported-PNG loader, and the keyboard controls.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "GeneratedTerrain.generated.h"   // must be the last include

class UProceduralMeshComponent;
class UCameraComponent;
class UMaterialInterface;
class UTerrainGenerator;
class APlayerController;

UCLASS()
class GAMEDIT_API AGeneratedTerrain : public AActor
{
    GENERATED_BODY()

public:
    AGeneratedTerrain();
    virtual void Tick(float DeltaSeconds) override;

    // [G] in-game: generate a brand-new terrain with the model, watched from the overview camera
    UFUNCTION(BlueprintCallable, Category = "GameDiT")
    void GenerateNew();

    // Load <Project>/Heightmaps/terrain_XX.png + .json (sample.py export output)
    UFUNCTION(BlueprintCallable, Category = "GameDiT")
    bool LoadExport(int32 Index);

    // [N] in-game: cycle through the exported heightmaps
    UFUNCTION(BlueprintCallable, Category = "GameDiT")
    void NextExport();

    // ---- Components ----
    UPROPERTY(VisibleAnywhere, Category = "GameDiT")
    TObjectPtr<UProceduralMeshComponent> Mesh;

    UPROPERTY(VisibleAnywhere, Category = "GameDiT")
    TObjectPtr<UCameraComponent> OverviewCamera;

    UPROPERTY(VisibleAnywhere, Category = "GameDiT")
    TObjectPtr<UTerrainGenerator> Generator;

    // ---- Settings ----
    UPROPERTY(EditAnywhere, Category = "GameDiT|Terrain")
    TObjectPtr<UMaterialInterface> Material;

    // Resolution of the final, walkable mesh (matches your 1009 export size)
    UPROPERTY(EditAnywhere, Category = "GameDiT|Terrain", meta = (ClampMin = "2"))
    int32 FinalRes = 1009;

    // Resolution of the live previews (no collision, rebuilt every few steps)
    UPROPERTY(EditAnywhere, Category = "GameDiT|Terrain", meta = (ClampMin = "2"))
    int32 PreviewRes = 257;

    UPROPERTY(EditAnywhere, Category = "GameDiT|Terrain", meta = (ClampMin = "1"))
    int32 Chunks = 8;

    // Smoothing after upsampling generated terrain, in final-mesh pixels (hides bilinear facets)
    UPROPERTY(EditAnywhere, Category = "GameDiT|Terrain", meta = (ClampMin = "0"))
    float BlurSigma = 2.0f;

    // Ground distance per model pixel; matches metres_per_pixel in gamedit_step.json
    UPROPERTY(EditAnywhere, Category = "GameDiT|Terrain")
    float NativeMetresPerPixel = 60.f;

    // Folder under the project root holding terrain_XX.png / .json
    UPROPERTY(EditAnywhere, Category = "GameDiT|Exports")
    FString ExportSubdir = TEXT("Heightmaps");

    // true: start on terrain_00.png; false (or no exports found): generate on start
    UPROPERTY(EditAnywhere, Category = "GameDiT|Exports")
    bool bStartWithExport = true;

protected:
    virtual void BeginPlay() override;

private:
    void HandleHeights(const TArray<float>& HeightsM, int32 N, bool bFinal);   // bound to Generator->OnHeights
    void BuildFinal(const TArray<float>& HeightsM, int32 N, float CellCm);
    float GeneratedExtentCm() const;
    void PointOverviewCamera(float ExtentCm, float ReliefCm);

    void TryFreezePlayer(APlayerController* PC);
    void TryPlacePlayer(APlayerController* PC);
    void HandleInput(APlayerController* PC);
    void ShowStatus() const;

    TArray<FString> ExportFiles;     // full paths, sorted
    int32 ExportIndex = -1;

    float SpawnHeightCm = 0.f;       // terrain height at the centre, where the player is placed
    bool bPendingFreeze = false;     // retried in Tick until the pawn exists
    bool bPendingPlacePlayer = false;
    int32 LastSeed = 0;
    FString Status;
};