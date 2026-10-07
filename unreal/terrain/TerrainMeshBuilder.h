// TerrainGenerator.h - runs the exported GameDiT Heun step (gamedit_step.onnx) inside Unreal via NNE
#pragma once

#include <atomic>
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Async/Future.h"
#include "NNERuntimeRunSync.h"
#include "TerrainGenerator.generated.h"   // must be the last include (Unreal's code generator needs this)

class UNNEModelData;

// Heights in metres (N x N, row-major). bFinal = false for live previews, true for the finished terrain.
DECLARE_MULTICAST_DELEGATE_ThreeParams(FOnTerrainHeights, const TArray<float>& /*HeightsM*/, int32 /*N*/, bool /*bFinal*/);

UCLASS(ClassGroup = (GameDiT), meta = (BlueprintSpawnableComponent))
class GAMEDIT_API UTerrainGenerator : public UActorComponent
{
    GENERATED_BODY()

public:
    UTerrainGenerator();

    // ---- Settings (editable in the Details panel) ----

    // The imported gamedit_step.onnx asset
    UPROPERTY(EditAnywhere, Category = "GameDiT|Model")
    TObjectPtr<UNNEModelData> ModelData;

    // Tried first (DirectML on the GPU), then the CPU runtime as a fallback
    UPROPERTY(EditAnywhere, Category = "GameDiT|Model")
    FString GpuRuntimeName = TEXT("NNERuntimeORTDml");

    UPROPERTY(EditAnywhere, Category = "GameDiT|Model")
    FString CpuRuntimeName = TEXT("NNERuntimeORTCpu");

    UPROPERTY(EditAnywhere, Category = "GameDiT|Sampling", meta = (ClampMin = "1"))
    int32 Steps = 50;

    UPROPERTY(EditAnywhere, Category = "GameDiT|Sampling")
    float Guidance = 2.0f;

    // Relief condition y in [0, 1] (log-scaled height range, as in data.py)
    UPROPERTY(EditAnywhere, Category = "GameDiT|Sampling", meta = (ClampMin = "0", ClampMax = "1"))
    float Relief = 0.7f;

    // Send a preview to the mesh every N sampler steps
    UPROPERTY(EditAnywhere, Category = "GameDiT|Preview", meta = (ClampMin = "1"))
    int32 PreviewEvery = 2;

    // true: show x1_hat (the model's guess of the final terrain); false: show raw x_t
    UPROPERTY(EditAnywhere, Category = "GameDiT|Preview")
    bool bPreviewPrediction = true;

    // Must match relief_min_m / relief_max_m in gamedit_step.json
    UPROPERTY(EditAnywhere, Category = "GameDiT|Data")
    float ReliefMinM = 150.f;

    UPROPERTY(EditAnywhere, Category = "GameDiT|Data")
    float ReliefMaxM = 4000.f;

    // ---- API ----

    // Starts a generation on a background thread. Returns false if busy or the model isn't loaded.
    bool Generate(int32 Seed);

    bool IsReady() const { return Instance.IsValid(); }
    bool IsBusy() const { return bBusy.load(); }
    int32 GetCurrentStep() const { return CurrentStep.load(); }
    int32 GetImgSize() const { return ImgSize; }
    const FString& GetActiveRuntime() const { return ActiveRuntime; }

    // y -> relief in metres, same formula as y_to_relief in data.py
    float GetReliefMetres() const;

    // Fired on the game thread with each preview and with the final terrain
    FOnTerrainHeights OnHeights;

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    bool InitModel();

    // CPU and GPU instances share this interface, so the sampling code doesn't care which one it got
    TSharedPtr<UE::NNE::IModelInstanceRunSync> Instance;
    FString ActiveRuntime;
    int32 ImgSize = 256;                  // read from the model's input shape at load time

    // Shared between the game thread and the worker thread, hence atomic
    std::atomic<bool> bBusy{false};
    std::atomic<bool> bCancel{false};
    std::atomic<int32> CurrentStep{0};

    TFuture<void> Task;                   // the running background generation
};