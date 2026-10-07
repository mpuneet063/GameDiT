// TerrainGenerator.cpp - NNE model loading + the 50-step sampling loop (mirrors sample.py's generate())

#include "TerrainGenerator.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "Async/Async.h"
#include "Math/RandomStream.h"

namespace
{
    // NNE passes tensors as (pointer, size in bytes) pairs
    UE::NNE::FTensorBindingCPU Bind(void* Data, uint64 Bytes)
    {
        UE::NNE::FTensorBindingCPU Binding;
        Binding.Data = Data;
        Binding.SizeInBytes = Bytes;
        return Binding;
    }
}

UTerrainGenerator::UTerrainGenerator()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UTerrainGenerator::BeginPlay()
{
    Super::BeginPlay();
    InitModel();
}

void UTerrainGenerator::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    // Stop the worker and wait for it, so it never touches this component after it's gone
    bCancel = true;
    if (Task.IsValid())
    {
        Task.Wait();
    }
    Super::EndPlay(EndPlayReason);
}

float UTerrainGenerator::GetReliefMetres() const
{
    const float Lo = FMath::Loge(ReliefMinM);
    const float Hi = FMath::Loge(ReliefMaxM);
    return FMath::Exp(Lo + Relief * (Hi - Lo));
}

bool UTerrainGenerator::InitModel()
{
    // Printed to the Output Log: shows the exact runtime names your engine version registers
    for (const FString& Name : UE::NNE::GetAllRuntimeNames())
    {
        UE_LOG(LogTemp, Log, TEXT("GameDiT: NNE runtime available: %s"), *Name);
    }

    if (!ModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("GameDiT: ModelData not set - assign the imported gamedit_step asset"));
        return false;
    }

    // 1. GPU runtime first
    TWeakInterfacePtr<INNERuntimeGPU> Gpu = UE::NNE::GetRuntime<INNERuntimeGPU>(GpuRuntimeName);
    if (Gpu.IsValid())
    {
        TSharedPtr<UE::NNE::IModelGPU> Model = Gpu->CreateModelGPU(ModelData);
        if (Model.IsValid())
        {
            Instance = Model->CreateModelInstanceGPU();
            if (Instance.IsValid())
            {
                ActiveRuntime = GpuRuntimeName;
            }
        }
    }

    // 2. CPU fallback
    if (!Instance.IsValid())
    {
        TWeakInterfacePtr<INNERuntimeCPU> Cpu = UE::NNE::GetRuntime<INNERuntimeCPU>(CpuRuntimeName);
        if (Cpu.IsValid())
        {
            TSharedPtr<UE::NNE::IModelCPU> Model = Cpu->CreateModelCPU(ModelData);
            if (Model.IsValid())
            {
                Instance = Model->CreateModelInstanceCPU();
                if (Instance.IsValid())
                {
                    ActiveRuntime = CpuRuntimeName;
                }
            }
        }
    }

    if (!Instance.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("GameDiT: could not create a model instance on %s or %s"),
               *GpuRuntimeName, *CpuRuntimeName);
        return false;
    }

    // 3. Input shapes. The export used fixed shapes, so take them straight from the model:
    //    x [1,1,H,W], t [1], h [1], y [1], w [1]
    TArray<UE::NNE::FTensorShape> Shapes;
    for (const UE::NNE::FTensorDesc& Desc : Instance->GetInputTensorDescs())
    {
        Shapes.Add(UE::NNE::FTensorShape::MakeFromSymbolic(Desc.GetShape()));
    }
    if (Shapes.Num() != 5 ||
        Instance->SetInputTensorShapes(Shapes) != UE::NNE::IModelInstanceRunSync::ESetInputTensorShapesStatus::Ok)
    {
        UE_LOG(LogTemp, Error, TEXT("GameDiT: unexpected model inputs (expected x, t, h, y, w)"));
        Instance.Reset();
        return false;
    }

    ImgSize = int32(Shapes[0].GetData()[3]);   // W of x
    UE_LOG(LogTemp, Log, TEXT("GameDiT: model ready on %s, %dx%d"), *ActiveRuntime, ImgSize, ImgSize);
    return true;
}

bool UTerrainGenerator::Generate(int32 Seed)
{
    if (!Instance.IsValid() || bBusy)
    {
        return false;
    }
    bBusy = true;
    bCancel = false;
    CurrentStep = 0;

    // Copy every setting now, so changing them in-game mid-generation can't affect this run
    const int32 NumSteps = Steps;
    const float W = Guidance;
    const float Y = Relief;
    const int32 Every = FMath::Max(1, PreviewEvery);
    const bool bPred = bPreviewPrediction;
    const float ReliefM = GetReliefMetres();
    const int32 N = ImgSize;
    TSharedPtr<UE::NNE::IModelInstanceRunSync> Inst = Instance;
    TWeakObjectPtr<UTerrainGenerator> WeakThis(this);   // safe handle to check from the game thread later

    // Everything inside this lambda runs on a background thread, so the game keeps rendering
    Task = Async(EAsyncExecution::Thread, [this, WeakThis, Inst, Seed, NumSteps, W, Y, Every, bPred, ReliefM, N]()
    {
        const int32 Count = N * N;
        const uint64 GridBytes = uint64(Count) * sizeof(float);

        // 1. Gaussian noise x0 (Box-Muller). Different generator from torch, so seeds won't match Python.
        TArray<float> X;
        X.SetNumUninitialized(Count);
        FRandomStream Rng(Seed);
        for (int32 i = 0; i < Count; i += 2)
        {
            const float U1 = FMath::Max(Rng.FRand(), 1e-7f);   // avoid log(0)
            const float U2 = Rng.FRand();
            const float R = FMath::Sqrt(-2.f * FMath::Loge(U1));
            X[i] = R * FMath::Cos(2.f * UE_PI * U2);
            if (i + 1 < Count)
            {
                X[i + 1] = R * FMath::Sin(2.f * UE_PI * U2);
            }
        }

        TArray<float> XNext, X1Hat;
        XNext.SetNumUninitialized(Count);
        X1Hat.SetNumUninitialized(Count);

        float T = 0.f, H = 0.f, YIn = Y, WIn = W;   // scalar inputs, each a [1] tensor
        bool bPostedFinal = false;

        // Heights in metres, same as (x + 1) / 2 * relief_m in cmd_export
        auto ToMetres = [ReliefM](const TArray<float>& Src)
        {
            TArray<float> Out;
            Out.SetNumUninitialized(Src.Num());
            for (int32 i = 0; i < Src.Num(); ++i)
            {
                Out[i] = (Src[i] + 1.f) * 0.5f * ReliefM;
            }
            return Out;
        };

        // 2. Sampling loop: one ONNX call = one full Heun step with CFG
        for (int32 Step = 0; Step < NumSteps && !bCancel; ++Step)
        {
            T = float(Step) / NumSteps;                 // same uniform grid as make_time_grid
            H = float(Step + 1) / NumSteps - T;

            // Order must match the export: inputs x, t, h, y, w; outputs x_next, x1_hat
            const TArray<UE::NNE::FTensorBindingCPU> Inputs = {
                Bind(X.GetData(), GridBytes), Bind(&T, sizeof(float)), Bind(&H, sizeof(float)),
                Bind(&YIn, sizeof(float)), Bind(&WIn, sizeof(float))
            };
            const TArray<UE::NNE::FTensorBindingCPU> Outputs = {
                Bind(XNext.GetData(), GridBytes), Bind(X1Hat.GetData(), GridBytes)
            };

            if (Inst->RunSync(Inputs, Outputs) != UE::NNE::IModelInstanceRunSync::ERunSyncStatus::Ok)
            {
                UE_LOG(LogTemp, Error, TEXT("GameDiT: inference failed at step %d"), Step);
                break;
            }

            Swap(X, XNext);                              // x <- x_next (swaps buffers, no copy)
            CurrentStep = Step + 1;

            const bool bLast = (Step == NumSteps - 1);
            if (bLast || (Step + 1) % Every == 0)
            {
                // Final terrain always uses x; previews use x1_hat or x_t
                TArray<float> Heights = ToMetres((bLast || !bPred) ? X : X1Hat);

                // Hand the heights to the game thread (only it may touch meshes)
                AsyncTask(ENamedThreads::GameThread, [WeakThis, Payload = MoveTemp(Heights), N, bLast]()
                {
                    if (UTerrainGenerator* Self = WeakThis.Get())
                    {
                        Self->OnHeights.Broadcast(Payload, N, bLast);
                        if (bLast)
                        {
                            Self->bBusy = false;         // only free once the final terrain is delivered
                        }
                    }
                });
                bPostedFinal = bPostedFinal || bLast;
            }
        }

        if (!bPostedFinal)
        {
            bBusy = false;                               // cancelled or failed
        }
    });

    return true;
}