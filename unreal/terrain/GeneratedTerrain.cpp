// GeneratedTerrain.cpp

#include "GeneratedTerrain.h"
#include "TerrainGenerator.h"
#include "TerrainMeshBuilder.h"

#include "ProceduralMeshComponent.h"
#include "Camera/CameraComponent.h"
#include "Engine/CollisionProfile.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/PlayerController.h"
#include "InputCoreTypes.h"

#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

AGeneratedTerrain::AGeneratedTerrain()
{
    PrimaryActorTick.bCanEverTick = true;

    Mesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("Mesh"));
    SetRootComponent(Mesh);
    Mesh->bUseAsyncCooking = false;   // collision exists as soon as Build() returns (brief hitch on the final build)
    Mesh->SetCollisionProfileName(UCollisionProfile::BlockAll_ProfileName);

    OverviewCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("OverviewCamera"));
    OverviewCamera->SetupAttachment(Mesh);

    Generator = CreateDefaultSubobject<UTerrainGenerator>(TEXT("Generator"));
}

void AGeneratedTerrain::BeginPlay()
{
    Super::BeginPlay();   // also runs Generator's BeginPlay, which loads the model

    Generator->OnHeights.AddUObject(this, &AGeneratedTerrain::HandleHeights);

    // Find <Project>/Heightmaps/terrain_*.png
    const FString Dir = FPaths::Combine(FPaths::ProjectDir(), ExportSubdir);
    IFileManager::Get().FindFiles(ExportFiles, *FPaths::Combine(Dir, TEXT("terrain_*.png")), true, false);
    ExportFiles.Sort();
    for (FString& File : ExportFiles)
    {
        File = FPaths::Combine(Dir, File);   // FindFiles returns bare names
    }
    UE_LOG(LogTemp, Log, TEXT("GameDiT: found %d exported heightmaps in %s"), ExportFiles.Num(), *Dir);

    if (bStartWithExport && ExportFiles.Num() > 0)
    {
        LoadExport(0);
    }
    else
    {
        GenerateNew();
    }
}

void AGeneratedTerrain::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    APlayerController* PC = GetWorld()->GetFirstPlayerController();
    if (!PC)
    {
        return;
    }
    if (bPendingFreeze)
    {
        TryFreezePlayer(PC);
    }
    if (bPendingPlacePlayer)
    {
        TryPlacePlayer(PC);
    }
    HandleInput(PC);
    ShowStatus();
}

// ---------------------------------------------------------------------------
// Generation with the model
// ---------------------------------------------------------------------------
float AGeneratedTerrain::GeneratedExtentCm() const
{
    // Same as extent_m = img_size * 60 in cmd_export, in cm
    return Generator->GetImgSize() * NativeMetresPerPixel * 100.f;
}

void AGeneratedTerrain::GenerateNew()
{
    if (!Generator->IsReady())
    {
        Status = TEXT("Model not loaded - check the Output Log");
        return;
    }
    if (Generator->IsBusy())
    {
        return;
    }

    LastSeed = FMath::Rand();
    PointOverviewCamera(GeneratedExtentCm(), Generator->GetReliefMetres() * 100.f);
    bPendingFreeze = true;            // Tick freezes the player and switches to the overview camera
    Generator->Generate(LastSeed);
}

void AGeneratedTerrain::HandleHeights(const TArray<float>& HeightsM, int32 N, bool bFinal)
{
    const float ExtentCm = GeneratedExtentCm();

    if (!bFinal)
    {
        // Live preview: low-res, no collision, fast to rebuild
        const TArray<float> Preview = FTerrainMeshBuilder::Resample(HeightsM, N, PreviewRes);
        FTerrainMeshBuilder::Build(Mesh, Preview, PreviewRes, ExtentCm / (PreviewRes - 1), Chunks, false, Material);
        return;
    }

    // Final: upsample, smooth, full-res walkable mesh
    TArray<float> Final = FTerrainMeshBuilder::Resample(HeightsM, N, FinalRes);
    Final = FTerrainMeshBuilder::Blur(Final, FinalRes, BlurSigma);
    BuildFinal(Final, FinalRes, ExtentCm / (FinalRes - 1));

    Status = FString::Printf(TEXT("Generated: seed %d, y %.2f, w %.1f (%s)"), LastSeed,
                             Generator->Relief, Generator->Guidance, *Generator->GetActiveRuntime());
}

void AGeneratedTerrain::BuildFinal(const TArray<float>& HeightsM, int32 N, float CellCm)
{
    FTerrainMeshBuilder::Build(Mesh, HeightsM, N, CellCm, Chunks, true, Material);

    // Height of the centre vertex above the lowest point (Build puts the lowest point at Z = 0)
    float MinH = HeightsM[0];
    for (const float H : HeightsM)
    {
        MinH = FMath::Min(MinH, H);
    }
    const int32 Mid = N / 2;
    SpawnHeightCm = (HeightsM[Mid * N + Mid] - MinH) * 100.f;
    bPendingPlacePlayer = true;
}

void AGeneratedTerrain::PointOverviewCamera(float ExtentCm, float ReliefCm)
{
    // High above one corner, looking at the middle of the terrain
    const float Half = ExtentCm * 0.5f;
    const FVector CamPos(-0.9f * Half, -0.9f * Half, FMath::Max(ReliefCm * 2.f, Half * 0.7f));
    const FVector Target(0.f, 0.f, ReliefCm * 0.3f);
    OverviewCamera->SetRelativeLocationAndRotation(CamPos, (Target - CamPos).Rotation());
}

// ---------------------------------------------------------------------------
// Exported heightmaps (sample.py export)
// ---------------------------------------------------------------------------
bool AGeneratedTerrain::LoadExport(int32 Index)
{
    if (Generator->IsBusy() || !ExportFiles.IsValidIndex(Index))
    {
        return false;
    }
    const FString& PngPath = ExportFiles[Index];
    const FString JsonPath = FPaths::ChangeExtension(PngPath, TEXT("json"));

    // 1. Metadata: real height range and ground spacing
    FString JsonText;
    TSharedPtr<FJsonObject> Meta;
    if (!FFileHelper::LoadFileToString(JsonText, *JsonPath))
    {
        UE_LOG(LogTemp, Error, TEXT("GameDiT: missing %s"), *JsonPath);
        return false;
    }
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonText);
    if (!FJsonSerializer::Deserialize(Reader, Meta) || !Meta.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("GameDiT: could not parse %s"), *JsonPath);
        return false;
    }
    const float RangeM = float(Meta->GetNumberField(TEXT("height_range_m")));
    const float MetresPerPixel = float(Meta->GetNumberField(TEXT("metres_per_pixel")));

    // 2. 16-bit grayscale PNG -> raw pixels
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *PngPath))
    {
        UE_LOG(LogTemp, Error, TEXT("GameDiT: could not read %s"), *PngPath);
        return false;
    }
    IImageWrapperModule& ImageModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
    TSharedPtr<IImageWrapper> Png = ImageModule.CreateImageWrapper(EImageFormat::PNG);
    TArray64<uint8> Raw;
    if (!Png.IsValid() || !Png->SetCompressed(Bytes.GetData(), Bytes.Num()) || !Png->GetRaw(ERGBFormat::Gray, 16, Raw))
    {
        UE_LOG(LogTemp, Error, TEXT("GameDiT: could not decode %s as 16-bit grayscale"), *PngPath);
        return false;
    }
    const int32 Width = int32(Png->GetWidth());
    const int32 Height = int32(Png->GetHeight());
    if (Width != Height || Width < 2)
    {
        UE_LOG(LogTemp, Error, TEXT("GameDiT: %s is %dx%d, expected square"), *PngPath, Width, Height);
        return false;
    }

    // 3. uint16 -> metres (inverse of the u16 normalisation in cmd_export)
    const int32 N = Width;
    const uint16* Pixels = reinterpret_cast<const uint16*>(Raw.GetData());
    TArray<float> HeightsM;
    HeightsM.SetNumUninitialized(N * N);
    for (int32 i = 0; i < N * N; ++i)
    {
        HeightsM[i] = Pixels[i] / 65535.f * RangeM;
    }

    // 4. Build at FinalRes, keeping the real ground extent
    const float ExtentCm = MetresPerPixel * 100.f * (N - 1);
    if (N != FinalRes)
    {
        HeightsM = FTerrainMeshBuilder::Resample(HeightsM, N, FinalRes);
    }
    BuildFinal(HeightsM, FinalRes, ExtentCm / (FinalRes - 1));

    ExportIndex = Index;
    Status = FString::Printf(TEXT("Export %d/%d: %s (%.0f m relief)"), Index + 1, ExportFiles.Num(),
                             *FPaths::GetCleanFilename(PngPath), RangeM);
    return true;
}

void AGeneratedTerrain::NextExport()
{
    if (ExportFiles.Num() > 0)
    {
        LoadExport((ExportIndex + 1) % ExportFiles.Num());
    }
}

// ---------------------------------------------------------------------------
// Player and camera
// ---------------------------------------------------------------------------
void AGeneratedTerrain::TryFreezePlayer(APlayerController* PC)
{
    ACharacter* Character = Cast<ACharacter>(PC->GetPawn());
    if (!Character)
    {
        return;   // pawn not spawned yet: Tick tries again next frame
    }
    // The old terrain's collision disappears during previews, so stop the player falling
    Character->GetCharacterMovement()->DisableMovement();
    PC->SetViewTargetWithBlend(this, 1.0f);   // view through OverviewCamera
    bPendingFreeze = false;
}

void AGeneratedTerrain::TryPlacePlayer(APlayerController* PC)
{
    APawn* Pawn = PC->GetPawn();
    if (!Pawn)
    {
        return;
    }
    const FVector Spawn = GetActorLocation() + FVector(0.f, 0.f, SpawnHeightCm + 300.f);
    Pawn->SetActorLocation(Spawn, false, nullptr, ETeleportType::TeleportPhysics);

    if (ACharacter* Character = Cast<ACharacter>(Pawn))
    {
        Character->GetCharacterMovement()->StopMovementImmediately();
        Character->GetCharacterMovement()->SetMovementMode(MOVE_Falling);   // drops 3 m onto the ground
    }
    PC->SetViewTargetWithBlend(Pawn, 1.0f);   // back to the third-person camera
    bPendingPlacePlayer = false;
}

void AGeneratedTerrain::HandleInput(APlayerController* PC)
{
    // Polling keys directly works the same with or without Enhanced Input
    if (PC->WasInputKeyJustPressed(EKeys::G)) { GenerateNew(); }
    if (PC->WasInputKeyJustPressed(EKeys::N)) { NextExport(); }
    if (PC->WasInputKeyJustPressed(EKeys::One))   { Generator->Relief = 0.3f; }   // hills
    if (PC->WasInputKeyJustPressed(EKeys::Two))   { Generator->Relief = 0.6f; }   // rugged
    if (PC->WasInputKeyJustPressed(EKeys::Three)) { Generator->Relief = 0.9f; }   // alpine
    if (PC->WasInputKeyJustPressed(EKeys::V))
    {
        Generator->bPreviewPrediction = !Generator->bPreviewPrediction;
    }
}

void AGeneratedTerrain::ShowStatus() const
{
    if (!GEngine)
    {
        return;
    }
    const FString Help = FString::Printf(
        TEXT("[G] generate   [1/2/3] relief y=%.1f (~%.0f m)   [V] preview: %s   [N] next export"),
        Generator->Relief, Generator->GetReliefMetres(),
        Generator->bPreviewPrediction ? TEXT("model's guess (x1_hat)") : TEXT("raw state (x_t)"));

    const FString Line = Generator->IsBusy()
        ? FString::Printf(TEXT("Generating... step %d / %d on %s"), Generator->GetCurrentStep(),
                          Generator->Steps, *Generator->GetActiveRuntime())
        : Status;

    // Same key each frame = the line is replaced, not stacked; 0 s = shown for this frame only
    GEngine->AddOnScreenDebugMessage(1, 0.f, FColor::White, Help);
    GEngine->AddOnScreenDebugMessage(2, 0.f, FColor::Yellow, Line);
}