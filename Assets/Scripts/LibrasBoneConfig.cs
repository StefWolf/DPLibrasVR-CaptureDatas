using System.Collections.Generic;
using UnityEngine;

public static class LibrasBoneConfig
{
    // Sufixos das 7 features geradas por osso, na ordem de escrita no CSV/modelo.
    public static readonly string[] ComponentSuffixes =
        { "PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ", "RotW" };

    // Colunas constantes/desnecessarias a serem ignoradas.
    public static readonly HashSet<string> IgnoredColumns = new HashSet<string>
    {
        "R_IndexMetacarpal_RotX", "R_IndexMetacarpal_RotY", "R_IndexMetacarpal_RotZ", "R_IndexMetacarpal_RotW",
        "R_IndexIntermediate_PosX", "R_IndexIntermediate_PosY", "R_IndexIntermediate_PosZ",
        "R_IndexDistal_PosX", "R_IndexDistal_PosY", "R_IndexDistal_PosZ",

        "R_LittleMetacarpal_PosX", "R_LittleMetacarpal_PosY", "R_LittleMetacarpal_PosZ",
        "R_LittleMetacarpal_RotX", "R_LittleMetacarpal_RotY", "R_LittleMetacarpal_RotZ", "R_LittleMetacarpal_RotW",
        "R_LittleProximal_PosX", "R_LittleProximal_PosY", "R_LittleProximal_PosZ",
        "R_LittleIntermediate_PosX", "R_LittleIntermediate_PosY", "R_LittleIntermediate_PosZ",
        "R_LittleDistal_PosX", "R_LittleDistal_PosY", "R_LittleDistal_PosZ",

        "R_MiddleMetacarpal_RotX", "R_MiddleMetacarpal_RotY", "R_MiddleMetacarpal_RotZ", "R_MiddleMetacarpal_RotW",
        "R_MiddleIntermediate_PosX", "R_MiddleIntermediate_PosY", "R_MiddleIntermediate_PosZ",
        "R_MiddleDistal_PosX", "R_MiddleDistal_PosY", "R_MiddleDistal_PosZ",

        "R_Palm_RotX", "R_Palm_RotY", "R_Palm_RotZ", "R_Palm_RotW",

        "R_RingMetacarpal_PosX", "R_RingMetacarpal_PosY", "R_RingMetacarpal_PosZ",
        "R_RingMetacarpal_RotX", "R_RingMetacarpal_RotY", "R_RingMetacarpal_RotZ", "R_RingMetacarpal_RotW",
        "R_RingProximal_PosX", "R_RingProximal_PosY", "R_RingProximal_PosZ",
        "R_RingIntermediate_PosX", "R_RingIntermediate_PosY", "R_RingIntermediate_PosZ",
        "R_RingDistal_PosX", "R_RingDistal_PosY", "R_RingDistal_PosZ",

        "R_ThumbProximal_PosX", "R_ThumbProximal_PosY", "R_ThumbProximal_PosZ",
        "R_ThumbDistal_PosX", "R_ThumbDistal_PosY", "R_ThumbDistal_PosZ",
        "R_ThumbDistal_RotX", "R_ThumbDistal_RotY", "R_ThumbDistal_RotZ", "R_ThumbDistal_RotW",
    };

    /// <summary>
    /// Verifica se o osso deve ser totalmente ignorado na coleta do esqueleto.
    /// </summary>
    public static bool ShouldIgnoreBone(string boneName)
    {
        if (string.IsNullOrEmpty(boneName)) return true;
        string n = boneName.ToLower();
        return n.Contains("velocity") || n.Contains("tip");
    }

    /// <summary>
    /// Retorna o valor escalar de um componente especifico de um Transform.
    /// (0..2: PosX, PosY, PosZ | 3..6: RotX, RotY, RotZ, RotW)
    /// </summary>
    public static float GetComponentValue(Transform bone, int componentIndex)
    {
        if (bone == null) return 0f;

        switch (componentIndex)
        {
            case 0: return bone.localPosition.x;
            case 1: return bone.localPosition.y;
            case 2: return bone.localPosition.z;
            case 3: return bone.localRotation.x;
            case 4: return bone.localRotation.y;
            case 5: return bone.localRotation.z;
            case 6: return bone.localRotation.w;
            default: return 0f;
        }
    }
}