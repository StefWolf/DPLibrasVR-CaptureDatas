using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using System.Text;
using System.Globalization;

public class DataRecorder : MonoBehaviour
{
    [System.Serializable]
    public class BoneTransformData
    {
        public string boneName;
        public Vector3 localPosition;
        public Quaternion localRotation;
    }
    [System.Serializable]
    public class HandFrameData
    {
        public float timeStamp;
        public List<BoneTransformData> bonesData = new List<BoneTransformData>();
    }
    [System.Serializable]
    public class LetterDataset
    {
        public string letter;
        public List<HandFrameData> capturedFrames = new List<HandFrameData>();
    }

    [SerializeField] private TextMeshProUGUI statusText;
    [SerializeField] private List<Transform> rootBones = new List<Transform>();
    [SerializeField] private float captureDurationSeconds = 10f;
    [SerializeField] private int targetCapturesCount = 100;
    [SerializeField] private char currentLetter = 'A';
    [SerializeField] private string csvFileName = "LibrasDataset.csv";
    [SerializeField] private List<LetterDataset> recordedData = new List<LetterDataset>();
    [SerializeField] private int delayToStart = 3;

    private List<Transform> allBones = new List<Transform>();
    private bool isRecording = false;

    // Letras que NAO devem ser capturadas (tem movimento em Libras).
    private static readonly HashSet<char> ignoredLetters = new HashSet<char>
        { 'H', 'J', 'K', 'W', 'X', 'Y', 'Z' };

    private void Start()
    {
        InitializeBones();
        UpdateStatusText();
    }

    public void InitializeBones()
    {
        allBones.Clear();
        foreach (Transform root in rootBones)
        {
            if (root != null)
                CollectBonesRecursive(root);
        }
    }

    private void CollectBonesRecursive(Transform parent)
    {
        if (!LibrasBoneConfig.ShouldIgnoreBone(parent.name))
        {
            allBones.Add(parent);
        }
        foreach (Transform child in parent)
        {
            CollectBonesRecursive(child);
        }
    }

    public void StartRecording()
    {
        if (!isRecording)
        {
            InitializeBones();
            StartCoroutine(CaptureRoutine());
        }
    }

    private IEnumerator CaptureRoutine(){
        isRecording = true;

        for (int i = delayToStart; i > 0; i--){
            if (statusText != null)
                statusText.text = $"Gravando em {i}..."; 
            yield return new WaitForSeconds(1f);
        }

        int validCaptures = Mathf.Max(1, targetCapturesCount);
        float intervalInSeconds = captureDurationSeconds / validCaptures;
        LetterDataset dataset = new LetterDataset { letter = currentLetter.ToString() };
        for (int i = 0; i < validCaptures; i++)
        {
            float currentTime = i * intervalInSeconds;
            if (statusText != null)
                statusText.text = $"Capturing letter {currentLetter}... ({i + 1}/{validCaptures})";
            HandFrameData frame = new HandFrameData { timeStamp = currentTime };
            Transform wristTransform = rootBones[0];
            foreach (Transform bone in allBones)
            {
                if (bone == null || LibrasBoneConfig.ShouldIgnoreBone(bone.name)) continue;

                Vector3 relativePos = wristTransform.InverseTransformPoint(bone.position);
                Quaternion relativeRot = Quaternion.Inverse(wristTransform.rotation) * bone.rotation;

                if (bone == wristTransform)
                    relativeRot = LibrasBoneConfig.ExtractSwing(relativeRot, Vector3.forward);

                frame.bonesData.Add(new BoneTransformData
                {
                    boneName = bone.name,
                    localPosition = relativePos,
                    localRotation = relativeRot
                });
            }
            dataset.capturedFrames.Add(frame);
            yield return new WaitForSeconds(intervalInSeconds);
        }
        int existingIndex = recordedData.FindIndex(d => d.letter == currentLetter.ToString());
        if (existingIndex >= 0)
            recordedData[existingIndex] = dataset;
        else
            recordedData.Add(dataset);
        isRecording = false;
        NextLetter();
    }

    public void ChangeLetter(int step)
    {
        if (isRecording) return;

        int direction = step >= 0 ? 1 : -1;
        int newChar = currentLetter;

        // Avanca de 1 em 1 na direcao pedida ate cair numa letra valida (com wraparound).
        do
        {
            newChar += direction;
            if (newChar < 'A') newChar = 'Z';
            if (newChar > 'Z') newChar = 'A';
        } while (ignoredLetters.Contains((char)newChar));

        currentLetter = (char)newChar;
        UpdateStatusText();
    }

    public void NextLetter() => ChangeLetter(1);
    public void PreviousLetter() => ChangeLetter(-1);

    public void UpdateStatusText()
    {
        if (statusText != null)
            statusText.text = $"Ready to capture letter {currentLetter}";
    }

    private static float GetComponent(BoneTransformData bone, int comp)
    {
        switch (comp)
        {
            case 0: return bone.localPosition.x;
            case 1: return bone.localPosition.y;
            case 2: return bone.localPosition.z;
            case 3: return bone.localRotation.x;
            case 4: return bone.localRotation.y;
            case 5: return bone.localRotation.z;
            case 6: return bone.localRotation.w;
        }
        return 0f;
    }

    [ContextMenu("Force Export CSV")]
    public void ExportToCSV()
    {
        try
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string targetFolder = Path.Combine(projectRoot, "Assets", "Data");
            if (!Directory.Exists(targetFolder))
                Directory.CreateDirectory(targetFolder);
            string filePath = Path.Combine(targetFolder, csvFileName);

            // 1) Ordem canonica dos ossos (estavel, na ordem de 1a aparicao).
            List<string> boneOrder = new List<string>();
            HashSet<string> seen = new HashSet<string>();
            foreach (var dataset in recordedData)
                foreach (var frame in dataset.capturedFrames)
                    foreach (var bone in frame.bonesData)
                    {
                        if (LibrasBoneConfig.ShouldIgnoreBone(bone.boneName)) continue;
                        if (seen.Add(bone.boneName))
                            boneOrder.Add(bone.boneName);
                    }

            List<(string bone, int comp, string col)> features = new List<(string, int, string)>();
            foreach (string boneName in boneOrder)
            {
                string b = boneName.Replace(",", "_");
                for (int c = 0; c < LibrasBoneConfig.ComponentSuffixes.Length; c++)
                {
                    string col = $"{b}_{LibrasBoneConfig.ComponentSuffixes[c]}";
                    if (LibrasBoneConfig.IgnoredColumns.Contains(col)) continue;
                    features.Add((boneName, c, col));
                }
            }

            StringBuilder sb = new StringBuilder();

            // 3) Cabecalho: apenas as features mantidas + Letter no FINAL.
            StringBuilder header = new StringBuilder();
            for (int i = 0; i < features.Count; i++)
            {
                if (i > 0) header.Append(',');
                header.Append(features[i].col);
            }
            header.Append(",Letter");
            sb.AppendLine(header.ToString());

            // 4) Uma linha por frame: features mantidas..., Letter.
            foreach (var dataset in recordedData)
            {
                foreach (var frame in dataset.capturedFrames)
                {
                    Dictionary<string, BoneTransformData> boneMap = new Dictionary<string, BoneTransformData>();
                    foreach (var bone in frame.bonesData)
                    {
                        if (LibrasBoneConfig.ShouldIgnoreBone(bone.boneName)) continue;
                        boneMap[bone.boneName] = bone;
                    }
                    StringBuilder row = new StringBuilder();
                    for (int i = 0; i < features.Count; i++)
                    {
                        if (i > 0) row.Append(',');
                        if (boneMap.TryGetValue(features[i].bone, out BoneTransformData bone))
                        {
                            float value = GetComponent(bone, features[i].comp);
                            row.Append(value.ToString("F6", CultureInfo.InvariantCulture));
                        }
                        // senao deixa o campo vazio (osso ausente naquele frame)
                    }
                    row.Append(',').Append(dataset.letter); // Letter por ultimo
                    sb.AppendLine(row.ToString());
                }
            }

            File.WriteAllText(filePath, sb.ToString());
#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif
            Debug.Log($"<color=green>[DataRecorder] CSV salvo com SUCESSO em:</color> {filePath} " +
                      $"({features.Count} features mantidas + 1 label.)");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"<color=red>[DataRecorder] ERRO ao salvar CSV:</color> {ex.Message}");
        }
    }
}