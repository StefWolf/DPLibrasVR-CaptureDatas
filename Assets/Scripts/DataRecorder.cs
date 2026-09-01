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

    private List<Transform> allBones = new List<Transform>();
    private bool isRecording = false;

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
        if (!ShouldIgnoreBone(parent.name))
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



    private IEnumerator CaptureRoutine()
    {
        isRecording = true;

        int validCaptures = Mathf.Max(1, targetCapturesCount);
        float intervalInSeconds = captureDurationSeconds / validCaptures;

        LetterDataset dataset = new LetterDataset { letter = currentLetter.ToString() };

        for (int i = 0; i < validCaptures; i++)
        {
            float currentTime = i * intervalInSeconds;

            if (statusText != null)
                statusText.text = $"Capturing letter {currentLetter}... ({i + 1}/{validCaptures})";

            HandFrameData frame = new HandFrameData { timeStamp = currentTime };

            foreach (Transform bone in allBones)
            {
                if (bone == null || ShouldIgnoreBone(bone.name)) continue;

                frame.bonesData.Add(new BoneTransformData
                {
                    boneName = bone.name,
                    localPosition = bone.localPosition,
                    localRotation = bone.localRotation
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

        int newChar = currentLetter + step;
        if (newChar < 'A') newChar = 'Z';
        if (newChar > 'Z') newChar = 'A';
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

    private bool ShouldIgnoreBone(string boneName)
    {
        if (string.IsNullOrEmpty(boneName)) return true;
        string n = boneName.ToLower();
        return n.Contains("velocity") || n.Contains("tip");
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
                        if (ShouldIgnoreBone(bone.boneName)) continue;
                        if (seen.Add(bone.boneName))
                            boneOrder.Add(bone.boneName);
                    }

            StringBuilder sb = new StringBuilder();

            // 2) Cabecalho: 7 colunas por osso (features) + Letter no FINAL.
            StringBuilder header = new StringBuilder();
            for (int i = 0; i < boneOrder.Count; i++)
            {
                string b = boneOrder[i].Replace(",", "_");
                if (i > 0) header.Append(',');
                header.Append($"{b}_PosX,{b}_PosY,{b}_PosZ,{b}_RotX,{b}_RotY,{b}_RotZ,{b}_RotW");
            }
            header.Append(",Letter");
            sb.AppendLine(header.ToString());

            // 3) Uma linha por frame: features..., Letter.
            foreach (var dataset in recordedData)
            {
                foreach (var frame in dataset.capturedFrames)
                {
                    Dictionary<string, BoneTransformData> boneMap =
                        new Dictionary<string, BoneTransformData>();
                    foreach (var bone in frame.bonesData)
                    {
                        if (ShouldIgnoreBone(bone.boneName)) continue;
                        boneMap[bone.boneName] = bone;
                    }

                    StringBuilder row = new StringBuilder();
                    for (int i = 0; i < boneOrder.Count; i++)
                    {
                        if (i > 0) row.Append(',');
                        if (boneMap.TryGetValue(boneOrder[i], out BoneTransformData bone))
                        {
                            row.Append(string.Format(CultureInfo.InvariantCulture,
                                "{0:F6},{1:F6},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6}",
                                bone.localPosition.x, bone.localPosition.y, bone.localPosition.z,
                                bone.localRotation.x, bone.localRotation.y,
                                bone.localRotation.z, bone.localRotation.w));
                        }
                        else
                        {
                            row.Append(",,,,,,"); // 6 virgulas = 7 campos vazios
                        }
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
                      $"({boneOrder.Count} ossos x 7 = {boneOrder.Count * 7} features + 1 label)");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"<color=red>[DataRecorder] ERRO ao salvar CSV:</color> {ex.Message}");
        }
    }
}