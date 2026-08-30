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
        if (!parent.name.ToLower().Contains("velocity"))
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
                if (bone == null || bone.name.ToLower().Contains("velocity")) continue;

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

    [ContextMenu("Force Export CSV")]
    public void ExportToCSV()
    {
        try
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string targetFolder = Path.Combine(projectRoot, "Data");

            if (!Directory.Exists(targetFolder))
                Directory.CreateDirectory(targetFolder);

            string filePath = Path.Combine(targetFolder, csvFileName);

            StringBuilder sb = new StringBuilder();

            // Cabecalho do CSV
            sb.AppendLine("Letter,TimeStamp,BoneName,PosX,PosY,PosZ,RotX,RotY,RotZ,RotW");

            // Itera sobre as letras e gravacoes
            foreach (var dataset in recordedData)
            {
                foreach (var frame in dataset.capturedFrames)
                {
                    foreach (var bone in frame.bonesData)
                    {
                        if (bone.boneName.ToLower().Contains("velocity")) continue;

                        sb.AppendLine(string.Format(
                            CultureInfo.InvariantCulture,
                            "{0},{1:F4},{2},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6},{8:F6},{9:F6}",
                            dataset.letter,
                            frame.timeStamp,
                            bone.boneName,
                            bone.localPosition.x,
                            bone.localPosition.y,
                            bone.localPosition.z,
                            bone.localRotation.x,
                            bone.localRotation.y,
                            bone.localRotation.z,
                            bone.localRotation.w
                        ));
                    }
                }
            }

            File.WriteAllText(filePath, sb.ToString());

#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif

            Debug.Log($"<color=green>[DataRecorder] CSV salvo com SUCESSO em:</color> {filePath}");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"<color=red>[DataRecorder] ERRO ao salvar CSV:</color> {ex.Message}");
        }
    }
}